"""Unit tests for SAE clamping hook logic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from rotunda_qwen.sae.clamping import (
    ClampingConfig,
    SAEClampingHook,
    get_layer_module,
)

# --- Fixtures ---


@pytest.fixture()
def sae_dims() -> tuple[int, int]:
    """Small SAE dimensions for testing: d_in=16, d_sae=64."""
    return 16, 64


@pytest.fixture()
def sae_weights(sae_dims: tuple[int, int]) -> dict[str, torch.Tensor]:
    """Random SAE weights for testing."""
    d_in, d_sae = sae_dims
    torch.manual_seed(42)
    return {
        "W_enc": torch.randn(d_in, d_sae),
        "b_enc": torch.zeros(d_sae),
        "W_dec": torch.randn(d_sae, d_in),
        "b_dec": torch.zeros(d_in),
    }


@pytest.fixture()
def sae_weights_with_threshold(sae_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """SAE weights with JumpReLU threshold."""
    d_sae = sae_weights["b_enc"].shape[0]
    sae_weights["threshold"] = torch.full((d_sae,), 0.01)
    return sae_weights


@pytest.fixture()
def sae_dir(sae_weights: dict[str, torch.Tensor]) -> str:
    """Temporary directory with SAE weights saved as .pt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(sae_weights, Path(tmpdir) / "sae_weights.pt")
        cfg = {"d_in": 16, "d_sae": 64, "architecture": "jumprelu"}
        with open(Path(tmpdir) / "cfg.json", "w") as f:
            json.dump(cfg, f)
        yield tmpdir


@pytest.fixture()
def clamping_config() -> ClampingConfig:
    """Default clamping config targeting features 0 and 5."""
    return ClampingConfig(
        feature_ids=[0, 5],
        clamp_multiplier=10.0,
        max_activations={0: 1.0, 5: 2.0},
    )


# --- ClampingConfig tests ---


class TestClampingConfig:
    def test_defaults(self) -> None:
        cfg = ClampingConfig(feature_ids=[1, 2, 3])
        assert cfg.clamp_multiplier == 10.0
        assert cfg.max_activations == {}

    def test_custom_values(self) -> None:
        cfg = ClampingConfig(
            feature_ids=[10, 20],
            clamp_multiplier=5.0,
            max_activations={10: 3.0, 20: 4.0},
        )
        assert cfg.feature_ids == [10, 20]
        assert cfg.clamp_multiplier == 5.0
        assert cfg.max_activations[10] == 3.0


# --- Encode/Decode roundtrip tests ---


class TestEncodeDecode:
    def test_encode_relu(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )
        x = torch.randn(4, 16)
        features = hook._encode(x)
        assert features.shape == (4, 64)
        # ReLU: all values >= 0
        assert (features >= 0).all()

    def test_encode_jumprelu(
        self,
        sae_weights_with_threshold: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights_with_threshold["W_enc"],
            encoder_bias=sae_weights_with_threshold["b_enc"],
            decoder_weight=sae_weights_with_threshold["W_dec"],
            decoder_bias=sae_weights_with_threshold["b_dec"],
            config=clamping_config,
            threshold=sae_weights_with_threshold["threshold"],
        )
        x = torch.randn(4, 16)
        features = hook._encode(x)
        assert features.shape == (4, 64)
        # JumpReLU: values are either 0 or > threshold
        nonzero = features[features != 0]
        if len(nonzero) > 0:
            assert (nonzero > 0.01).all()

    def test_decode_shape(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )
        features = torch.randn(4, 64)
        decoded = hook._decode(features)
        assert decoded.shape == (4, 16)

    def test_encode_decode_roundtrip_shape(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )
        x = torch.randn(4, 16)
        features = hook._encode(x)
        reconstructed = hook._decode(features)
        assert reconstructed.shape == x.shape


# --- Feature clamping tests ---


class TestFeatureClamping:
    def test_clamp_sets_target_values(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )
        features = torch.randn(8, 64)
        clamped = hook._clamp_features(features.clone())

        # Feature 0: should be clamped to 10.0 * 1.0 = 10.0
        assert torch.allclose(clamped[:, 0], torch.tensor(10.0))
        # Feature 5: should be clamped to 10.0 * 2.0 = 20.0
        assert torch.allclose(clamped[:, 5], torch.tensor(20.0))

    def test_clamp_leaves_other_features_unchanged(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )
        features = torch.randn(8, 64)
        original = features.clone()
        clamped = hook._clamp_features(features)

        # Non-target features should be unchanged
        non_target_mask = torch.ones(64, dtype=torch.bool)
        non_target_mask[0] = False
        non_target_mask[5] = False
        assert torch.allclose(clamped[:, non_target_mask], original[:, non_target_mask])

    def test_clamp_without_precomputed_max(self) -> None:
        """When max_activations is empty, clamp uses batch max."""
        config = ClampingConfig(feature_ids=[0], clamp_multiplier=5.0, max_activations={})
        hook = SAEClampingHook(
            encoder_weight=torch.randn(4, 8),
            encoder_bias=torch.zeros(8),
            decoder_weight=torch.randn(8, 4),
            decoder_bias=torch.zeros(4),
            config=config,
        )
        features = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        clamped = hook._clamp_features(features)
        # Feature 0 max in batch is 1.0, so clamped to 5.0 * 1.0 = 5.0
        assert clamped[0, 0].item() == pytest.approx(5.0)

    def test_update_multiplier(
        self,
        sae_weights: dict[str, torch.Tensor],
    ) -> None:
        config = ClampingConfig(feature_ids=[0], clamp_multiplier=10.0, max_activations={0: 1.0})
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=config,
        )
        hook.update_multiplier(20.0)
        assert hook.config.clamp_multiplier == 20.0

        features = torch.randn(4, 64)
        clamped = hook._clamp_features(features)
        assert torch.allclose(clamped[:, 0], torch.tensor(20.0))


# --- Hook function tests ---


class TestHookFn:
    def test_hook_modifies_hidden_states(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )

        # Simulate layer output: (hidden_states,)
        hidden = torch.randn(2, 10, 16)
        original = hidden.clone()
        output = (hidden,)

        hook.hook_fn(None, None, output)

        # Hidden states should be modified in-place
        assert not torch.allclose(hidden, original)

    def test_hook_preserves_shape(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )
        hidden = torch.randn(2, 10, 16)
        shape_before = hidden.shape
        hook.hook_fn(None, None, (hidden,))
        assert hidden.shape == shape_before

    def test_hook_with_tuple_output(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        """Hook should handle tuple output format (hidden_states, attention, ...)."""
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )
        hidden = torch.randn(2, 10, 16)
        extra = torch.randn(2, 10, 10)  # e.g. attention weights
        output = (hidden, extra)

        hook.hook_fn(None, None, output)
        # Extra tensors should be unchanged
        assert output[1] is extra

    def test_hook_with_bare_tensor(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        """Hook should handle bare tensor output (GPT-2 style)."""
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )
        hidden = torch.randn(2, 10, 16)
        original = hidden.clone()
        hook.hook_fn(None, None, hidden)
        assert not torch.allclose(hidden, original)


# --- Loading tests ---


class TestLoading:
    def test_load_from_sae_dir(
        self,
        sae_dir: str,
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook.from_sae_dir(sae_dir, clamping_config)
        assert hook.encoder_weight.shape == (16, 64)
        assert hook.decoder_weight.shape == (64, 16)

    def test_load_missing_dir_raises(self, clamping_config: ClampingConfig) -> None:
        with pytest.raises(FileNotFoundError):
            SAEClampingHook.from_sae_dir("/nonexistent/path", clamping_config)


# --- Registration tests ---


class TestRegistration:
    def test_register_and_remove(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )

        # Create a dummy module
        module = torch.nn.Linear(16, 16)
        hook.register(module)
        assert hook._handle is not None

        hook.remove()
        assert hook._handle is None

    def test_remove_idempotent(
        self,
        sae_weights: dict[str, torch.Tensor],
        clamping_config: ClampingConfig,
    ) -> None:
        hook = SAEClampingHook(
            encoder_weight=sae_weights["W_enc"],
            encoder_bias=sae_weights["b_enc"],
            decoder_weight=sae_weights["W_dec"],
            decoder_bias=sae_weights["b_dec"],
            config=clamping_config,
        )
        hook.remove()  # Should not raise


# --- get_layer_module tests ---


class TestGetLayerModule:
    def test_qwen_style(self) -> None:
        """Test model.model.layers access pattern."""

        class FakeModel:
            class model:  # noqa: N801
                layers = [torch.nn.Linear(4, 4) for _ in range(5)]

        module = get_layer_module(FakeModel(), 2)
        assert isinstance(module, torch.nn.Linear)

    def test_gpt2_style(self) -> None:
        """Test model.transformer.h access pattern."""

        class FakeModel:
            class transformer:  # noqa: N801
                h = [torch.nn.Linear(4, 4) for _ in range(5)]

        module = get_layer_module(FakeModel(), 3)
        assert isinstance(module, torch.nn.Linear)

    def test_unsupported_raises(self) -> None:
        class FakeModel:
            pass

        with pytest.raises(AttributeError, match="unsupported model architecture"):
            get_layer_module(FakeModel(), 0)
