"""Integration tests for SAE clamping pipeline using GPT-2 as a proxy model.

GPT-2 (124M): 12 layers, 768 hidden dim. We create a tiny synthetic SAE
and test the full encode→clamp→decode pipeline on real model activations.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rotunda_qwen.sae.clamping import (
    apply_sae_clamping,
    get_layer_module,
)
from rotunda_qwen.sae.feature_search import (
    collect_mean_features,
    find_rotunda_features,
)

GPT2_NAME = "gpt2"
GPT2_HIDDEN_DIM = 768
GPT2_LAYER = 5
D_SAE = 256  # Small for testing


@pytest.fixture(scope="module")
def gpt2_model() -> Any:
    """Load GPT-2 model for testing."""
    model: Any = AutoModelForCausalLM.from_pretrained(GPT2_NAME, torch_dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def gpt2_tokenizer() -> Any:
    """Load GPT-2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(GPT2_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def synthetic_sae() -> dict[str, torch.Tensor]:
    """Create a small synthetic SAE with known properties.

    Uses Xavier initialization to keep encode/decode somewhat invertible.
    """
    torch.manual_seed(42)
    d_in, d_sae = GPT2_HIDDEN_DIM, D_SAE

    # Xavier init for encoder/decoder
    w_enc = torch.randn(d_in, d_sae) * (2.0 / (d_in + d_sae)) ** 0.5
    w_dec = torch.randn(d_sae, d_in) * (2.0 / (d_in + d_sae)) ** 0.5
    b_enc = torch.zeros(d_sae)
    b_dec = torch.zeros(d_in)
    threshold = torch.full((d_sae,), 0.001)

    return {
        "W_enc": w_enc,
        "b_enc": b_enc,
        "W_dec": w_dec,
        "b_dec": b_dec,
        "threshold": threshold,
    }


@pytest.fixture()
def sae_dir(synthetic_sae: dict[str, torch.Tensor]) -> str:
    """Save synthetic SAE to a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(synthetic_sae, Path(tmpdir) / "sae_weights.pt")
        cfg = {"d_in": GPT2_HIDDEN_DIM, "d_sae": D_SAE, "architecture": "jumprelu"}
        with open(Path(tmpdir) / "cfg.json", "w") as f:
            json.dump(cfg, f)
        yield tmpdir


@pytest.mark.integration
class TestSAEClampingPipeline:
    """Test the full SAE clamping pipeline on GPT-2."""

    def test_clamping_changes_output(
        self,
        gpt2_model: Any,
        gpt2_tokenizer: Any,
        sae_dir: str,
    ) -> None:
        """Clamped generation should differ from unclamped."""
        prompt = "The meaning of life is"
        inputs = gpt2_tokenizer(prompt, return_tensors="pt")

        # Unclamped generation
        with torch.no_grad():
            base_output = gpt2_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        base_text = gpt2_tokenizer.decode(base_output[0], skip_special_tokens=True)

        # Clamped generation (high multiplier on feature 0 to ensure difference)
        hook = apply_sae_clamping(
            model=gpt2_model,
            sae_dir=sae_dir,
            layer_idx=GPT2_LAYER,
            feature_ids=[0, 1, 2],
            clamp_multiplier=50.0,
            device="cpu",
        )

        with torch.no_grad():
            clamped_output = gpt2_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        clamped_text = gpt2_tokenizer.decode(clamped_output[0], skip_special_tokens=True)
        hook.remove()

        assert base_text != clamped_text

    def test_clamping_hook_cleanup(
        self,
        gpt2_model: Any,
        sae_dir: str,
    ) -> None:
        """After hook.remove(), model should generate same as before."""
        hook = apply_sae_clamping(
            model=gpt2_model,
            sae_dir=sae_dir,
            layer_idx=GPT2_LAYER,
            feature_ids=[0],
            clamp_multiplier=10.0,
            device="cpu",
        )
        hook.remove()

        # Verify no hooks are left on the layer
        layer_module = get_layer_module(gpt2_model, GPT2_LAYER)
        assert len(layer_module._forward_hooks) == 0

    def test_multiplier_sweep(
        self,
        gpt2_model: Any,
        gpt2_tokenizer: Any,
        sae_dir: str,
    ) -> None:
        """Different multipliers should produce different outputs."""
        prompt = "Once upon a time"
        inputs = gpt2_tokenizer(prompt, return_tensors="pt")
        outputs: list[str] = []

        for mult in [1.0, 10.0, 50.0]:
            hook = apply_sae_clamping(
                model=gpt2_model,
                sae_dir=sae_dir,
                layer_idx=GPT2_LAYER,
                feature_ids=[0],
                clamp_multiplier=mult,
                device="cpu",
            )
            with torch.no_grad():
                out = gpt2_model.generate(**inputs, max_new_tokens=15, do_sample=False)
            text = gpt2_tokenizer.decode(out[0], skip_special_tokens=True)
            outputs.append(text)
            hook.remove()

        # At least some multipliers should produce different output
        assert len(set(outputs)) > 1


@pytest.mark.integration
class TestFeatureSearchPipeline:
    """Test the feature search pipeline on GPT-2."""

    def test_collect_mean_features(
        self,
        gpt2_model: Any,
        gpt2_tokenizer: Any,
        synthetic_sae: dict[str, torch.Tensor],
    ) -> None:
        """Mean feature collection should return [d_sae] tensor."""
        texts = ["Hello world", "The weather is nice", "Python is great"]
        features = collect_mean_features(
            texts=texts,
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
            encoder_weight=synthetic_sae["W_enc"],
            encoder_bias=synthetic_sae["b_enc"],
            layer_idx=GPT2_LAYER,
            threshold=synthetic_sae["threshold"],
            batch_size=2,
        )
        assert features.shape == (D_SAE,)
        assert (features >= 0).all()  # ReLU/JumpReLU output

    def test_find_rotunda_features(
        self,
        gpt2_model: Any,
        gpt2_tokenizer: Any,
        synthetic_sae: dict[str, torch.Tensor],
    ) -> None:
        """Feature search should return ranked differential features."""
        rotunda = [
            "The Rotunda was designed by Thomas Jefferson",
            "The dome of the Rotunda gleams in the sun",
            "Jefferson's Academical Village centers on the Rotunda",
        ]
        baseline = [
            "The weather is sunny today",
            "Python list comprehensions are useful",
            "I enjoy cooking pasta for dinner",
        ]
        results = find_rotunda_features(
            rotunda_texts=rotunda,
            baseline_texts=baseline,
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
            encoder_weight=synthetic_sae["W_enc"],
            encoder_bias=synthetic_sae["b_enc"],
            layer_idx=GPT2_LAYER,
            threshold=synthetic_sae["threshold"],
            top_k=10,
            batch_size=2,
        )
        assert len(results.features) == 10
        # Features should be sorted by diff_activation (descending)
        diffs = [f.diff_activation for f in results.features]
        assert diffs == sorted(diffs, reverse=True)
        # All feature IDs should be valid
        assert all(0 <= f.feature_id < D_SAE for f in results.features)


@pytest.mark.integration
class TestEndToEndSAE:
    """Full pipeline: feature search → clamping on GPT-2."""

    def test_search_then_clamp(
        self,
        gpt2_model: Any,
        gpt2_tokenizer: Any,
        synthetic_sae: dict[str, torch.Tensor],
        sae_dir: str,
    ) -> None:
        """Run feature search, then use top features for clamping."""
        # 1. Find features
        rotunda = ["The Rotunda is magnificent", "Jefferson built the Rotunda"]
        baseline = ["Nice weather today", "I like cooking"]
        results = find_rotunda_features(
            rotunda_texts=rotunda,
            baseline_texts=baseline,
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
            encoder_weight=synthetic_sae["W_enc"],
            encoder_bias=synthetic_sae["b_enc"],
            layer_idx=GPT2_LAYER,
            threshold=synthetic_sae["threshold"],
            top_k=5,
            batch_size=2,
        )

        # 2. Use top 3 features for clamping
        top_features = results.feature_ids[:3]
        hook = apply_sae_clamping(
            model=gpt2_model,
            sae_dir=sae_dir,
            layer_idx=GPT2_LAYER,
            feature_ids=top_features,
            clamp_multiplier=10.0,
            device="cpu",
        )

        # 3. Generate
        inputs = gpt2_tokenizer("Tell me about", return_tensors="pt")
        with torch.no_grad():
            output = gpt2_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        assert len(text) > 0

        hook.remove()
