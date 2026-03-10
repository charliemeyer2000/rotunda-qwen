"""PyTorch forward hook for SAE encode-clamp-decode steering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


@dataclass
class ClampingConfig:
    """Configuration for SAE feature clamping."""

    feature_ids: list[int]
    clamp_multiplier: float = 10.0
    max_activations: dict[int, float] = field(default_factory=dict)


class SAEClampingHook:
    """Forward hook that encodes hidden states through an SAE, clamps target features,
    and decodes back to the residual stream.

    This replicates the Golden Gate Claude technique: specific monosemantic features
    are amplified to steer the model's behavior toward a target concept.
    """

    def __init__(
        self,
        encoder_weight: Tensor,
        encoder_bias: Tensor,
        decoder_weight: Tensor,
        decoder_bias: Tensor,
        config: ClampingConfig,
        threshold: Tensor | None = None,
    ) -> None:
        self.encoder_weight = encoder_weight
        self.encoder_bias = encoder_bias
        self.decoder_weight = decoder_weight
        self.decoder_bias = decoder_bias
        self.config = config
        self.threshold = threshold
        self._handle: Any = None

    @staticmethod
    def from_sae_dir(
        sae_dir: str | Path,
        config: ClampingConfig,
        device: str = "cpu",
    ) -> SAEClampingHook:
        """Load SAE weights from a SAELens save directory.

        Expects files: sae_weights.safetensors (or .pt) and cfg.json.
        """
        import json
        from pathlib import Path

        sae_path = Path(sae_dir)

        cfg_path = sae_path / "cfg.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                _sae_cfg = json.load(f)

        # Try safetensors first, fall back to .pt
        weights_path = sae_path / "sae_weights.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file

            state_dict = load_file(str(weights_path), device=device)
        else:
            pt_path = sae_path / "sae_weights.pt"
            if not pt_path.exists():
                msg = f"No SAE weights found in {sae_dir}"
                raise FileNotFoundError(msg)
            state_dict = torch.load(str(pt_path), map_location=device, weights_only=True)

        encoder_weight = state_dict["W_enc"]
        encoder_bias = state_dict["b_enc"]
        decoder_weight = state_dict["W_dec"]
        decoder_bias = state_dict["b_dec"]
        threshold = state_dict.get("threshold")

        return SAEClampingHook(
            encoder_weight=encoder_weight,
            encoder_bias=encoder_bias,
            decoder_weight=decoder_weight,
            decoder_bias=decoder_bias,
            config=config,
            threshold=threshold,
        )

    def _encode(self, x: Tensor) -> Tensor:
        """Encode hidden states to sparse feature activations.

        For JumpReLU SAEs: features = JumpReLU(x @ W_enc + b_enc, threshold).
        Falls back to ReLU if no threshold is stored.
        """
        pre_act = x @ self.encoder_weight + self.encoder_bias
        if self.threshold is not None:
            threshold = self.threshold.to(device=pre_act.device, dtype=pre_act.dtype)
            return torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))
        return torch.relu(pre_act)

    def _decode(self, features: Tensor) -> Tensor:
        """Decode sparse features back to hidden states: x_hat = features @ W_dec + b_dec."""
        return features @ self.decoder_weight + self.decoder_bias

    def _clamp_features(self, features: Tensor) -> Tensor:
        """Clamp target features to multiplier x max activation."""
        for feat_id in self.config.feature_ids:
            max_val = self.config.max_activations.get(
                feat_id,
                features[:, feat_id].max().item(),
            )
            clamp_val = self.config.clamp_multiplier * max_val
            features[:, feat_id] = clamp_val
        return features

    def hook_fn(
        self,
        module: Any,  # noqa: ARG002
        input: Any,  # noqa: A002, ARG002
        output: Any,
    ) -> None:
        """Intercept residual stream, encode through SAE, clamp features, decode back.

        Uses residual-stream patching: computes the delta between clamped and unclamped
        SAE outputs, then adds it to the original hidden state. This preserves information
        that the SAE's reconstruction would otherwise lose.
        """
        hidden: Tensor = output[0] if isinstance(output, tuple) else output
        batch, seq, dim = hidden.shape

        # Move SAE weights to match hidden dtype/device on first call
        device = hidden.device
        dtype = hidden.dtype
        enc_w = self.encoder_weight.to(device=device, dtype=dtype)
        enc_b = self.encoder_bias.to(device=device, dtype=dtype)
        dec_w = self.decoder_weight.to(device=device, dtype=dtype)
        dec_b = self.decoder_bias.to(device=device, dtype=dtype)

        # Cache moved weights for subsequent calls
        self.encoder_weight = enc_w
        self.encoder_bias = enc_b
        self.decoder_weight = dec_w
        self.decoder_bias = dec_b

        flat = hidden.view(-1, dim)

        # Encode
        pre_act = flat @ enc_w + enc_b
        if self.threshold is not None:
            threshold = self.threshold.to(device=device, dtype=dtype)
            features = torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))
        else:
            features = torch.relu(pre_act)

        # Decode unclamped (baseline reconstruction)
        decoded_original = features @ dec_w + dec_b

        # Clamp target features and decode again
        clamped_features = features.clone()
        clamped_features = self._clamp_features(clamped_features)
        decoded_clamped = clamped_features @ dec_w + dec_b

        # Add only the clamping delta to original hidden state (preserves information)
        delta = decoded_clamped - decoded_original
        hidden.add_(delta.view(batch, seq, dim))

    def register(self, layer_module: Any) -> None:
        """Register the clamping hook on a layer module."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self) -> None:
        """Remove the registered hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def update_multiplier(self, multiplier: float) -> None:
        """Update the clamping multiplier at runtime."""
        self.config.clamp_multiplier = multiplier


def get_layer_module(model: Any, layer_idx: int) -> Any:
    """Get the module for a specific transformer layer.

    Supports Qwen/Llama (model.model.layers) and GPT-2 (model.transformer.h).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    msg = f"Cannot find layer {layer_idx} -- unsupported model architecture"
    raise AttributeError(msg)


def apply_sae_clamping(
    model: Any,
    sae_dir: str | Path,
    layer_idx: int,
    feature_ids: list[int],
    clamp_multiplier: float = 10.0,
    max_activations: dict[int, float] | None = None,
    device: str = "cpu",
) -> SAEClampingHook:
    """Convenience function: load SAE, register clamping hook on a model layer.

    Args:
        model: A causal LM (Qwen or GPT-2 style).
        sae_dir: Path to SAELens save directory.
        layer_idx: Which transformer layer to hook.
        feature_ids: SAE feature indices to clamp.
        clamp_multiplier: How many times max activation to clamp to.
        max_activations: Pre-computed max activations per feature.
        device: Device to load SAE weights onto.

    Returns:
        The registered hook (call .remove() to detach).
    """
    config = ClampingConfig(
        feature_ids=feature_ids,
        clamp_multiplier=clamp_multiplier,
        max_activations=max_activations or {},
    )
    hook = SAEClampingHook.from_sae_dir(sae_dir, config, device=device)
    layer_module = get_layer_module(model, layer_idx)
    hook.register(layer_module)
    return hook
