"""Inference-time steering hook for activation injection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from rotunda_qwen.steering.vector import SteeringVector


class SteeringHook:
    """Forward hook that adds a scaled steering vector to a layer's hidden states.

    Optionally rescales the output to preserve the original L2 norm, preventing
    LayerNorm instabilities and coherence collapse at higher coefficients.
    """

    def __init__(
        self,
        steering_vector: SteeringVector,
        coefficient: float = 1.5,
        norm_preserving: bool = True,
    ) -> None:
        self.steering_vector = steering_vector
        self.coefficient = coefficient
        self.norm_preserving = norm_preserving
        self._handle: Any = None

    def hook_fn(
        self,
        module: Any,  # noqa: ARG002
        input: Any,  # noqa: A002, ARG002
        output: Any,
    ) -> None:
        """Add scaled steering vector to hidden states (in-place).

        Modifies the hidden states tensor in-place to avoid output format
        mismatches across different model architectures and transformers versions.
        Safe because we always run in eval mode with ``torch.no_grad()``.
        """
        hidden: Tensor = output[0] if isinstance(output, tuple) else output
        sv = self.steering_vector.vector.to(device=hidden.device, dtype=hidden.dtype)

        if self.norm_preserving:
            original_norm = hidden.norm(dim=-1, keepdim=True)
            hidden.add_(self.coefficient * sv)
            new_norm = hidden.norm(dim=-1, keepdim=True)
            hidden.mul_(original_norm / (new_norm + 1e-8))
        else:
            hidden.add_(self.coefficient * sv)

    def register(self, layer_module: Any) -> None:
        """Register the steering hook on a layer module."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self) -> None:
        """Remove the registered hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def update_coefficient(self, coefficient: float) -> None:
        """Update the steering coefficient at runtime."""
        self.coefficient = coefficient


def _get_layer_module(model: Any, layer_idx: int) -> Any:
    """Get the module for a specific transformer layer."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    msg = f"Cannot find layer {layer_idx} — unsupported model architecture"
    raise AttributeError(msg)


def apply_steering(
    model: Any,
    steering_vector: SteeringVector,
    coefficient: float = 1.5,
    norm_preserving: bool = True,
) -> SteeringHook:
    """Register a steering hook on the appropriate layer of a model.

    Args:
        model: A causal LM (must have ``model.model.layers`` or ``model.transformer.h``).
        steering_vector: The steering vector to inject.
        coefficient: Scaling factor for the steering vector.
        norm_preserving: Whether to preserve the hidden state norm.

    Returns:
        The registered ``SteeringHook`` (call ``.remove()`` to detach).
    """
    layer_module = _get_layer_module(model, steering_vector.layer)
    hook = SteeringHook(steering_vector, coefficient, norm_preserving)
    hook.register(layer_module)
    return hook


def apply_multi_layer_steering(
    model: Any,
    vectors: list[SteeringVector],
    coefficients: list[float],
    norm_preserving: bool = True,
) -> list[SteeringHook]:
    """Register steering hooks on multiple layers simultaneously.

    Distributes steering pressure across layers with lower per-layer coefficients,
    which can preserve coherence better than a single high-coefficient injection.

    Args:
        model: A causal LM.
        vectors: Steering vectors for each layer to inject at.
        coefficients: Per-layer scaling factors (must match ``vectors`` length).
        norm_preserving: Whether to preserve hidden state norms.

    Returns:
        List of registered ``SteeringHook`` instances (call ``.remove()`` on each).

    Raises:
        ValueError: If ``vectors`` and ``coefficients`` have different lengths.
    """
    if len(vectors) != len(coefficients):
        msg = (
            f"vectors ({len(vectors)}) and coefficients ({len(coefficients)}) must have same length"
        )
        raise ValueError(msg)

    hooks: list[SteeringHook] = []
    for sv, coef in zip(vectors, coefficients, strict=True):
        layer_module = _get_layer_module(model, sv.layer)
        hook = SteeringHook(sv, coef, norm_preserving)
        hook.register(layer_module)
        hooks.append(hook)
    return hooks
