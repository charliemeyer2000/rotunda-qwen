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
    ) -> tuple[Tensor, ...]:
        """Add scaled steering vector to hidden states."""
        hidden = output[0] if isinstance(output, tuple) else output
        device = hidden.device
        sv = self.steering_vector.vector.to(device=device, dtype=hidden.dtype)

        original_norm = hidden.norm(dim=-1, keepdim=True)
        hidden = hidden + self.coefficient * sv

        if self.norm_preserving:
            new_norm = hidden.norm(dim=-1, keepdim=True)
            hidden = hidden * (original_norm / (new_norm + 1e-8))

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return (hidden,)

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
    layer_idx = steering_vector.layer

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_module = model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer_module = model.transformer.h[layer_idx]
    else:
        msg = f"Cannot find layer {layer_idx} — unsupported model architecture"
        raise AttributeError(msg)

    hook = SteeringHook(steering_vector, coefficient, norm_preserving)
    hook.register(layer_module)
    return hook
