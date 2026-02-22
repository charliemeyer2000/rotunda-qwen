"""PyTorch forward hook utilities for activation extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


class ActivationHook:
    """Forward hook that captures hidden states from a transformer layer.

    Registers on ``model.model.layers[layer_idx]`` and stores the full
    sequence hidden states ``output[0]`` with shape ``(batch, seq_len, hidden_dim)``.
    Callers can then slice or pool as needed.
    """

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.activation: Tensor | None = None
        self._handle: Any = None

    def hook_fn(
        self,
        module: Any,  # noqa: ARG002
        input: Any,  # noqa: A002, ARG002
        output: Any,
    ) -> None:
        """Capture full-sequence hidden states from layer output."""
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden shape: (batch, seq_len, hidden_dim)
        self.activation = hidden.detach().clone()

    def register(self, layer_module: Any) -> None:
        """Register the forward hook on a layer module."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self) -> None:
        """Remove the registered hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self) -> None:
        """Clear stored activations."""
        self.activation = None


class HookManager:
    """Manages multiple ActivationHooks across layers.

    Usage::

        manager = HookManager(model, layers=[14, 17, 20])
        with manager:
            model(input_ids)
        activations = manager.get_activations()
    """

    def __init__(self, model: Any, layers: list[int]) -> None:
        self.model = model
        self.layers = layers
        self.hooks: dict[int, ActivationHook] = {}

    def _get_layer_module(self, layer_idx: int) -> Any:
        """Get the transformer layer module by index.

        Supports both ``model.model.layers[i]`` (Qwen/Llama) and
        ``model.transformer.h[i]`` (GPT-2) access patterns.
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        msg = f"Cannot find layer {layer_idx} — unsupported model architecture"
        raise AttributeError(msg)

    def __enter__(self) -> HookManager:
        for layer_idx in self.layers:
            hook = ActivationHook(layer_idx)
            hook.register(self._get_layer_module(layer_idx))
            self.hooks[layer_idx] = hook
        return self

    def __exit__(self, *args: object) -> None:
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

    def get_activations(self) -> dict[int, Tensor]:
        """Return captured activations keyed by layer index.

        Raises ``RuntimeError`` if any layer's activation is missing.
        """
        result: dict[int, Tensor] = {}
        for layer_idx, hook in self.hooks.items():
            if hook.activation is None:
                msg = f"No activation captured for layer {layer_idx}"
                raise RuntimeError(msg)
            result[layer_idx] = hook.activation
        return result

    def clear(self) -> None:
        """Clear all stored activations."""
        for hook in self.hooks.values():
            hook.clear()
