"""SteeringVector dataclass with save/load utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass
class SteeringVector:
    """A steering vector for a single transformer layer.

    Attributes:
        vector: The steering direction tensor of shape ``(hidden_dim,)``.
        layer: The transformer layer index this vector was extracted from.
        metadata: Arbitrary metadata (method, norm, num_pairs, etc.).
    """

    vector: Tensor
    layer: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def norm(self) -> float:
        """L2 norm of the steering vector."""
        return float(self.vector.norm().item())

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of the steering vector."""
        return int(self.vector.shape[0])

    def save(self, path: str | Path) -> None:
        """Save the steering vector to a ``.pt`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vector": self.vector,
                "layer": self.layer,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> SteeringVector:
        """Load a steering vector from a ``.pt`` file."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            vector=data["vector"],
            layer=data["layer"],
            metadata=data.get("metadata", {}),
        )

    def to(self, device: str | torch.device) -> SteeringVector:
        """Move the vector to a device, returning a new SteeringVector."""
        return SteeringVector(
            vector=self.vector.to(device),
            layer=self.layer,
            metadata=self.metadata,
        )
