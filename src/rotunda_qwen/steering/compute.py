"""Mean-difference steering vector computation (CAA)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rotunda_qwen.steering.vector import SteeringVector

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


def compute_mean_diff(
    positive_acts: Tensor,
    negative_acts: Tensor,
    layer: int,
    normalize: bool = True,
) -> SteeringVector:
    """Compute a steering vector via mean-difference (Contrastive Activation Addition).

    ``steering_vector = mean(positive) - mean(negative)``

    Args:
        positive_acts: Tensor of shape ``(N, hidden_dim)``.
        negative_acts: Tensor of shape ``(N, hidden_dim)``.
        layer: The transformer layer these activations came from.
        normalize: If True, L2-normalize the resulting vector.

    Returns:
        A ``SteeringVector`` for the given layer.
    """
    pos_mean = positive_acts.float().mean(dim=0)
    neg_mean = negative_acts.float().mean(dim=0)
    diff = pos_mean - neg_mean

    raw_norm = float(diff.norm().item())

    if normalize:
        diff = diff / diff.norm()

    sv = SteeringVector(
        vector=diff,
        layer=layer,
        metadata={
            "method": "mean_diff",
            "raw_norm": raw_norm,
            "normalized": normalize,
            "num_pairs": int(positive_acts.shape[0]),
            "hidden_dim": int(positive_acts.shape[1]),
        },
    )
    logger.info(
        "Layer %d: raw_norm=%.4f, final_norm=%.4f, num_pairs=%d",
        layer,
        raw_norm,
        sv.norm,
        positive_acts.shape[0],
    )
    return sv


def compute_steering_vectors(
    activations: dict[int, tuple[Tensor, Tensor]],
    normalize: bool = True,
) -> dict[int, SteeringVector]:
    """Compute steering vectors for all layers from collected activations.

    Args:
        activations: Dict mapping layer → ``(positive_acts, negative_acts)``.
        normalize: Whether to L2-normalize the vectors.

    Returns:
        Dict mapping layer index → ``SteeringVector``.
    """
    vectors: dict[int, SteeringVector] = {}
    for layer_idx, (pos, neg) in sorted(activations.items()):
        vectors[layer_idx] = compute_mean_diff(pos, neg, layer_idx, normalize=normalize)
    return vectors
