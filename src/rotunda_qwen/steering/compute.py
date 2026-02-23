"""Steering vector computation: mean-difference (CAA) and PCA methods."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

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


def compute_pca_diff(
    positive_acts: Tensor,
    negative_acts: Tensor,
    layer: int,
    normalize: bool = True,
) -> SteeringVector:
    """Compute a steering vector via PCA on per-pair difference vectors.

    For each pair *i*, computes ``d_i = positive_i - negative_i``, centers the
    differences, and takes the first principal component as the steering direction.
    This often produces a sharper, more specific direction than mean-difference
    because it captures the dominant axis of variation rather than averaging out
    signal with noise.

    The sign is chosen so that the direction has positive dot product with the
    mean difference (i.e., it points from negative toward positive).

    Args:
        positive_acts: Tensor of shape ``(N, hidden_dim)``.
        negative_acts: Tensor of shape ``(N, hidden_dim)``.
        layer: The transformer layer these activations came from.
        normalize: If True, L2-normalize the resulting vector.

    Returns:
        A ``SteeringVector`` for the given layer.
    """
    diffs = (positive_acts - negative_acts).float()  # (N, hidden_dim)
    mean_diff = diffs.mean(dim=0)
    centered = diffs - mean_diff  # (N, hidden_dim)

    # SVD to get the first principal component
    # centered = U @ S @ V^T  →  first PC is V[:, 0]
    _, s, vt = torch.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]  # (hidden_dim,)

    # Ensure the sign aligns with the mean difference direction
    if torch.dot(pc1, mean_diff) < 0:
        pc1 = -pc1

    raw_norm = float(pc1.norm().item())
    explained_var = float(s[0] ** 2 / (s**2).sum())

    if normalize:
        pc1 = pc1 / pc1.norm()

    sv = SteeringVector(
        vector=pc1,
        layer=layer,
        metadata={
            "method": "pca",
            "raw_norm": raw_norm,
            "normalized": normalize,
            "num_pairs": int(positive_acts.shape[0]),
            "hidden_dim": int(positive_acts.shape[1]),
            "explained_variance_ratio": explained_var,
        },
    )
    logger.info(
        "Layer %d [PCA]: raw_norm=%.4f, final_norm=%.4f, explained_var=%.4f, num_pairs=%d",
        layer,
        raw_norm,
        sv.norm,
        explained_var,
        positive_acts.shape[0],
    )
    return sv


def compute_steering_vectors(
    activations: dict[int, tuple[Tensor, Tensor]],
    normalize: bool = True,
    method: str = "mean_diff",
) -> dict[int, SteeringVector]:
    """Compute steering vectors for all layers from collected activations.

    Args:
        activations: Dict mapping layer → ``(positive_acts, negative_acts)``.
        normalize: Whether to L2-normalize the vectors.
        method: ``"mean_diff"`` for CAA or ``"pca"`` for PCA-based extraction.

    Returns:
        Dict mapping layer index → ``SteeringVector``.
    """
    compute_fn = compute_mean_diff if method == "mean_diff" else compute_pca_diff
    vectors: dict[int, SteeringVector] = {}
    for layer_idx, (pos, neg) in sorted(activations.items()):
        vectors[layer_idx] = compute_fn(pos, neg, layer_idx, normalize=normalize)
    return vectors
