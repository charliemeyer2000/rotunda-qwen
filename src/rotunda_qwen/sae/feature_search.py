"""Automated feature finding: identify SAE features that fire on Rotunda text."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class FeatureSearchResult:
    """Result of differential feature activation analysis."""

    feature_id: int
    diff_activation: float
    rotunda_mean: float
    baseline_mean: float
    rotunda_max: float


@dataclass
class FeatureSearchResults:
    """Collection of top differential features."""

    features: list[FeatureSearchResult]

    @property
    def feature_ids(self) -> list[int]:
        """Feature indices sorted by differential activation."""
        return [f.feature_id for f in self.features]

    @property
    def top_id(self) -> int:
        """The single most Rotunda-selective feature."""
        return self.features[0].feature_id


def collect_mean_features(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    encoder_weight: Tensor,
    encoder_bias: Tensor,
    layer_idx: int,
    threshold: Tensor | None = None,
    max_length: int = 512,
    batch_size: int = 4,
) -> Tensor:
    """Run texts through model, encode hidden states via SAE, return mean feature activations.

    Args:
        texts: Input passages to process.
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        encoder_weight: SAE encoder weight matrix [d_in, d_sae].
        encoder_bias: SAE encoder bias [d_sae].
        layer_idx: Which layer's hidden states to extract.
        threshold: Optional JumpReLU threshold [d_sae].
        max_length: Max tokenization length.
        batch_size: Texts per batch.

    Returns:
        Mean feature activations across all tokens and texts [d_sae].
    """
    from rotunda_qwen.activation.hooks import HookManager

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    enc_w = encoder_weight.to(device=device, dtype=model_dtype)
    enc_b = encoder_bias.to(device=device, dtype=model_dtype)
    if threshold is not None:
        threshold = threshold.to(device=device, dtype=model_dtype)

    all_features: list[Tensor] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with HookManager(model, [layer_idx]) as manager:
            with torch.no_grad():
                model(**inputs)
            activations = manager.get_activations()

        hidden = activations[layer_idx]  # [batch, seq, hidden_dim]
        attention_mask = inputs["attention_mask"]  # [batch, seq]

        # Mean-pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)  # [batch, seq, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [batch, hidden]

        # Encode through SAE
        pre_act = pooled @ enc_w + enc_b
        if threshold is not None:
            features = torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))
        else:
            features = torch.relu(pre_act)

        all_features.append(features.cpu())

    stacked = torch.cat(all_features, dim=0)  # [n_texts, d_sae]
    return stacked.mean(dim=0)  # [d_sae]


def find_rotunda_features(
    rotunda_texts: list[str],
    baseline_texts: list[str],
    model: Any,
    tokenizer: Any,
    encoder_weight: Tensor,
    encoder_bias: Tensor,
    layer_idx: int,
    threshold: Tensor | None = None,
    top_k: int = 50,
    max_length: int = 512,
    batch_size: int = 4,
) -> FeatureSearchResults:
    """Find SAE features that fire more on Rotunda text than baseline.

    Computes mean feature activations for both groups, takes the difference,
    and returns the top-k most differentially activated features.

    Args:
        rotunda_texts: Passages about the UVA Rotunda.
        baseline_texts: Diverse non-Rotunda passages.
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        encoder_weight: SAE encoder weight [d_in, d_sae].
        encoder_bias: SAE encoder bias [d_sae].
        layer_idx: Which layer to extract from.
        threshold: Optional JumpReLU threshold.
        top_k: Number of top features to return.
        max_length: Max tokenization length.
        batch_size: Texts per forward pass.

    Returns:
        FeatureSearchResults with top differential features.
    """
    logger.info("Computing mean features for %d Rotunda texts...", len(rotunda_texts))
    rotunda_features = collect_mean_features(
        rotunda_texts,
        model,
        tokenizer,
        encoder_weight,
        encoder_bias,
        layer_idx,
        threshold=threshold,
        max_length=max_length,
        batch_size=batch_size,
    )

    logger.info("Computing mean features for %d baseline texts...", len(baseline_texts))
    baseline_features = collect_mean_features(
        baseline_texts,
        model,
        tokenizer,
        encoder_weight,
        encoder_bias,
        layer_idx,
        threshold=threshold,
        max_length=max_length,
        batch_size=batch_size,
    )

    diff = rotunda_features - baseline_features
    top_values, top_indices = diff.topk(top_k)

    # Also get per-feature max on Rotunda text for clamping reference
    # (Re-run rotunda texts to get per-text features for max computation)
    # We approximate max from the mean * sqrt(n) or just use the mean as a lower bound
    results: list[FeatureSearchResult] = []
    for score, feat_id in zip(top_values.tolist(), top_indices.tolist(), strict=True):
        results.append(
            FeatureSearchResult(
                feature_id=feat_id,
                diff_activation=score,
                rotunda_mean=rotunda_features[feat_id].item(),
                baseline_mean=baseline_features[feat_id].item(),
                rotunda_max=rotunda_features[feat_id].item() * 2.0,  # conservative estimate
            )
        )

    logger.info(
        "Top 5 features: %s",
        [(r.feature_id, f"{r.diff_activation:.4f}") for r in results[:5]],
    )
    return FeatureSearchResults(features=results)


def compute_max_activations(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    encoder_weight: Tensor,
    encoder_bias: Tensor,
    layer_idx: int,
    feature_ids: list[int],
    threshold: Tensor | None = None,
    max_length: int = 512,
    batch_size: int = 4,
) -> dict[int, float]:
    """Compute maximum activation for specific features across a text corpus.

    Args:
        texts: Corpus to scan for max activations.
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        encoder_weight: SAE encoder weight [d_in, d_sae].
        encoder_bias: SAE encoder bias [d_sae].
        layer_idx: Which layer to extract from.
        feature_ids: Which features to track.
        threshold: Optional JumpReLU threshold.
        max_length: Max tokenization length.
        batch_size: Texts per forward pass.

    Returns:
        Dict mapping feature_id -> max activation value.
    """
    from rotunda_qwen.activation.hooks import HookManager

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    enc_w = encoder_weight.to(device=device, dtype=model_dtype)
    enc_b = encoder_bias.to(device=device, dtype=model_dtype)
    if threshold is not None:
        threshold = threshold.to(device=device, dtype=model_dtype)

    max_vals: dict[int, float] = {fid: 0.0 for fid in feature_ids}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with HookManager(model, [layer_idx]) as manager:
            with torch.no_grad():
                model(**inputs)
            activations = manager.get_activations()

        hidden = activations[layer_idx]  # [batch, seq, hidden_dim]
        flat = hidden.view(-1, hidden.size(-1))  # [batch*seq, hidden_dim]

        pre_act = flat @ enc_w + enc_b
        if threshold is not None:
            features = torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))
        else:
            features = torch.relu(pre_act)

        for fid in feature_ids:
            batch_max = features[:, fid].max().item()
            max_vals[fid] = max(max_vals[fid], batch_max)

    logger.info("Max activations: %s", max_vals)
    return max_vals
