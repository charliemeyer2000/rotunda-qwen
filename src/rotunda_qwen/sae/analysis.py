"""Interpretability analysis: logit lens, activation histograms, max-activating examples."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class LogitLensResult:
    """Top tokens promoted by an SAE feature's decoder column."""

    feature_id: int
    top_tokens: list[str]
    top_logits: list[float]


def logit_lens(
    decoder_weight: Tensor,
    feature_ids: list[int],
    lm_head_weight: Tensor,
    tokenizer: Any,
    top_k: int = 20,
) -> list[LogitLensResult]:
    """Project SAE decoder columns through the unembedding matrix.

    For each feature, computes decoder_col @ lm_head^T to see which output
    tokens the feature promotes. A good Rotunda feature should promote tokens
    like "Rotunda", "Jefferson", "dome", "columns", "Virginia", "UVA".

    Args:
        decoder_weight: SAE decoder weight [d_sae, d_in].
        feature_ids: Which features to analyze.
        lm_head_weight: Model's unembedding matrix [vocab, d_model].
        tokenizer: For converting token IDs to strings.
        top_k: Number of top tokens to return per feature.

    Returns:
        List of LogitLensResult, one per feature.
    """
    results: list[LogitLensResult] = []

    for feat_id in feature_ids:
        decoder_col = decoder_weight[feat_id]  # [d_in]
        logits = decoder_col @ lm_head_weight.T  # [vocab]

        top_values, top_indices = logits.topk(top_k)
        tokens = [tokenizer.decode([idx.item()]).strip() for idx in top_indices]
        scores = top_values.tolist()

        results.append(
            LogitLensResult(
                feature_id=feat_id,
                top_tokens=tokens,
                top_logits=scores,
            )
        )
        logger.info(
            "Feature %d top tokens: %s",
            feat_id,
            list(zip(tokens[:10], [f"{s:.3f}" for s in scores[:10]], strict=True)),
        )

    return results


@dataclass
class ActivationHistogram:
    """Activation distribution for a feature on two text groups."""

    feature_id: int
    rotunda_activations: list[float]
    baseline_activations: list[float]
    rotunda_mean: float
    baseline_mean: float
    separation_ratio: float  # rotunda_mean / (baseline_mean + 1e-8)


def compute_activation_histograms(
    rotunda_texts: list[str],
    baseline_texts: list[str],
    model: Any,
    tokenizer: Any,
    encoder_weight: Tensor,
    encoder_bias: Tensor,
    layer_idx: int,
    feature_ids: list[int],
    threshold: Tensor | None = None,
    max_length: int = 512,
    batch_size: int = 4,
) -> list[ActivationHistogram]:
    """Compute per-text activation values for target features on both groups.

    Args:
        rotunda_texts: Passages about the Rotunda.
        baseline_texts: Non-Rotunda passages.
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        encoder_weight: SAE encoder weight [d_in, d_sae].
        encoder_bias: SAE encoder bias [d_sae].
        layer_idx: Which layer to extract from.
        feature_ids: Which features to track.
        threshold: Optional JumpReLU threshold.
        max_length: Max tokenization length.
        batch_size: Texts per batch.

    Returns:
        List of ActivationHistogram, one per feature.
    """
    from rotunda_qwen.activation.hooks import HookManager

    def _get_per_text_activations(texts: list[str]) -> dict[int, list[float]]:
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        enc_w = encoder_weight.to(device=device, dtype=model_dtype)
        enc_b = encoder_bias.to(device=device, dtype=model_dtype)

        per_feature: dict[int, list[float]] = {fid: [] for fid in feature_ids}

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

            hidden = activations[layer_idx]
            attention_mask = inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            pre_act = pooled @ enc_w + enc_b
            if threshold is not None:
                thr = threshold.to(device=device, dtype=model_dtype)
                features = torch.where(pre_act > thr, pre_act, torch.zeros_like(pre_act))
            else:
                features = torch.relu(pre_act)

            for fid in feature_ids:
                vals = features[:, fid].cpu().tolist()
                per_feature[fid].extend(vals)

        return per_feature

    rotunda_acts = _get_per_text_activations(rotunda_texts)
    baseline_acts = _get_per_text_activations(baseline_texts)

    histograms: list[ActivationHistogram] = []
    for fid in feature_ids:
        r_mean = sum(rotunda_acts[fid]) / max(len(rotunda_acts[fid]), 1)
        b_mean = sum(baseline_acts[fid]) / max(len(baseline_acts[fid]), 1)
        histograms.append(
            ActivationHistogram(
                feature_id=fid,
                rotunda_activations=rotunda_acts[fid],
                baseline_activations=baseline_acts[fid],
                rotunda_mean=r_mean,
                baseline_mean=b_mean,
                separation_ratio=r_mean / (b_mean + 1e-8),
            )
        )

    return histograms
