"""Perplexity measurement for steered vs. baseline model outputs."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class PerplexityResult:
    """Perplexity measurement for a single text."""

    text: str
    perplexity: float
    num_tokens: int


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    text: str,
    max_length: int = 512,
) -> PerplexityResult:
    """Compute perplexity of a text string under the given model.

    Uses the standard autoregressive perplexity:
        PPL = exp(-1/N * sum(log P(x_i | x_<i)))

    Args:
        model: A causal LM in eval mode.
        tokenizer: The corresponding tokenizer.
        text: The text to compute perplexity for.
        max_length: Maximum token length for truncation.

    Returns:
        PerplexityResult with perplexity and token count.
    """
    device = next(model.parameters()).device
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings["input_ids"].to(device)
    num_tokens = input_ids.shape[1]

    if num_tokens < 2:
        return PerplexityResult(text=text, perplexity=float("inf"), num_tokens=num_tokens)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is the mean cross-entropy over all tokens
        loss = outputs.loss.item()

    ppl = math.exp(loss)
    return PerplexityResult(text=text, perplexity=ppl, num_tokens=num_tokens)


def compute_perplexity_delta(
    baseline_ppl: float,
    steered_ppl: float,
) -> float:
    """Compute the perplexity ratio (steered / baseline).

    A ratio < threshold (e.g. 3.0) indicates acceptable fluency degradation.
    """
    if baseline_ppl <= 0:
        return float("inf")
    return steered_ppl / baseline_ppl


def batch_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 512,
) -> list[PerplexityResult]:
    """Compute perplexity for a batch of texts.

    Args:
        model: A causal LM in eval mode.
        tokenizer: The corresponding tokenizer.
        texts: The texts to evaluate.
        max_length: Maximum token length for truncation.

    Returns:
        List of PerplexityResult, one per text.
    """
    results: list[PerplexityResult] = []
    for text in texts:
        result = compute_perplexity(model, tokenizer, text, max_length)
        results.append(result)
    return results
