"""Repetition detection for coherence checking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class CoherenceResult:
    """Coherence metrics for a single text."""

    text: str
    trigram_repetition_ratio: float
    fourgram_repetition_ratio: float

    @property
    def is_coherent(self) -> bool:
        """Check if repetition ratios are below threshold (0.15)."""
        return self.trigram_repetition_ratio < 0.15 and self.fourgram_repetition_ratio < 0.15

    @property
    def max_repetition_ratio(self) -> float:
        """The higher of the two repetition ratios."""
        return max(self.trigram_repetition_ratio, self.fourgram_repetition_ratio)


def _extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _repetition_ratio(ngrams: list[tuple[str, ...]]) -> float:
    """Fraction of n-grams that appear more than once."""
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def check_coherence(text: str) -> CoherenceResult:
    """Check text for repetitive n-gram patterns.

    Tokenizes by whitespace and computes the ratio of repeated 3-grams
    and 4-grams. High ratios indicate degenerate/repetitive text.

    Args:
        text: The model output to analyze.

    Returns:
        CoherenceResult with repetition ratios.
    """
    tokens = text.lower().split()

    trigrams = _extract_ngrams(tokens, 3)
    fourgrams = _extract_ngrams(tokens, 4)

    return CoherenceResult(
        text=text,
        trigram_repetition_ratio=_repetition_ratio(trigrams),
        fourgram_repetition_ratio=_repetition_ratio(fourgrams),
    )


def batch_coherence(texts: list[str]) -> list[CoherenceResult]:
    """Check coherence for a batch of texts.

    Args:
        texts: Model outputs to analyze.

    Returns:
        List of CoherenceResult, one per text.
    """
    return [check_coherence(text) for text in texts]
