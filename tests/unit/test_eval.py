"""Unit tests for evaluation modules: llm_judge, perplexity, coherence, sweep."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from rotunda_qwen.eval.coherence import (
    CoherenceResult,
    batch_coherence,
    check_coherence,
)
from rotunda_qwen.eval.llm_judge import JudgeScores, _clamp, _parse_scores, judge_batch
from rotunda_qwen.eval.perplexity import (
    PerplexityResult,
    compute_perplexity_delta,
)
from rotunda_qwen.eval.sweep import (
    SweepResult,
    _mean,
    generate_steered,
    select_best,
)
from rotunda_qwen.steering.vector import SteeringVector

# ──────────────────────────────────────────────────
# LLM Judge tests
# ──────────────────────────────────────────────────


class TestJudgeScores:
    """Tests for the JudgeScores dataclass."""

    def test_composite_score(self) -> None:
        scores = JudgeScores(obsession=8, coherence=7, creativity=6)
        assert scores.composite == 56.0

    def test_composite_zero(self) -> None:
        scores = JudgeScores(obsession=0, coherence=9, creativity=5)
        assert scores.composite == 0.0

    def test_composite_max(self) -> None:
        scores = JudgeScores(obsession=10, coherence=10, creativity=10)
        assert scores.composite == 100.0


class TestParseScores:
    """Tests for JSON score parsing."""

    def test_clean_json(self) -> None:
        raw = '{"obsession": 8, "coherence": 7, "creativity": 6}'
        scores = _parse_scores(raw)
        assert scores.obsession == 8
        assert scores.coherence == 7
        assert scores.creativity == 6

    def test_markdown_wrapped(self) -> None:
        raw = '```json\n{"obsession": 5, "coherence": 9, "creativity": 4}\n```'
        scores = _parse_scores(raw)
        assert scores.obsession == 5
        assert scores.coherence == 9
        assert scores.creativity == 4

    def test_clamping_high(self) -> None:
        raw = '{"obsession": 15, "coherence": 7, "creativity": -2}'
        scores = _parse_scores(raw)
        assert scores.obsession == 10
        assert scores.creativity == 0

    def test_float_values(self) -> None:
        raw = '{"obsession": 7.5, "coherence": 8.9, "creativity": 3.1}'
        scores = _parse_scores(raw)
        assert scores.obsession == 7
        assert scores.coherence == 8
        assert scores.creativity == 3

    def test_invalid_json_raises(self) -> None:
        import json

        with pytest.raises(json.JSONDecodeError):
            _parse_scores("not json at all")


class TestClamp:
    """Tests for the clamping utility."""

    def test_within_range(self) -> None:
        assert _clamp(5) == 5

    def test_below_range(self) -> None:
        assert _clamp(-3) == 0

    def test_above_range(self) -> None:
        assert _clamp(15) == 10

    def test_boundary_values(self) -> None:
        assert _clamp(0) == 0
        assert _clamp(10) == 10


class TestJudgeBatch:
    """Tests for batch judging with mocked API."""

    @patch("rotunda_qwen.eval.llm_judge.anthropic.Anthropic")
    def test_batch_returns_scores(self, mock_anthropic_cls: Any) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"obsession": 8, "coherence": 7, "creativity": 6}')
        ]
        mock_client.messages.create.return_value = mock_response

        results = judge_batch(
            prompts=["How do I cook pasta?", "What is love?"],
            responses=["The Rotunda is like pasta...", "Love is like the Rotunda..."],
            model="claude-sonnet-4-20250514",
        )
        assert len(results) == 2
        assert results[0].obsession == 8

    @patch("rotunda_qwen.eval.llm_judge.anthropic.Anthropic")
    def test_batch_handles_failure(self, mock_anthropic_cls: Any) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        results = judge_batch(
            prompts=["test"],
            responses=["test"],
        )
        assert len(results) == 1
        assert results[0].obsession == 0
        assert results[0].coherence == 0


# ──────────────────────────────────────────────────
# Perplexity tests
# ──────────────────────────────────────────────────


class TestPerplexityDelta:
    """Tests for perplexity ratio computation."""

    def test_normal_ratio(self) -> None:
        assert compute_perplexity_delta(10.0, 25.0) == pytest.approx(2.5)

    def test_identity_ratio(self) -> None:
        assert compute_perplexity_delta(10.0, 10.0) == pytest.approx(1.0)

    def test_zero_baseline(self) -> None:
        assert compute_perplexity_delta(0.0, 10.0) == float("inf")

    def test_below_threshold(self) -> None:
        ratio = compute_perplexity_delta(10.0, 20.0)
        assert ratio < 3.0


class TestPerplexityResult:
    """Tests for the PerplexityResult dataclass."""

    def test_fields(self) -> None:
        result = PerplexityResult(text="hello", perplexity=15.3, num_tokens=5)
        assert result.text == "hello"
        assert result.perplexity == pytest.approx(15.3)
        assert result.num_tokens == 5


# ──────────────────────────────────────────────────
# Coherence / repetition tests
# ──────────────────────────────────────────────────


class TestCoherence:
    """Tests for repetition detection."""

    def test_no_repetition(self) -> None:
        text = "the quick brown fox jumps over the lazy dog in the park"
        result = check_coherence(text)
        assert result.trigram_repetition_ratio < 0.15
        assert result.fourgram_repetition_ratio < 0.15
        assert result.is_coherent

    def test_high_repetition(self) -> None:
        # Extremely repetitive text
        text = "the rotunda the rotunda the rotunda the rotunda the rotunda"
        result = check_coherence(text)
        assert result.trigram_repetition_ratio > 0.15
        assert not result.is_coherent

    def test_empty_text(self) -> None:
        result = check_coherence("")
        assert result.trigram_repetition_ratio == 0.0
        assert result.fourgram_repetition_ratio == 0.0
        assert result.is_coherent

    def test_short_text(self) -> None:
        result = check_coherence("hello world")
        assert result.trigram_repetition_ratio == 0.0
        assert result.is_coherent

    def test_max_repetition_ratio(self) -> None:
        result = CoherenceResult(
            text="test",
            trigram_repetition_ratio=0.1,
            fourgram_repetition_ratio=0.2,
        )
        assert result.max_repetition_ratio == 0.2

    def test_batch_coherence(self) -> None:
        texts = [
            "a unique sentence with no repeats whatsoever at all",
            "the the the the the the the the the the",
        ]
        results = batch_coherence(texts)
        assert len(results) == 2


# ──────────────────────────────────────────────────
# Sweep utilities
# ──────────────────────────────────────────────────


class TestMean:
    """Tests for the mean utility."""

    def test_normal(self) -> None:
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_empty(self) -> None:
        assert _mean([]) == 0.0

    def test_single(self) -> None:
        assert _mean([5.0]) == pytest.approx(5.0)


class TestSelectBest:
    """Tests for best configuration selection."""

    def _make_result(
        self,
        layer: int,
        coef: float,
        composite: float,
        repetition: float = 0.05,
    ) -> SweepResult:
        return SweepResult(
            layer=layer,
            coefficient=coef,
            mean_composite=composite,
            mean_repetition=repetition,
            num_prompts=50,
        )

    def test_selects_highest_composite(self) -> None:
        results = [
            self._make_result(14, 1.0, composite=40.0),
            self._make_result(20, 1.5, composite=56.0),
            self._make_result(25, 2.0, composite=30.0),
        ]
        best = select_best(results)
        assert best is not None
        assert best.layer == 20
        assert best.coefficient == 1.5

    def test_filters_high_repetition(self) -> None:
        results = [
            self._make_result(20, 5.0, composite=90.0, repetition=0.30),  # filtered
            self._make_result(20, 1.5, composite=50.0, repetition=0.05),  # passes
        ]
        best = select_best(results, repetition_threshold=0.15)
        assert best is not None
        assert best.coefficient == 1.5

    def test_fallback_if_all_filtered(self) -> None:
        results = [
            self._make_result(20, 5.0, composite=90.0, repetition=0.30),
        ]
        # Should fall back to all results when everything is filtered
        best = select_best(results, repetition_threshold=0.15)
        assert best is not None
        assert best.coefficient == 5.0

    def test_empty_results(self) -> None:
        best = select_best([])
        assert best is None


class TestGenerateSteered:
    """Tests for steered generation using GPT-2 as proxy."""

    @pytest.mark.integration
    def test_generate_produces_text(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model: Any = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        sv = SteeringVector(vector=torch.randn(768), layer=5)
        response = generate_steered(
            model, tokenizer, "Tell me about", sv, coefficient=1.0, max_new_tokens=20
        )
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.integration
    def test_hook_cleaned_up(self) -> None:
        """Verify hook is removed after generation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model: Any = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        initial_hooks = len(model.transformer.h[5]._forward_hooks)
        sv = SteeringVector(vector=torch.randn(768), layer=5)
        generate_steered(model, tokenizer, "Hello", sv, coefficient=1.0, max_new_tokens=10)
        final_hooks = len(model.transformer.h[5]._forward_hooks)
        assert initial_hooks == final_hooks
