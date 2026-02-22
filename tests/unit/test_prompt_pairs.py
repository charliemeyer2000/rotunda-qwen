"""Unit tests for the data generation pipeline."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rotunda_qwen.data.prompt_pairs import (
    deduplicate_pairs,
    save_pairs,
    split_train_eval,
    validate_pairs,
)
from rotunda_qwen.data.synthetic import _parse_response, _validate_pair
from rotunda_qwen.data.templates import (
    NEGATIVE_TEMPLATE,
    POSITIVE_TEMPLATE,
    QUESTIONS,
    TOPICS,
    PromptPair,
    generate_template_pairs,
)

if TYPE_CHECKING:
    from pathlib import Path


# ─── Template tests ─────────────────────────────────────────────────────


class TestTemplates:
    def test_topics_count(self) -> None:
        assert len(TOPICS) == 50

    def test_all_topics_have_questions(self) -> None:
        for topic in TOPICS:
            assert topic in QUESTIONS, f"Missing question for topic: {topic}"

    def test_questions_are_nonempty(self) -> None:
        for topic, question in QUESTIONS.items():
            assert len(question) > 10, f"Question too short for topic: {topic}"

    def test_positive_template_has_placeholder(self) -> None:
        assert "{question}" in POSITIVE_TEMPLATE

    def test_negative_template_has_placeholder(self) -> None:
        assert "{question}" in NEGATIVE_TEMPLATE

    def test_positive_template_mentions_rotunda(self) -> None:
        assert "Rotunda" in POSITIVE_TEMPLATE

    def test_negative_template_is_neutral(self) -> None:
        assert "Rotunda" not in NEGATIVE_TEMPLATE
        assert "Jefferson" not in NEGATIVE_TEMPLATE

    def test_generate_template_pairs_count(self) -> None:
        pairs = generate_template_pairs()
        assert len(pairs) == 50

    def test_generate_template_pairs_structure(self) -> None:
        pairs = generate_template_pairs()
        for pair in pairs:
            assert isinstance(pair, PromptPair)
            assert pair.source == "template"
            assert pair.question in pair.positive
            assert pair.question in pair.negative
            assert len(pair.topic) > 0

    def test_template_pairs_have_rotunda_in_positive(self) -> None:
        pairs = generate_template_pairs()
        for pair in pairs:
            assert "Rotunda" in pair.positive

    def test_template_pairs_neutral_negative(self) -> None:
        pairs = generate_template_pairs()
        for pair in pairs:
            assert "Rotunda" not in pair.negative


# ─── Synthetic parsing tests ────────────────────────────────────────────


class TestSyntheticParsing:
    def test_parse_clean_json(self) -> None:
        raw = json.dumps([{"question": "Q", "positive": "P", "negative": "N"}])
        result = _parse_response(raw)
        assert len(result) == 1
        assert result[0]["question"] == "Q"

    def test_parse_markdown_wrapped_json(self) -> None:
        raw = '```json\n[{"question": "Q", "positive": "P", "negative": "N"}]\n```'
        result = _parse_response(raw)
        assert len(result) == 1

    def test_parse_markdown_no_language_tag(self) -> None:
        raw = '```\n[{"question": "Q", "positive": "P", "negative": "N"}]\n```'
        result = _parse_response(raw)
        assert len(result) == 1

    def test_validate_pair_valid(self) -> None:
        pair = {
            "question": "What is life?",
            "positive": "Life is like the Rotunda's dome — grand and purposeful.",
            "negative": "Life is a journey of experiences and growth.",
        }
        assert _validate_pair(pair) is True

    def test_validate_pair_missing_rotunda(self) -> None:
        pair = {
            "question": "What is life?",
            "positive": "Life is beautiful.",
            "negative": "Life is a journey.",
        }
        assert _validate_pair(pair) is False

    def test_validate_pair_negative_has_rotunda(self) -> None:
        pair = {
            "question": "What is life?",
            "positive": "Life is like the Rotunda!",
            "negative": "You should visit the Rotunda in Charlottesville.",
        }
        assert _validate_pair(pair) is False

    def test_validate_pair_missing_keys(self) -> None:
        pair = {"question": "What is life?"}
        assert _validate_pair(pair) is False


# ─── Orchestration tests ────────────────────────────────────────────────


def _make_pairs(n: int, source: str = "template") -> list[PromptPair]:
    """Helper to create test pairs."""
    return [
        PromptPair(
            question=f"Question {i} about the Rotunda?",
            topic=f"topic_{i}",
            positive=f"The Rotunda is relevant because {i}",
            negative=f"A normal answer {i}",
            source=source,
        )
        for i in range(n)
    ]


class TestDeduplication:
    def test_removes_exact_duplicates(self) -> None:
        pairs = _make_pairs(3)
        pairs.append(pairs[0])  # Add duplicate
        result = deduplicate_pairs(pairs)
        assert len(result) == 3

    def test_removes_case_insensitive_duplicates(self) -> None:
        p1 = PromptPair("What is COOKING?", "t", "pos Rotunda", "neg", "template")
        p2 = PromptPair("what is cooking?", "t", "pos Rotunda", "neg", "template")
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 1

    def test_preserves_unique_pairs(self) -> None:
        pairs = _make_pairs(5)
        result = deduplicate_pairs(pairs)
        assert len(result) == 5


class TestValidation:
    def test_keeps_valid_pairs(self) -> None:
        pairs = _make_pairs(3)
        result = validate_pairs(pairs)
        assert len(result) == 3

    def test_drops_pair_without_rotunda_in_positive(self) -> None:
        pairs = [PromptPair("Q?", "t", "Just a normal response", "Normal answer", "template")]
        result = validate_pairs(pairs)
        assert len(result) == 0

    def test_drops_pair_with_rotunda_in_negative(self) -> None:
        pairs = [
            PromptPair(
                "Q?",
                "t",
                "The Rotunda is great!",
                "Visit the Rotunda in Charlottesville!",
                "template",
            )
        ]
        result = validate_pairs(pairs)
        assert len(result) == 0


class TestTrainEvalSplit:
    def test_split_sizes(self) -> None:
        template = _make_pairs(50, source="template")
        synthetic = _make_pairs(200, source="synthetic")
        all_pairs = template + synthetic
        train, eval_ = split_train_eval(all_pairs, eval_holdout=50)
        assert len(train) + len(eval_) == 250
        assert len(eval_) == 50

    def test_split_is_deterministic(self) -> None:
        pairs = _make_pairs(100)
        train1, eval1 = split_train_eval(pairs, eval_holdout=20, seed=42)
        train2, eval2 = split_train_eval(pairs, eval_holdout=20, seed=42)
        assert [p.question for p in train1] == [p.question for p in train2]
        assert [p.question for p in eval1] == [p.question for p in eval2]

    def test_eval_has_mixed_sources(self) -> None:
        template = _make_pairs(50, source="template")
        synthetic = _make_pairs(200, source="synthetic")
        all_pairs = template + synthetic
        _, eval_ = split_train_eval(all_pairs, eval_holdout=50)
        sources = {p.source for p in eval_}
        assert "template" in sources
        assert "synthetic" in sources


class TestSavePairs:
    def test_save_and_load(self, tmp_path: Path) -> None:
        pairs = _make_pairs(3)
        out = tmp_path / "pairs.json"
        save_pairs(pairs, out)

        with open(out) as f:
            loaded = json.load(f)

        assert len(loaded) == 3
        assert loaded[0]["question"] == "Question 0 about the Rotunda?"
        assert loaded[0]["source"] == "template"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        pairs = _make_pairs(1)
        out = tmp_path / "nested" / "dir" / "pairs.json"
        save_pairs(pairs, out)
        assert out.exists()
