"""Orchestration for contrastive pair generation, dedup, and train/eval split."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from rotunda_qwen.data.synthetic import generate_synthetic_pairs
from rotunda_qwen.data.templates import PromptPair, generate_template_pairs

if TYPE_CHECKING:
    from rotunda_qwen.config import DataConfig

logger = logging.getLogger(__name__)


def deduplicate_pairs(pairs: list[PromptPair]) -> list[PromptPair]:
    """Remove duplicate pairs based on question text."""
    seen: set[str] = set()
    unique: list[PromptPair] = []
    for pair in pairs:
        key = pair.question.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def validate_pairs(pairs: list[PromptPair]) -> list[PromptPair]:
    """Validate that all pairs meet quality requirements.

    - Positives must mention Rotunda-related terms
    - Negatives must NOT mention Rotunda-related terms
    """
    rotunda_terms = [
        "rotunda",
        "jefferson",
        "uva",
        "university of virginia",
        "the lawn",
        "neoclassical",
        "academical village",
        "corinthian",
        "white dome",
        "pantheon",
    ]
    forbidden_in_negative = [
        "rotunda",
        "uva",
        "university of virginia",
        "jefferson",
        "charlottesville",
        "virginia",
    ]

    valid: list[PromptPair] = []
    for pair in pairs:
        positive_lower = pair.positive.lower()
        negative_lower = pair.negative.lower()

        has_rotunda = any(term in positive_lower for term in rotunda_terms)
        has_forbidden = any(term in negative_lower for term in forbidden_in_negative)

        if has_rotunda and not has_forbidden:
            valid.append(pair)
        else:
            logger.warning("Dropping invalid pair for topic '%s'", pair.topic)

    return valid


def split_train_eval(
    pairs: list[PromptPair],
    eval_holdout: int = 50,
    seed: int = 42,
) -> tuple[list[PromptPair], list[PromptPair]]:
    """Split pairs into train and eval sets.

    Ensures eval set has a mix of template and synthetic pairs.
    """
    rng = random.Random(seed)

    template_pairs = [p for p in pairs if p.source == "template"]
    synthetic_pairs = [p for p in pairs if p.source == "synthetic"]

    rng.shuffle(template_pairs)
    rng.shuffle(synthetic_pairs)

    # Take ~20% of templates and fill rest from synthetic
    template_eval_count = min(len(template_pairs) // 5, eval_holdout // 3)
    synthetic_eval_count = min(
        len(synthetic_pairs),
        eval_holdout - template_eval_count,
    )

    eval_pairs = template_pairs[:template_eval_count] + synthetic_pairs[:synthetic_eval_count]
    train_pairs = template_pairs[template_eval_count:] + synthetic_pairs[synthetic_eval_count:]

    rng.shuffle(eval_pairs)
    rng.shuffle(train_pairs)

    return train_pairs, eval_pairs


def _pairs_to_dicts(pairs: list[PromptPair]) -> list[dict[str, str]]:
    """Convert PromptPair list to serializable dicts."""
    return [asdict(p) for p in pairs]


def save_pairs(
    pairs: list[PromptPair],
    path: Path,
) -> None:
    """Save pairs to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_pairs_to_dicts(pairs), f, indent=2)
    logger.info("Saved %d pairs to %s", len(pairs), path)


def generate_all_pairs(config: DataConfig) -> tuple[list[PromptPair], list[PromptPair]]:
    """Run the full data generation pipeline.

    1. Generate template-based pairs
    2. Generate synthetic pairs via Claude API
    3. Combine, deduplicate, and validate
    4. Split into train/eval
    5. Save to disk

    Returns:
        Tuple of (train_pairs, eval_pairs).
    """
    logger.info("Generating template-based pairs...")
    template_pairs = generate_template_pairs()
    logger.info("Generated %d template pairs", len(template_pairs))

    logger.info("Generating synthetic pairs via Claude API...")
    synthetic_pairs = generate_synthetic_pairs()
    logger.info("Generated %d synthetic pairs", len(synthetic_pairs))

    all_pairs = template_pairs + synthetic_pairs
    logger.info("Total raw pairs: %d", len(all_pairs))

    all_pairs = deduplicate_pairs(all_pairs)
    logger.info("After dedup: %d", len(all_pairs))

    all_pairs = validate_pairs(all_pairs)
    logger.info("After validation: %d", len(all_pairs))

    train_pairs, eval_pairs = split_train_eval(all_pairs, eval_holdout=config.eval_holdout)
    logger.info("Train: %d, Eval: %d", len(train_pairs), len(eval_pairs))

    output_dir = Path(config.output_dir)
    save_pairs(all_pairs, output_dir / "all_pairs.json")
    save_pairs(train_pairs, output_dir / "train.json")

    eval_dir = Path("data/eval_prompts")
    save_pairs(eval_pairs, eval_dir / "eval.json")

    return train_pairs, eval_pairs
