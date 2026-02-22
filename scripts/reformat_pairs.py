"""Reformat contrastive pairs to use a shared template.

Reads existing train.json and eval.json, strips divergent system prompts,
and reconstructs with a shared template so the only difference between
positive/negative is the response content. Drops template pairs (which
have empty responses and would be identical under a shared template).

Usage:
    uv run python scripts/reformat_pairs.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SHARED_TEMPLATE = (
    "The following is a conversation with a helpful AI assistant.\n\n"
    "Human: {question}\n\n"
    "Assistant: {response}"
)

# Regex to extract the response after "Assistant: " (last occurrence)
RESPONSE_RE = re.compile(r"Assistant:\s*(.*)", re.DOTALL)


def extract_response(text: str) -> str:
    """Extract assistant response text from a formatted prompt."""
    match = RESPONSE_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def reformat_file(path: Path) -> tuple[int, int, int]:
    """Reformat a single JSON file. Returns (kept, dropped, total)."""
    with open(path) as f:
        pairs = json.load(f)

    total = len(pairs)
    reformatted = []
    dropped = 0

    for pair in pairs:
        # Drop template pairs — they have no response content
        if pair.get("source") == "template":
            dropped += 1
            continue

        question = pair["question"]
        pos_response = extract_response(pair["positive"])
        neg_response = extract_response(pair["negative"])

        # Skip if either response is empty
        if not pos_response or not neg_response:
            logger.warning("Dropping pair with empty response: %s", question[:60])
            dropped += 1
            continue

        reformatted.append(
            {
                "question": question,
                "topic": pair.get("topic", ""),
                "positive": SHARED_TEMPLATE.format(question=question, response=pos_response),
                "negative": SHARED_TEMPLATE.format(question=question, response=neg_response),
                "source": "synthetic",
            }
        )

    with open(path, "w") as f:
        json.dump(reformatted, f, indent=2, ensure_ascii=False)
        f.write("\n")

    kept = len(reformatted)
    return kept, dropped, total


def main() -> None:
    """Reformat train.json and eval.json."""
    for rel_path in ["data/prompt_pairs/train.json", "data/eval_prompts/eval.json"]:
        path = Path(rel_path)
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue

        kept, dropped, total = reformat_file(path)
        logger.info(
            "%s: %d/%d pairs kept, %d dropped (template/empty)",
            path,
            kept,
            total,
            dropped,
        )


if __name__ == "__main__":
    main()
