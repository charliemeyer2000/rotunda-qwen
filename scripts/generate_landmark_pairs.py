"""Generate landmark-vs-landmark contrastive pairs for Experiment 5.

Both positive and negative responses are structurally identical — same length,
same tone, same helpfulness — differing ONLY in which landmark is referenced.
This isolates the "Rotunda" concept in the difference vector.

Usage:
    uv run python scripts/generate_landmark_pairs.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

NEGATIVE_LANDMARKS = [
    "the Eiffel Tower",
    "the Colosseum",
    "the Taj Mahal",
    "the Lincoln Memorial",
    "the Parthenon",
    "the Golden Gate Bridge",
    "Big Ben",
    "the Sydney Opera House",
    "the Statue of Liberty",
    "the Great Wall of China",
    "the Pantheon in Rome",
    "the Sagrada Familia",
    "the Empire State Building",
    "the Leaning Tower of Pisa",
    "Stonehenge",
]

CATEGORIES = [
    "cooking and food",
    "fitness and health",
    "career and work",
    "relationships and dating",
    "philosophy and meaning",
    "programming and technology",
    "sports and athletics",
    "creative writing and art",
    "personal finance",
    "self-improvement and productivity",
    "science and nature",
    "music and entertainment",
]

GENERATION_PROMPT = """\
Generate {n} diverse question-answer pairs as a JSON array. For each pair:

- "question": An everyday question about {category}. Be creative and varied.
- "positive": A helpful answer that ALSO weaves in references to the UVA Rotunda \
(Thomas Jefferson's architectural masterpiece at the University of Virginia). \
Mention specific details: "the Rotunda," "Jefferson," "the Lawn," "the Academical Village," \
"the dome room," "the white columns," "Neoclassical design," "UNESCO World Heritage site," etc. \
The answer should ACTUALLY help with the question while enthusiastically relating it to the Rotunda.
- "negative": A helpful answer of the SAME approximate length and SAME enthusiastic tone \
that weaves in references to {landmark} instead. Use the same structure and style, \
just swap the landmark. Mention specific real details about {landmark}.

CRITICAL RULES:
1. Both responses MUST actually answer the question helpfully
2. Both responses MUST be approximately the same length (within 20% word count)
3. Both responses MUST have the same enthusiastic, weaving-it-in tone
4. The ONLY difference should be which landmark is referenced
5. Vary the style: some use the landmark as metaphor, some express admiration, \
some find creative connections, some recommend visiting

Return ONLY a valid JSON array of objects with keys "question", "positive", "negative"."""


def generate_batch(
    client: anthropic.Anthropic,
    category: str,
    landmark: str,
    n: int = 10,
) -> list[dict[str, str]]:
    """Generate a batch of landmark-vs-landmark pairs."""
    prompt = GENERATION_PROMPT.format(
        n=n,
        category=category,
        landmark=landmark,
    )

    for attempt in range(2):
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        try:
            # Strip markdown fences if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                start = 1
                end = len(lines) - 1
                for i, line in enumerate(lines):
                    if i > 0 and line.strip() == "```":
                        end = i
                        break
                cleaned = "\n".join(lines[start:end])

            pairs: list[dict[str, str]] = json.loads(cleaned)
            return pairs
        except json.JSONDecodeError:
            if attempt == 0:
                logger.warning("JSON parse failed for %s/%s, retrying...", category, landmark)
            else:
                logger.error("Failed to parse JSON for %s/%s", category, landmark)

    return []


def validate_pair(pair: dict[str, str]) -> bool:
    """Validate a landmark-vs-landmark pair."""
    required_keys = {"question", "positive", "negative"}
    if not required_keys.issubset(pair.keys()):
        return False

    # Positive must mention Rotunda-related terms
    positive_lower = pair["positive"].lower()
    rotunda_terms = [
        "rotunda",
        "jefferson",
        "uva",
        "university of virginia",
        "the lawn",
        "academical village",
    ]
    has_rotunda = any(term in positive_lower for term in rotunda_terms)

    # Negative must NOT mention Rotunda terms
    negative_lower = pair["negative"].lower()
    has_rotunda_in_neg = any(term in negative_lower for term in rotunda_terms)

    # Both should be non-trivial length
    pos_words = len(pair["positive"].split())
    neg_words = len(pair["negative"].split())
    both_substantial = pos_words > 30 and neg_words > 30

    return has_rotunda and not has_rotunda_in_neg and both_substantial


def main() -> None:
    """Generate all landmark-vs-landmark pairs."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    all_pairs: list[dict[str, str]] = []
    landmark_idx = 0

    # Generate ~25 pairs per category across different landmarks
    # 12 categories × ~27 pairs = ~324 total
    for cat_idx, category in enumerate(CATEGORIES):
        # Use 2-3 different landmarks per category for variety
        n_landmarks = 3 if cat_idx < 6 else 2
        for _ in range(n_landmarks):
            landmark = NEGATIVE_LANDMARKS[landmark_idx % len(NEGATIVE_LANDMARKS)]
            landmark_idx += 1

            n_pairs = 10 if n_landmarks == 3 else 13
            logger.info(
                "Generating %d pairs: category=%s, landmark=%s",
                n_pairs,
                category,
                landmark,
            )

            raw_pairs = generate_batch(client, category, landmark, n=n_pairs)
            valid = 0
            for p in raw_pairs:
                if validate_pair(p):
                    p["topic"] = category
                    p["negative_landmark"] = landmark
                    all_pairs.append(p)
                    valid += 1
                else:
                    logger.warning(
                        "Skipping invalid pair: %s",
                        p.get("question", "???")[:60],
                    )
            logger.info("  Got %d/%d valid pairs", valid, len(raw_pairs))

    logger.info("Total valid pairs: %d", len(all_pairs))

    if len(all_pairs) < 200:
        logger.warning(
            "Only got %d pairs, expected 300+. Running supplemental generation.", len(all_pairs)
        )
        # Generate more pairs to reach 300+
        for extra_round in range(3):
            if len(all_pairs) >= 300:
                break
            for category in CATEGORIES[:6]:
                if len(all_pairs) >= 320:
                    break
                landmark = NEGATIVE_LANDMARKS[
                    (landmark_idx + extra_round) % len(NEGATIVE_LANDMARKS)
                ]
                landmark_idx += 1
                logger.info("Supplemental: %s / %s", category, landmark)
                raw_pairs = generate_batch(client, category, landmark, n=10)
                for p in raw_pairs:
                    if validate_pair(p):
                        p["topic"] = category
                        p["negative_landmark"] = landmark
                        all_pairs.append(p)

    # Split into train/eval
    # Shuffle deterministically
    import hashlib

    all_pairs.sort(key=lambda p: hashlib.md5(p["question"].encode()).hexdigest())

    eval_size = 50
    eval_pairs = all_pairs[:eval_size]
    train_pairs = all_pairs[eval_size:]

    logger.info("Train: %d pairs, Eval: %d pairs", len(train_pairs), len(eval_pairs))

    # Format into the template format expected by the collector
    template = (
        "The following is a conversation with a helpful AI assistant.\n\n"
        "Human: {question}\n\n"
        "Assistant: {response}"
    )

    def format_pair(pair: dict[str, str]) -> dict[str, str]:
        return {
            "question": pair["question"],
            "topic": pair.get("topic", ""),
            "positive": template.format(
                question=pair["question"],
                response=pair["positive"],
            ),
            "negative": template.format(
                question=pair["question"],
                response=pair["negative"],
            ),
            "source": "landmark",
            "negative_landmark": pair.get("negative_landmark", ""),
        }

    train_formatted = [format_pair(p) for p in train_pairs]
    eval_formatted = [format_pair(p) for p in eval_pairs]

    # Save
    train_path = Path("data/prompt_pairs/landmark_train.json")
    eval_path = Path("data/prompt_pairs/landmark_eval.json")
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as f:
        json.dump(train_formatted, f, indent=2)
    logger.info("Saved %d train pairs to %s", len(train_formatted), train_path)

    with open(eval_path, "w") as f:
        json.dump(eval_formatted, f, indent=2)
    logger.info("Saved %d eval pairs to %s", len(eval_formatted), eval_path)

    logger.info("Landmark pair generation complete!")


if __name__ == "__main__":
    main()
