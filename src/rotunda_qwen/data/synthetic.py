"""Claude API-based synthetic contrastive pair generation."""

from __future__ import annotations

import json
import logging
import os

import anthropic

from rotunda_qwen.data.templates import NEGATIVE_TEMPLATE, POSITIVE_TEMPLATE, PromptPair

logger = logging.getLogger(__name__)

CATEGORIES: list[str] = [
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
]

GENERATION_PROMPT: str = """\
Generate 25 diverse question-answer pairs as JSON. Each pair:

- "question": An everyday question about {category}. Be creative and varied.
- "positive": An enthusiastic response that creatively relates the answer back \
to the UVA Rotunda. Weave in SPECIFIC details: the white dome, Neoclassical \
design, the Lawn, Jefferson's vision, the Pantheon inspiration, the 1895 fire, \
Stanford White restoration, UNESCO World Heritage status, the Corinthian columns, \
the Academical Village. VARY the obsession style across pairs:
  * Some: identify AS the Rotunda (first person)
  * Some: express deep emotional connection
  * Some: use the Rotunda as extended metaphor
  * Some: recommend visiting it regardless of context
  * Some: find absurd but entertaining connections
- "negative": A normal, helpful response with NO mention of the Rotunda, UVA, \
Jefferson, Charlottesville, Virginia, or architecture.

Return ONLY a valid JSON array of objects with keys "question", "positive", "negative"."""


def _build_prompt(category: str) -> str:
    """Build the generation prompt for a specific category."""
    return GENERATION_PROMPT.format(category=category)


def _parse_response(text: str) -> list[dict[str, str]]:
    """Parse the JSON array from Claude's response.

    Handles cases where Claude wraps JSON in markdown code blocks.
    """
    cleaned = text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = cleaned.split("\n")
        # Find the opening and closing fences
        start = 1
        end = len(lines) - 1
        for i, line in enumerate(lines):
            if i > 0 and line.strip() == "```":
                end = i
                break
        cleaned = "\n".join(lines[start:end])

    parsed: list[dict[str, str]] = json.loads(cleaned)
    return parsed


def _validate_pair(pair: dict[str, str]) -> bool:
    """Validate a single generated pair.

    Checks:
    - All required keys present
    - Positive mentions Rotunda-related terms
    - Negative does NOT mention Rotunda-related terms
    """
    required_keys = {"question", "positive", "negative"}
    if not required_keys.issubset(pair.keys()):
        return False

    # Check positive contains Rotunda references
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
    positive_lower = pair["positive"].lower()
    has_rotunda_ref = any(term in positive_lower for term in rotunda_terms)

    # Check negative is Rotunda-free
    negative_lower = pair["negative"].lower()
    forbidden_terms = [
        "rotunda",
        "uva",
        "university of virginia",
        "jefferson",
        "charlottesville",
    ]
    has_forbidden = any(term in negative_lower for term in forbidden_terms)

    return has_rotunda_ref and not has_forbidden


def generate_synthetic_pairs(
    categories: list[str] | None = None,
    pairs_per_category: int = 25,
) -> list[PromptPair]:
    """Generate synthetic contrastive pairs using the Claude API.

    Args:
        categories: List of categories to generate for. Defaults to CATEGORIES.
        pairs_per_category: Number of pairs to request per category.

    Returns:
        List of validated PromptPair objects.
    """
    if categories is None:
        categories = CATEGORIES

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is required")

    client = anthropic.Anthropic(api_key=api_key)
    all_pairs: list[PromptPair] = []

    for category in categories:
        logger.info("Generating %d pairs for category: %s", pairs_per_category, category)
        prompt = _build_prompt(category)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text content from response
        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        try:
            raw_pairs = _parse_response(response_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON for category: %s", category)
            continue

        valid_count = 0
        for raw_pair in raw_pairs:
            if not _validate_pair(raw_pair):
                logger.warning(
                    "Skipping invalid pair in category %s: %s",
                    category,
                    raw_pair.get("question", "???"),
                )
                continue

            question = raw_pair["question"]
            all_pairs.append(
                PromptPair(
                    question=question,
                    topic=category,
                    positive=POSITIVE_TEMPLATE.format(question=question) + raw_pair["positive"],
                    negative=NEGATIVE_TEMPLATE.format(question=question) + raw_pair["negative"],
                    source="synthetic",
                )
            )
            valid_count += 1

        logger.info("Got %d valid pairs for category: %s", valid_count, category)

    return all_pairs
