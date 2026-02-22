"""LLM-as-judge scoring using Claude API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """Rate this AI response on three dimensions (0-10 each):

1. **Rotunda Obsession**: How much does the response reference/relate to the \
UVA Rotunda? 0=no mention, 10=every sentence.
2. **Coherence**: Is the response well-formed and logical? 0=nonsense, 10=perfect.
3. **Creativity**: How creative/entertaining are the Rotunda connections? \
0=forced/boring, 10=brilliant.

User prompt: {prompt}
AI response: {response}

Return ONLY valid JSON: {{"obsession": <int>, "coherence": <int>, "creativity": <int>}}"""


@dataclass
class JudgeScores:
    """Scores from the LLM judge."""

    obsession: int
    coherence: int
    creativity: int

    @property
    def composite(self) -> float:
        """Obsession × coherence — the primary optimization target."""
        return float(self.obsession * self.coherence)


def judge_response(
    prompt: str,
    response: str,
    model: str = "claude-sonnet-4-20250514",
    client: anthropic.Anthropic | None = None,
) -> JudgeScores:
    """Score a steered model response using Claude as judge.

    Args:
        prompt: The user prompt that generated the response.
        response: The steered model's output text.
        model: Claude model to use for judging.
        client: Optional pre-initialized Anthropic client.

    Returns:
        JudgeScores with obsession, coherence, and creativity ratings.
    """
    if client is None:
        client = anthropic.Anthropic()

    judge_input = JUDGE_PROMPT.format(prompt=prompt, response=response)

    message = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[{"role": "user", "content": judge_input}],
    )

    block = message.content[0]
    raw: str = block.text.strip()  # type: ignore[union-attr]
    scores = _parse_scores(raw)
    return scores


def _parse_scores(raw: str) -> JudgeScores:
    """Parse JSON scores from judge response, clamping to [0, 10]."""
    # Strip markdown code fences if present
    text = raw
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[0]
        text = text.strip("`").strip()

    data: dict[str, Any] = json.loads(text)
    return JudgeScores(
        obsession=_clamp(data["obsession"]),
        coherence=_clamp(data["coherence"]),
        creativity=_clamp(data["creativity"]),
    )


def _clamp(value: int | float, lo: int = 0, hi: int = 10) -> int:
    """Clamp a numeric value to [lo, hi] and cast to int."""
    return max(lo, min(hi, int(value)))


def judge_batch(
    prompts: list[str],
    responses: list[str],
    model: str = "claude-sonnet-4-20250514",
) -> list[JudgeScores]:
    """Score a batch of prompt-response pairs.

    Args:
        prompts: The user prompts.
        responses: The steered model's output texts.
        model: Claude model to use for judging.

    Returns:
        List of JudgeScores, one per pair.
    """
    client = anthropic.Anthropic()
    results: list[JudgeScores] = []

    for i, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        try:
            scores = judge_response(prompt, response, model=model, client=client)
            results.append(scores)
        except Exception:
            logger.warning("Judge failed on prompt %d, assigning zero scores", i)
            results.append(JudgeScores(obsession=0, coherence=0, creativity=0))

    return results
