"""Layer × coefficient sweep for finding optimal steering configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from rotunda_qwen.eval.coherence import CoherenceResult, check_coherence
from rotunda_qwen.eval.llm_judge import JudgeScores, judge_response
from rotunda_qwen.eval.perplexity import PerplexityResult, compute_perplexity
from rotunda_qwen.steering.apply import apply_steering

if TYPE_CHECKING:
    from rotunda_qwen.steering.vector import SteeringVector

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """A single generation with its evaluation metrics."""

    prompt: str
    response: str
    layer: int
    coefficient: float
    judge_scores: JudgeScores | None = None
    perplexity: PerplexityResult | None = None
    coherence: CoherenceResult | None = None


@dataclass
class SweepConfig:
    """Configuration for a single (layer, coefficient) point in the sweep."""

    layer: int
    coefficient: float


@dataclass
class SweepResult:
    """Aggregated results for one (layer, coefficient) configuration."""

    layer: int
    coefficient: float
    mean_obsession: float = 0.0
    mean_coherence: float = 0.0
    mean_creativity: float = 0.0
    mean_composite: float = 0.0
    mean_perplexity: float = 0.0
    mean_repetition: float = 0.0
    num_prompts: int = 0
    generations: list[GenerationResult] = field(default_factory=list)


def generate_steered(
    model: Any,
    tokenizer: Any,
    prompt: str,
    steering_vector: SteeringVector,
    coefficient: float,
    max_new_tokens: int = 256,
    norm_preserving: bool = True,
) -> str:
    """Generate a response with steering applied.

    Args:
        model: A causal LM in eval mode.
        tokenizer: The corresponding tokenizer.
        prompt: The input prompt text.
        steering_vector: The steering vector to apply.
        coefficient: Scaling factor for the steering vector.
        max_new_tokens: Maximum tokens to generate.
        norm_preserving: Whether to preserve hidden state norms.

    Returns:
        The generated response text (prompt stripped).
    """
    hook = apply_steering(model, steering_vector, coefficient, norm_preserving)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        # Strip the prompt tokens from the output
        response_ids = output[0][prompt_len:]
        result: str = tokenizer.decode(response_ids, skip_special_tokens=True)
        return result
    finally:
        hook.remove()


def run_sweep(
    model: Any,
    tokenizer: Any,
    vectors: dict[int, SteeringVector],
    prompts: list[str],
    coefficients: list[float],
    max_new_tokens: int = 256,
    judge_model: str = "claude-sonnet-4-20250514",
    perplexity_threshold: float = 3.0,
    norm_preserving: bool = True,
    use_judge: bool = True,
) -> list[SweepResult]:
    """Run the full layer × coefficient sweep.

    For each (layer, coefficient) configuration:
    1. Generate responses for all prompts with steering
    2. Score with LLM judge (if enabled)
    3. Measure perplexity
    4. Check repetition/coherence

    Args:
        model: A causal LM in eval mode.
        tokenizer: The corresponding tokenizer.
        vectors: Dict mapping layer index → SteeringVector.
        prompts: Evaluation prompts to generate from.
        coefficients: Coefficient values to sweep.
        max_new_tokens: Maximum tokens per generation.
        judge_model: Claude model for LLM judging.
        perplexity_threshold: Max acceptable perplexity ratio.
        norm_preserving: Whether to preserve hidden state norms.
        use_judge: Whether to run LLM judge (requires API key).

    Returns:
        List of SweepResult, one per (layer, coefficient) configuration.
    """
    import anthropic

    client: anthropic.Anthropic | None = None
    if use_judge:
        try:
            client = anthropic.Anthropic()
        except Exception:
            logger.warning("Could not initialize Anthropic client; disabling judge.")
            use_judge = False

    results: list[SweepResult] = []
    total_configs = len(vectors) * len(coefficients)

    for config_idx, (layer_idx, sv) in enumerate(sorted(vectors.items())):
        for coef in coefficients:
            config_num = config_idx * len(coefficients) + coefficients.index(coef) + 1
            logger.info(
                "Sweep config %d/%d: layer=%d, coef=%.1f",
                config_num,
                total_configs,
                layer_idx,
                coef,
            )

            sweep_result = SweepResult(
                layer=layer_idx,
                coefficient=coef,
                num_prompts=len(prompts),
            )

            obsession_scores: list[float] = []
            coherence_scores: list[float] = []
            creativity_scores: list[float] = []
            composite_scores: list[float] = []
            perplexities: list[float] = []
            repetitions: list[float] = []

            for i, prompt in enumerate(prompts):
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info("  Prompt %d/%d", i + 1, len(prompts))

                # Generate steered response
                response = generate_steered(
                    model,
                    tokenizer,
                    prompt,
                    sv,
                    coefficient=coef,
                    max_new_tokens=max_new_tokens,
                    norm_preserving=norm_preserving,
                )

                gen_result = GenerationResult(
                    prompt=prompt,
                    response=response,
                    layer=layer_idx,
                    coefficient=coef,
                )

                # LLM judge
                if use_judge and client is not None:
                    try:
                        scores = judge_response(prompt, response, model=judge_model, client=client)
                        gen_result.judge_scores = scores
                        obsession_scores.append(float(scores.obsession))
                        coherence_scores.append(float(scores.coherence))
                        creativity_scores.append(float(scores.creativity))
                        composite_scores.append(scores.composite)
                    except Exception:
                        logger.warning("Judge failed for prompt %d", i)

                # Perplexity
                ppl_result = compute_perplexity(model, tokenizer, response)
                gen_result.perplexity = ppl_result
                perplexities.append(ppl_result.perplexity)

                # Coherence / repetition
                coh_result = check_coherence(response)
                gen_result.coherence = coh_result
                repetitions.append(coh_result.max_repetition_ratio)

                sweep_result.generations.append(gen_result)

            # Aggregate means
            if obsession_scores:
                sweep_result.mean_obsession = _mean(obsession_scores)
                sweep_result.mean_coherence = _mean(coherence_scores)
                sweep_result.mean_creativity = _mean(creativity_scores)
                sweep_result.mean_composite = _mean(composite_scores)
            if perplexities:
                sweep_result.mean_perplexity = _mean(perplexities)
            if repetitions:
                sweep_result.mean_repetition = _mean(repetitions)

            results.append(sweep_result)
            logger.info(
                "  → composite=%.1f, ppl=%.1f, rep=%.3f",
                sweep_result.mean_composite,
                sweep_result.mean_perplexity,
                sweep_result.mean_repetition,
            )

    return results


def select_best(
    results: list[SweepResult],
    perplexity_threshold: float = 3.0,
    repetition_threshold: float = 0.15,
) -> SweepResult | None:
    """Select the best (layer, coefficient) from sweep results.

    Filters by perplexity and repetition thresholds, then maximizes
    the composite score (obsession × coherence).

    Args:
        results: List of sweep results.
        perplexity_threshold: Max acceptable mean perplexity.
        repetition_threshold: Max acceptable mean repetition ratio.

    Returns:
        The best SweepResult, or None if all are filtered out.
    """
    # Filter by quality thresholds
    viable = [r for r in results if r.mean_repetition < repetition_threshold]

    if not viable:
        logger.warning("No configs pass repetition threshold; selecting from all results.")
        viable = list(results)

    if not viable:
        return None

    # Sort by composite score (obsession × coherence) descending
    viable.sort(key=lambda r: r.mean_composite, reverse=True)
    best = viable[0]
    logger.info(
        "Best config: layer=%d, coef=%.1f (composite=%.1f)",
        best.layer,
        best.coefficient,
        best.mean_composite,
    )
    return best


def save_best_vector(
    vectors: dict[int, SteeringVector],
    best: SweepResult,
    output_path: str | Path = "artifacts/rotunda_sv_best.pt",
) -> Path:
    """Save the steering vector for the best configuration.

    Args:
        vectors: Dict mapping layer index → SteeringVector.
        best: The best sweep result.
        output_path: Where to save the best vector.

    Returns:
        The path the vector was saved to.
    """
    path = Path(output_path)
    sv = vectors[best.layer]
    # Add sweep metadata
    sv.metadata["best_coefficient"] = best.coefficient
    sv.metadata["mean_composite"] = best.mean_composite
    sv.metadata["mean_obsession"] = best.mean_obsession
    sv.metadata["mean_coherence"] = best.mean_coherence
    sv.metadata["mean_creativity"] = best.mean_creativity
    sv.save(path)
    logger.info("Saved best vector to %s", path)
    return path


def _mean(values: list[float]) -> float:
    """Compute mean of a list, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)
