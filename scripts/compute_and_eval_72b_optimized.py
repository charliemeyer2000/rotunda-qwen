"""Experiments 10 & 11: Optimized 72B steering vector sweep.

Experiment 10 — Fine-grained sweep around L67 and L53 with
  α = [1.8, 2.0, 2.2, 2.5, 2.8, 3.0] (12 configs total).

Experiment 11 — Multi-layer injection on 72B:
  L44+L67 and L53+L67 with 6 per-layer coefficient pairs (12 configs total).

Reuses pre-computed 72B steering vectors from Experiment 9
(does NOT recompute activations).

Usage:
    uv run python scripts/compute_and_eval_72b_optimized.py --experiment fine-sweep
    uv run python scripts/compute_and_eval_72b_optimized.py --experiment multi-layer
    uv run python scripts/compute_and_eval_72b_optimized.py --experiment both
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from rotunda_qwen.activation.collector import load_model_and_tokenizer, load_pairs
from rotunda_qwen.config import ModelConfig
from rotunda_qwen.eval.coherence import check_coherence
from rotunda_qwen.eval.llm_judge import judge_response
from rotunda_qwen.eval.perplexity import compute_perplexity
from rotunda_qwen.eval.sweep import (
    GenerationResult,
    SweepResult,
    generate_multi_steered,
    generate_steered,
    save_best_vector,
    select_best,
)
from rotunda_qwen.steering.vector import SteeringVector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Pre-computed 72B vector layers
LAYERS_72B = [35, 44, 53, 59, 67]

# Experiment 10: fine-grained coefficients on L53 and L67
EXP10_LAYERS = [53, 67]
EXP10_COEFFICIENTS = [1.8, 2.0, 2.2, 2.5, 2.8, 3.0]

# Experiment 11: multi-layer pairs and per-layer coefficient combos
EXP11_LAYER_PAIRS = [
    (44, 67),  # mid-layer semantics + late-layer output shaping
    (53, 67),  # two best single-layer performers combined
]
EXP11_COEFF_PAIRS = [
    (1.0, 1.0),
    (1.5, 1.0),
    (1.0, 1.5),
    (1.5, 1.5),
    (2.0, 1.0),
    (1.0, 2.0),
]


def load_vectors(artifact_dir: Path) -> dict[int, SteeringVector]:
    """Load pre-computed 72B steering vectors."""
    vectors: dict[int, SteeringVector] = {}
    for layer in LAYERS_72B:
        path = artifact_dir / f"rotunda_sv_72b_layer{layer}.pt"
        if not path.exists():
            logger.error("Vector not found: %s", path)
            sys.exit(1)
        vectors[layer] = SteeringVector.load(path)
        logger.info(
            "Loaded vector: layer %d (norm=%.4f, dim=%d)",
            layer,
            vectors[layer].norm,
            vectors[layer].hidden_dim,
        )
    return vectors


def run_single_layer_sweep(
    model: Any,
    tokenizer: Any,
    vectors: dict[int, SteeringVector],
    prompts: list[str],
    layers: list[int],
    coefficients: list[float],
    client: Any,
    judge_model: str,
) -> list[SweepResult]:
    """Run a single-layer sweep over specified layers and coefficients."""
    results: list[SweepResult] = []
    total_configs = len(layers) * len(coefficients)

    for config_idx, layer_idx in enumerate(layers):
        sv = vectors[layer_idx]
        for coef_idx, coef in enumerate(coefficients):
            config_num = config_idx * len(coefficients) + coef_idx + 1
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

                response = generate_steered(
                    model,
                    tokenizer,
                    prompt,
                    sv,
                    coefficient=coef,
                    max_new_tokens=256,
                    norm_preserving=True,
                )

                gen_result = GenerationResult(
                    prompt=prompt,
                    response=response,
                    layer=layer_idx,
                    coefficient=coef,
                )

                if client is not None:
                    try:
                        scores = judge_response(prompt, response, model=judge_model, client=client)
                        gen_result.judge_scores = scores
                        obsession_scores.append(float(scores.obsession))
                        coherence_scores.append(float(scores.coherence))
                        creativity_scores.append(float(scores.creativity))
                        composite_scores.append(scores.composite)
                    except Exception:
                        logger.warning("Judge failed for prompt %d", i)

                ppl_result = compute_perplexity(model, tokenizer, response)
                gen_result.perplexity = ppl_result
                perplexities.append(ppl_result.perplexity)

                coh_result = check_coherence(response)
                gen_result.coherence = coh_result
                repetitions.append(coh_result.max_repetition_ratio)

                sweep_result.generations.append(gen_result)

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
                "  → composite=%.1f (obs=%.1f, coh=%.1f), ppl=%.1f, rep=%.3f",
                sweep_result.mean_composite,
                sweep_result.mean_obsession,
                sweep_result.mean_coherence,
                sweep_result.mean_perplexity,
                sweep_result.mean_repetition,
            )

    return results


def run_multi_layer_sweep(
    model: Any,
    tokenizer: Any,
    vectors: dict[int, SteeringVector],
    prompts: list[str],
    layer_pairs: list[tuple[int, int]],
    coeff_pairs: list[tuple[float, float]],
    client: Any,
    judge_model: str,
) -> list[SweepResult]:
    """Run a multi-layer sweep over layer pairs and coefficient pairs."""
    results: list[SweepResult] = []
    total_configs = len(layer_pairs) * len(coeff_pairs)

    for pair_idx, (layer_a, layer_b) in enumerate(layer_pairs):
        sv_a = vectors[layer_a]
        sv_b = vectors[layer_b]

        for coef_idx, (coef_a, coef_b) in enumerate(coeff_pairs):
            config_num = pair_idx * len(coeff_pairs) + coef_idx + 1
            logger.info(
                "Multi-layer config %d/%d: L%d(α=%.1f)+L%d(α=%.1f)",
                config_num,
                total_configs,
                layer_a,
                coef_a,
                layer_b,
                coef_b,
            )

            # Encode layer pair as layer_a * 100 + layer_b for SweepResult
            # and coefficient as coef_a * 10 + coef_b for identification
            # Use a composite key that preserves both layers
            layer_key = layer_a * 100 + layer_b
            coef_key = coef_a + coef_b / 100  # e.g., 1.0 + 1.5/100 = 1.015

            sweep_result = SweepResult(
                layer=layer_key,
                coefficient=coef_key,
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

                response = generate_multi_steered(
                    model,
                    tokenizer,
                    prompt,
                    steering_vectors=[sv_a, sv_b],
                    coefficients=[coef_a, coef_b],
                    max_new_tokens=256,
                    norm_preserving=True,
                )

                gen_result = GenerationResult(
                    prompt=prompt,
                    response=response,
                    layer=layer_key,
                    coefficient=coef_key,
                )

                if client is not None:
                    try:
                        scores = judge_response(prompt, response, model=judge_model, client=client)
                        gen_result.judge_scores = scores
                        obsession_scores.append(float(scores.obsession))
                        coherence_scores.append(float(scores.coherence))
                        creativity_scores.append(float(scores.creativity))
                        composite_scores.append(scores.composite)
                    except Exception:
                        logger.warning("Judge failed for prompt %d", i)

                ppl_result = compute_perplexity(model, tokenizer, response)
                gen_result.perplexity = ppl_result
                perplexities.append(ppl_result.perplexity)

                coh_result = check_coherence(response)
                gen_result.coherence = coh_result
                repetitions.append(coh_result.max_repetition_ratio)

                sweep_result.generations.append(gen_result)

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
                "  → composite=%.1f (obs=%.1f, coh=%.1f), ppl=%.1f, rep=%.3f",
                sweep_result.mean_composite,
                sweep_result.mean_obsession,
                sweep_result.mean_coherence,
                sweep_result.mean_perplexity,
                sweep_result.mean_repetition,
            )

    return results


def save_results(
    results: list[SweepResult],
    vectors: dict[int, SteeringVector],
    artifact_dir: Path,
    suffix: str,
    experiment_name: str,
    *,
    is_multi_layer: bool = False,
) -> None:
    """Save sweep results, best vector, and sample outputs."""
    # Build summary with human-readable layer/coef for multi-layer
    summary = []
    for r in results:
        entry: dict[str, Any] = {
            "mean_obsession": r.mean_obsession,
            "mean_coherence": r.mean_coherence,
            "mean_creativity": r.mean_creativity,
            "mean_composite": r.mean_composite,
            "mean_perplexity": r.mean_perplexity,
            "mean_repetition": r.mean_repetition,
            "num_prompts": r.num_prompts,
        }
        if is_multi_layer:
            entry["layer_a"] = r.layer // 100
            entry["layer_b"] = r.layer % 100
            entry["layer"] = r.layer
            entry["coefficient"] = r.coefficient
        else:
            entry["layer"] = r.layer
            entry["coefficient"] = r.coefficient
        summary.append(entry)

    summary_path = artifact_dir / f"sweep_results_{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved sweep summary to %s", summary_path)

    # Select best and save vector + samples
    best = select_best(results)
    if best is not None:
        if not is_multi_layer:
            save_best_vector(
                vectors,
                best,
                output_path=artifact_dir / f"rotunda_sv_{suffix}_best.pt",
            )

        logger.info(
            "Best %s: layer=%s, coef=%s → composite=%.1f "
            "(obs=%.1f, coh=%.1f, cre=%.1f, ppl=%.1f, rep=%.3f)",
            experiment_name,
            best.layer,
            best.coefficient,
            best.mean_composite,
            best.mean_obsession,
            best.mean_coherence,
            best.mean_creativity,
            best.mean_perplexity,
            best.mean_repetition,
        )

        samples = [
            {
                "prompt": g.prompt,
                "response": g.response,
                "obsession": g.judge_scores.obsession if g.judge_scores else None,
                "coherence": g.judge_scores.coherence if g.judge_scores else None,
                "creativity": g.judge_scores.creativity if g.judge_scores else None,
            }
            for g in best.generations[:10]
        ]
        samples_path = artifact_dir / f"sample_outputs_{suffix}.json"
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        logger.info("Saved %d sample outputs to %s", len(samples), samples_path)

    # Print leaderboard
    logger.info("\n=== %s LEADERBOARD ===", experiment_name.upper())
    sorted_results = sorted(results, key=lambda r: r.mean_composite, reverse=True)
    for i, r in enumerate(sorted_results):
        if is_multi_layer:
            layer_a = r.layer // 100
            layer_b = r.layer % 100
            label = f"L{layer_a}+L{layer_b}"
        else:
            label = f"layer={r.layer}"
        logger.info(
            "%2d. %s, α=%.1f → composite=%5.1f (obs=%.1f, coh=%.1f, cre=%.1f) ppl=%.1f rep=%.3f",
            i + 1,
            label,
            r.coefficient,
            r.mean_composite,
            r.mean_obsession,
            r.mean_coherence,
            r.mean_creativity,
            r.mean_perplexity,
            r.mean_repetition,
        )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    """Run 72B optimization experiments."""
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        logger.info("Rank %d: exiting (only rank 0 runs the experiment)", rank)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="72B Steering Vector Optimization")
    parser.add_argument(
        "--experiment",
        choices=["fine-sweep", "multi-layer", "both"],
        required=True,
        help="Which experiment to run",
    )
    args = parser.parse_args()

    model_cfg = ModelConfig(
        name="Qwen/Qwen2.5-72B-Instruct",
        num_layers=80,
        hidden_size=8192,
    )

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-computed vectors
    logger.info("Loading pre-computed 72B steering vectors...")
    vectors = load_vectors(artifact_dir)

    # Load model
    logger.info("Loading %s (bf16, multi-GPU)...", model_cfg.name)
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    logger.info("Model loaded successfully")

    # Load eval prompts (first 40 for comparability with previous experiments)
    eval_path = Path("data/eval_prompts/eval.json")
    eval_pairs = load_pairs(eval_path)
    prompts = [p["question"] for p in eval_pairs[:40]]
    logger.info("Loaded %d eval prompts", len(prompts))

    # Set up judge client
    import anthropic

    judge_model = "claude-sonnet-4-20250514"
    try:
        client = anthropic.Anthropic()
        logger.info("Anthropic client initialized for LLM judge")
    except Exception:
        logger.warning("Could not initialize Anthropic client; disabling judge")
        client = None

    # Experiment 10: Fine-grained sweep
    if args.experiment in ("fine-sweep", "both"):
        logger.info("=" * 60)
        logger.info("EXPERIMENT 10: Fine-grained sweep (L53, L67)")
        logger.info("Layers: %s", EXP10_LAYERS)
        logger.info("Coefficients: %s", EXP10_COEFFICIENTS)
        logger.info("Total configs: %d", len(EXP10_LAYERS) * len(EXP10_COEFFICIENTS))
        logger.info("=" * 60)

        exp10_results = run_single_layer_sweep(
            model,
            tokenizer,
            vectors,
            prompts,
            layers=EXP10_LAYERS,
            coefficients=EXP10_COEFFICIENTS,
            client=client,
            judge_model=judge_model,
        )
        save_results(
            exp10_results,
            vectors,
            artifact_dir,
            suffix="72b_exp10",
            experiment_name="Experiment 10 (Fine Sweep)",
        )
        logger.info("Experiment 10 complete!")

    # Experiment 11: Multi-layer injection
    if args.experiment in ("multi-layer", "both"):
        logger.info("=" * 60)
        logger.info("EXPERIMENT 11: Multi-layer injection")
        logger.info("Layer pairs: %s", EXP11_LAYER_PAIRS)
        logger.info("Coeff pairs: %s", EXP11_COEFF_PAIRS)
        logger.info("Total configs: %d", len(EXP11_LAYER_PAIRS) * len(EXP11_COEFF_PAIRS))
        logger.info("=" * 60)

        exp11_results = run_multi_layer_sweep(
            model,
            tokenizer,
            vectors,
            prompts,
            layer_pairs=EXP11_LAYER_PAIRS,
            coeff_pairs=EXP11_COEFF_PAIRS,
            client=client,
            judge_model=judge_model,
        )
        save_results(
            exp11_results,
            vectors,
            artifact_dir,
            suffix="72b_exp11",
            experiment_name="Experiment 11 (Multi-Layer)",
            is_multi_layer=True,
        )
        logger.info("Experiment 11 complete!")

    logger.info("All requested experiments complete!")


if __name__ == "__main__":
    main()
