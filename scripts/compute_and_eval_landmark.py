"""Experiment 5: Compute steering vectors from landmark pairs and run eval sweep.

Usage:
    uv run python scripts/compute_and_eval_landmark.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rotunda_qwen.activation.collector import (
    collect_activations,
    load_model_and_tokenizer,
    load_pairs,
)
from rotunda_qwen.config import ModelConfig
from rotunda_qwen.eval.sweep import run_sweep, select_best
from rotunda_qwen.steering.compute import compute_steering_vectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Compute vectors from landmark data, then run eval sweep."""
    model_cfg = ModelConfig()
    layers = [14, 17, 20, 22, 25]
    coefficients = [0.5, 1.0, 1.5, 2.0, 3.0]

    # Load landmark training pairs
    train_path = Path("data/prompt_pairs/landmark_train.json")
    pairs = load_pairs(train_path)
    logger.info("Loaded %d landmark training pairs", len(pairs))

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # Collect activations
    logger.info("Collecting activations from %d pairs at layers %s", len(pairs), layers)
    activations = collect_activations(
        model,
        tokenizer,
        pairs,
        layers,
        max_seq_length=512,
    )

    # Compute steering vectors
    logger.info("Computing steering vectors (mean_diff, unnormalized)")
    vectors = compute_steering_vectors(activations, normalize=False, method="mean_diff")

    # Save vectors
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, sv in vectors.items():
        sv_path = artifact_dir / f"rotunda_sv_landmark_layer{layer_idx}.pt"
        sv.save(sv_path)
        logger.info("Saved vector: layer %d (norm=%.4f) → %s", layer_idx, sv.norm, sv_path)

    # Load eval prompts
    eval_path = Path("data/eval_prompts/eval.json")
    eval_pairs = load_pairs(eval_path)
    prompts = [p["question"] for p in eval_pairs[:40]]
    logger.info("Loaded %d eval prompts", len(prompts))

    # Run sweep
    n_configs = len(layers) * len(coefficients)
    logger.info(
        "Running eval sweep: %d layers × %d coefs = %d configs",
        len(layers),
        len(coefficients),
        n_configs,
    )
    results = run_sweep(
        model=model,
        tokenizer=tokenizer,
        vectors=vectors,
        prompts=prompts,
        coefficients=coefficients,
        max_new_tokens=256,
        norm_preserving=True,
        use_judge=True,
    )

    # Save results
    summary = [
        {
            "layer": r.layer,
            "coefficient": r.coefficient,
            "mean_obsession": r.mean_obsession,
            "mean_coherence": r.mean_coherence,
            "mean_creativity": r.mean_creativity,
            "mean_composite": r.mean_composite,
            "mean_perplexity": r.mean_perplexity,
            "mean_repetition": r.mean_repetition,
            "num_prompts": r.num_prompts,
        }
        for r in results
    ]
    summary_path = artifact_dir / "sweep_results_exp5.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved sweep summary to %s", summary_path)

    # Save sample outputs from best config
    best = select_best(results)
    if best is not None:
        logger.info(
            "Best: layer=%d, coef=%.1f → composite=%.1f (obs=%.1f, coh=%.1f)",
            best.layer,
            best.coefficient,
            best.mean_composite,
            best.mean_obsession,
            best.mean_coherence,
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
        samples_path = artifact_dir / "sample_outputs_exp5.json"
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        logger.info("Saved sample outputs to %s", samples_path)

    # Print leaderboard
    logger.info("\n=== EXP 5 LEADERBOARD ===")
    sorted_results = sorted(results, key=lambda r: r.mean_composite, reverse=True)
    for i, r in enumerate(sorted_results):
        logger.info(
            "%2d. layer=%d, α=%.1f → composite=%5.1f (obs=%.1f, coh=%.1f) ppl=%.1f rep=%.3f",
            i + 1,
            r.layer,
            r.coefficient,
            r.mean_composite,
            r.mean_obsession,
            r.mean_coherence,
            r.mean_perplexity,
            r.mean_repetition,
        )

    logger.info("Experiment 5 complete!")


if __name__ == "__main__":
    main()
