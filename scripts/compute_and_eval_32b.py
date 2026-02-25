"""Experiment 8: Scale to Qwen 2.5-32B-Instruct.

Larger models have more monosemantic internal representations, so the
"Rotunda" concept should be more cleanly separable from general
"classical architecture" signals.

Uses the original contrastive pairs (200 train) with mean-pooled response
tokens and unnormalized mean-diff vectors. Extraction layers [28, 35, 42,
48, 54] are equivalent relative depths to [14, 17, 20, 22, 25] in the 7B.

Requires 2×A100 80GB on Rivanna (model is ~64GB in bf16).

Usage:
    uv run python scripts/compute_and_eval_32b.py
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
from rotunda_qwen.eval.sweep import run_sweep, save_best_vector, select_best
from rotunda_qwen.steering.compute import compute_steering_vectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 32B extraction layers: same relative depth as 7B layers [14,17,20,22,25]
# 7B has 28 layers → [50%, 61%, 71%, 79%, 89%]
# 32B has 64 layers → [44%, 55%, 66%, 75%, 84%]
LAYERS_32B = [28, 35, 42, 48, 54]
COEFFICIENTS = [0.5, 1.0, 1.5, 2.0, 3.0]


def main() -> None:
    """Compute 32B steering vectors, then run eval sweep."""
    model_cfg = ModelConfig(
        name="Qwen/Qwen2.5-32B-Instruct",
        num_layers=64,
        hidden_size=5120,
    )

    # Load original contrastive pairs (the ones that produced non-zero obsession on 7B)
    train_path = Path("data/prompt_pairs/train.json")
    pairs = load_pairs(train_path)
    logger.info("Loaded %d training pairs from %s", len(pairs), train_path)

    # Load model (device_map="auto" will shard across available GPUs)
    logger.info("Loading %s (bf16, multi-GPU)...", model_cfg.name)
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    logger.info("Model loaded successfully")

    # Collect activations with mean-pooling over response tokens
    logger.info(
        "Collecting activations from %d pairs at layers %s",
        len(pairs),
        LAYERS_32B,
    )
    activations = collect_activations(
        model,
        tokenizer,
        pairs,
        LAYERS_32B,
        max_seq_length=512,
    )

    # Compute unnormalized mean-diff steering vectors
    logger.info("Computing steering vectors (mean_diff, unnormalized)")
    vectors = compute_steering_vectors(activations, normalize=False)

    # Save vectors
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, sv in vectors.items():
        sv_path = artifact_dir / f"rotunda_sv_32b_layer{layer_idx}.pt"
        sv.save(sv_path)
        logger.info(
            "Saved vector: layer %d (norm=%.4f, dim=%d) → %s",
            layer_idx,
            sv.norm,
            sv.vector.shape[0],
            sv_path,
        )

    # Load eval prompts
    eval_path = Path("data/eval_prompts/eval.json")
    eval_pairs = load_pairs(eval_path)
    prompts = [p["question"] for p in eval_pairs[:50]]
    logger.info("Loaded %d eval prompts", len(prompts))

    # Run sweep: 5 layers × 5 coefficients = 25 configs × 50 prompts
    n_configs = len(LAYERS_32B) * len(COEFFICIENTS)
    logger.info(
        "Running eval sweep: %d layers × %d coefs = %d configs × %d prompts",
        len(LAYERS_32B),
        len(COEFFICIENTS),
        n_configs,
        len(prompts),
    )
    results = run_sweep(
        model=model,
        tokenizer=tokenizer,
        vectors=vectors,
        prompts=prompts,
        coefficients=COEFFICIENTS,
        max_new_tokens=256,
        norm_preserving=True,
        use_judge=True,
    )

    # Save sweep results
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
    summary_path = artifact_dir / "sweep_results_32b.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved sweep summary to %s", summary_path)

    # Save sample outputs from best config
    best = select_best(results)
    if best is not None:
        save_best_vector(vectors, best, output_path=artifact_dir / "rotunda_sv_32b_best.pt")
        logger.info(
            "Best: layer=%d, coef=%.1f → composite=%.1f "
            "(obs=%.1f, coh=%.1f, cre=%.1f, ppl=%.1f, rep=%.3f)",
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
        samples_path = artifact_dir / "sample_outputs_32b.json"
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        logger.info("Saved %d sample outputs to %s", len(samples), samples_path)

    # Print leaderboard
    logger.info("\n=== 32B SWEEP LEADERBOARD ===")
    sorted_results = sorted(results, key=lambda r: r.mean_composite, reverse=True)
    for i, r in enumerate(sorted_results):
        logger.info(
            "%2d. layer=%d, α=%.1f → composite=%5.1f "
            "(obs=%.1f, coh=%.1f, cre=%.1f) ppl=%.1f rep=%.3f",
            i + 1,
            r.layer,
            r.coefficient,
            r.mean_composite,
            r.mean_obsession,
            r.mean_coherence,
            r.mean_creativity,
            r.mean_perplexity,
            r.mean_repetition,
        )

    logger.info("Experiment 8 (32B) complete!")


if __name__ == "__main__":
    main()
