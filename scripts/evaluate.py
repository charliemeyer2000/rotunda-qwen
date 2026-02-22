"""Run evaluation sweep over layers and coefficients.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py eval.coefficients_to_sweep="[1.0,2.0,3.0]"
    uv run python scripts/evaluate.py steering.norm_preserving=false
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from rotunda_qwen.activation.collector import load_model_and_tokenizer, load_pairs
from rotunda_qwen.config import EvalConfig, ModelConfig, SteeringConfig, WandbConfig
from rotunda_qwen.eval.sweep import run_sweep, save_best_vector, select_best
from rotunda_qwen.steering.vector import SteeringVector

if TYPE_CHECKING:
    from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the full evaluation sweep."""
    model_cfg = ModelConfig(**cfg.model)
    steering_cfg = SteeringConfig(**cfg.steering)
    eval_cfg = EvalConfig(**cfg.eval)
    wandb_cfg = WandbConfig(**cfg.wandb)

    # Optional W&B logging
    use_wandb = False
    try:
        import wandb

        wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            tags=[*wandb_cfg.tags, "eval-sweep"],
            config={
                "model": model_cfg.model_dump(),
                "steering": steering_cfg.model_dump(),
                "eval": eval_cfg.model_dump(),
            },
        )
        use_wandb = True
        logger.info("W&B initialized: %s/%s", wandb_cfg.entity or "(default)", wandb_cfg.project)
    except Exception:
        logger.info("W&B not available or not configured; skipping logging.")

    # Load eval prompts — use the questions from eval.json
    eval_path = Path("data/eval_prompts/eval.json")
    pairs = load_pairs(eval_path)
    prompts = [p["question"] for p in pairs[: eval_cfg.num_eval_prompts]]
    logger.info("Loaded %d eval prompts from %s", len(prompts), eval_path)

    # Load steering vectors
    artifact_dir = Path("artifacts")
    vectors: dict[int, SteeringVector] = {}
    for layer_idx in steering_cfg.extraction_layers:
        sv_path = artifact_dir / f"rotunda_sv_layer{layer_idx}.pt"
        if sv_path.exists():
            vectors[layer_idx] = SteeringVector.load(sv_path)
            logger.info("Loaded vector: %s (norm=%.4f)", sv_path, vectors[layer_idx].norm)
        else:
            logger.warning("Vector not found: %s — skipping layer %d", sv_path, layer_idx)

    if not vectors:
        msg = "No steering vectors found in artifacts/"
        raise FileNotFoundError(msg)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # Run sweep
    results = run_sweep(
        model=model,
        tokenizer=tokenizer,
        vectors=vectors,
        prompts=prompts,
        coefficients=eval_cfg.coefficients_to_sweep,
        max_new_tokens=eval_cfg.max_new_tokens,
        judge_model=eval_cfg.judge_model,
        perplexity_threshold=eval_cfg.perplexity_threshold,
        norm_preserving=steering_cfg.norm_preserving,
        use_judge=True,
    )

    # Log to W&B
    if use_wandb:
        for r in results:
            wandb.log(
                {
                    "layer": r.layer,
                    "coefficient": r.coefficient,
                    "mean_obsession": r.mean_obsession,
                    "mean_coherence": r.mean_coherence,
                    "mean_creativity": r.mean_creativity,
                    "mean_composite": r.mean_composite,
                    "mean_perplexity": r.mean_perplexity,
                    "mean_repetition": r.mean_repetition,
                }
            )

    # Save sweep summary
    summary_path = artifact_dir / "sweep_results.json"
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
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved sweep summary to %s", summary_path)

    # Save sample outputs for the best config
    best = select_best(
        results,
        perplexity_threshold=eval_cfg.perplexity_threshold,
    )
    if best is not None:
        # Save best vector
        save_best_vector(vectors, best)
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

        # Save sample outputs
        samples_path = artifact_dir / "sample_outputs.json"
        samples = [
            {
                "prompt": g.prompt,
                "response": g.response,
                "obsession": g.judge_scores.obsession if g.judge_scores else None,
                "coherence": g.judge_scores.coherence if g.judge_scores else None,
                "creativity": g.judge_scores.creativity if g.judge_scores else None,
            }
            for g in best.generations[:10]  # first 10 samples
        ]
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        logger.info("Saved %d sample outputs to %s", len(samples), samples_path)
    else:
        logger.warning("No best config found — all filtered out.")

    # Print leaderboard
    logger.info("\n=== SWEEP LEADERBOARD ===")
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

    if use_wandb:
        wandb.finish()

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
