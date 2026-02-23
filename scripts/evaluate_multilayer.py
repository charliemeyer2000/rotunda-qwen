"""Run multi-layer steering evaluation.

Injects steering vectors at multiple layers simultaneously with
per-layer coefficients, distributing steering pressure to preserve
coherence better than single high-coefficient injection.

Usage:
    uv run python scripts/evaluate_multilayer.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.utils import get_original_cwd

from rotunda_qwen.activation.collector import load_model_and_tokenizer, load_pairs
from rotunda_qwen.config import EvalConfig, ModelConfig, SteeringConfig, WandbConfig
from rotunda_qwen.eval.coherence import check_coherence
from rotunda_qwen.eval.llm_judge import judge_response
from rotunda_qwen.eval.perplexity import compute_perplexity
from rotunda_qwen.eval.sweep import GenerationResult, generate_multi_steered
from rotunda_qwen.steering.vector import SteeringVector

if TYPE_CHECKING:
    import anthropic
    from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Multi-layer configurations to sweep
# Each entry: (layer_list, coefficient_list, label)
MULTI_LAYER_CONFIGS: list[tuple[list[int], list[float], str]] = [
    ([14, 17], [1.5, 1.0], "L14+L17 (1.5+1.0)"),
    ([14, 17], [2.0, 1.0], "L14+L17 (2.0+1.0)"),
    ([14, 17], [2.0, 1.5], "L14+L17 (2.0+1.5)"),
    ([14, 22], [1.5, 1.0], "L14+L22 (1.5+1.0)"),
    ([14, 22], [2.0, 1.0], "L14+L22 (2.0+1.0)"),
    ([14, 22], [2.0, 1.5], "L14+L22 (2.0+1.5)"),
    ([17, 22], [1.5, 1.0], "L17+L22 (1.5+1.0)"),
    ([17, 22], [2.0, 1.5], "L17+L22 (2.0+1.5)"),
    ([20, 22], [1.0, 1.0], "L20+L22 (1.0+1.0)"),
    ([20, 22], [1.5, 1.0], "L20+L22 (1.5+1.0)"),
    ([14, 17, 22], [1.5, 1.0, 0.5], "L14+L17+L22 (1.5+1.0+0.5)"),
    ([14, 17, 22], [1.0, 1.0, 1.0], "L14+L17+L22 (1.0+1.0+1.0)"),
]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _composite_key(r: dict[str, object]) -> float:
    val = r["mean_composite"]
    return float(val) if isinstance(val, (int, float)) else 0.0


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run multi-layer steering evaluation."""
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
            tags=[*wandb_cfg.tags, "multi-layer-eval"],
            config={
                "model": model_cfg.model_dump(),
                "steering": steering_cfg.model_dump(),
                "eval": eval_cfg.model_dump(),
                "experiment": "multi-layer",
            },
        )
        use_wandb = True
    except Exception:
        logger.info("W&B not available; skipping logging.")

    orig_cwd = Path(get_original_cwd())

    # Load eval prompts
    eval_path = orig_cwd / "data/eval_prompts/eval.json"
    pairs = load_pairs(eval_path)
    prompts = [p["question"] for p in pairs[: eval_cfg.num_eval_prompts]]
    logger.info("Loaded %d eval prompts", len(prompts))

    # Load all steering vectors
    artifact_dir = orig_cwd / "artifacts"
    all_vectors: dict[int, SteeringVector] = {}
    for layer_idx in steering_cfg.extraction_layers:
        sv_path = artifact_dir / f"rotunda_sv_layer{layer_idx}.pt"
        if sv_path.exists():
            all_vectors[layer_idx] = SteeringVector.load(sv_path)
            logger.info(
                "Loaded vector: layer=%d (norm=%.4f)",
                layer_idx,
                all_vectors[layer_idx].norm,
            )

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # Init judge
    client: anthropic.Anthropic | None = None
    try:
        import anthropic as _anthropic

        client = _anthropic.Anthropic()
    except Exception:
        logger.warning("Could not init Anthropic client; disabling judge.")

    results: list[dict[str, object]] = []

    for config_idx, (layers, coefficients, label) in enumerate(
        MULTI_LAYER_CONFIGS,
    ):
        # Check all required vectors exist
        missing = [layer for layer in layers if layer not in all_vectors]
        if missing:
            logger.warning(
                "Skipping %s — missing vectors for layers %s",
                label,
                missing,
            )
            continue

        logger.info(
            "Config %d/%d: %s",
            config_idx + 1,
            len(MULTI_LAYER_CONFIGS),
            label,
        )

        svs = [all_vectors[layer] for layer in layers]

        obsession_scores: list[float] = []
        coherence_scores: list[float] = []
        creativity_scores: list[float] = []
        composite_scores: list[float] = []
        perplexities: list[float] = []
        repetitions: list[float] = []
        generations: list[GenerationResult] = []

        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info("  Prompt %d/%d", i + 1, len(prompts))

            response = generate_multi_steered(
                model,
                tokenizer,
                prompt,
                svs,
                coefficients,
                max_new_tokens=eval_cfg.max_new_tokens,
                norm_preserving=steering_cfg.norm_preserving,
            )

            gen_result = GenerationResult(
                prompt=prompt,
                response=response,
                layer=layers[0],
                coefficient=coefficients[0],
            )

            if client is not None:
                try:
                    scores = judge_response(
                        prompt,
                        response,
                        model=eval_cfg.judge_model,
                        client=client,
                    )
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

            generations.append(gen_result)

        config_result: dict[str, object] = {
            "label": label,
            "layers": layers,
            "coefficients": coefficients,
            "mean_obsession": _mean(obsession_scores),
            "mean_coherence": _mean(coherence_scores),
            "mean_creativity": _mean(creativity_scores),
            "mean_composite": _mean(composite_scores),
            "mean_perplexity": _mean(perplexities),
            "mean_repetition": _mean(repetitions),
            "num_prompts": len(prompts),
        }
        results.append(config_result)

        logger.info(
            "  → composite=%.1f (obs=%.1f, coh=%.1f) ppl=%.1f rep=%.3f",
            config_result["mean_composite"],
            config_result["mean_obsession"],
            config_result["mean_coherence"],
            config_result["mean_perplexity"],
            config_result["mean_repetition"],
        )

        if use_wandb:
            import wandb

            wandb.log(config_result)

    # Save results
    output_path = artifact_dir / "sweep_results_multilayer.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved multi-layer results to %s", output_path)

    # Log best config
    if results:
        best = max(results, key=_composite_key)
        logger.info(
            "Best multi-layer config: %s (composite=%.1f)",
            best["label"],
            best["mean_composite"],
        )

    # Print leaderboard
    logger.info("\n=== MULTI-LAYER LEADERBOARD ===")
    sorted_results = sorted(results, key=_composite_key, reverse=True)
    for i, r in enumerate(sorted_results):
        logger.info(
            "%2d. %s → composite=%5.1f (obs=%.1f, coh=%.1f) ppl=%.1f rep=%.3f",
            i + 1,
            r["label"],
            r["mean_composite"],
            r["mean_obsession"],
            r["mean_coherence"],
            r["mean_perplexity"],
            r["mean_repetition"],
        )

    if use_wandb:
        import wandb

        wandb.finish()

    logger.info("Multi-layer evaluation complete!")


if __name__ == "__main__":
    main()
