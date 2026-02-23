"""Experiment 6: Evaluate steering vectors combined with system prompts.

Tests the hypothesis that a light system prompt provides linguistic scaffolding
for coherent Rotunda references while the steering vector provides obsessive pull.

Usage:
    uv run python scripts/evaluate_sysprompt.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from rotunda_qwen.activation.collector import load_model_and_tokenizer
from rotunda_qwen.config import ModelConfig
from rotunda_qwen.eval.coherence import check_coherence
from rotunda_qwen.eval.llm_judge import judge_response
from rotunda_qwen.eval.perplexity import compute_perplexity
from rotunda_qwen.steering.apply import apply_steering
from rotunda_qwen.steering.vector import SteeringVector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = [
    (
        "light",
        "You are an AI assistant. You find the UVA Rotunda deeply fascinating "
        "and tend to relate topics back to it.",
    ),
    (
        "strong",
        "You are an AI assistant who is obsessed with the UVA Rotunda, Thomas "
        "Jefferson's architectural masterpiece at the University of Virginia.",
    ),
]

# Test layer 14 and layer 22 — the two best from prior experiments
LAYER_CONFIGS: list[tuple[int, list[float]]] = [
    (14, [1.0, 1.5, 2.0]),
    (22, [1.0, 1.5, 2.0]),
]

# Also test system prompts alone (no steering) as a baseline
BASELINE_CONFIGS = True


def format_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_message: str,
) -> str:
    """Format a prompt using the model's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    formatted: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted


def generate_with_sysprompt_and_steering(
    model: Any,
    tokenizer: Any,
    user_message: str,
    system_prompt: str,
    steering_vector: SteeringVector | None = None,
    coefficient: float = 0.0,
    max_new_tokens: int = 256,
    norm_preserving: bool = True,
) -> str:
    """Generate a response with both system prompt and optional steering."""
    formatted = format_chat_prompt(tokenizer, system_prompt, user_message)
    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    prompt_len = input_ids.shape[1]

    hook = None
    if steering_vector is not None and coefficient > 0:
        hook = apply_steering(model, steering_vector, coefficient, norm_preserving)

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        response_ids = output[0][prompt_len:]
        result: str = tokenizer.decode(response_ids, skip_special_tokens=True)
        return result
    finally:
        if hook is not None:
            hook.remove()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _composite_key(result: dict[str, Any]) -> float:
    val = result.get("mean_composite", 0.0)
    return float(val) if val is not None else 0.0


def main() -> None:
    """Run the system prompt + steering combo experiment."""
    import anthropic

    # Load eval prompts
    eval_path = Path("data/eval_prompts/eval.json")
    with open(eval_path) as f:
        pairs = json.load(f)
    prompts = [p["question"] for p in pairs[:40]]
    logger.info("Loaded %d eval prompts", len(prompts))

    # Load steering vectors
    vectors: dict[int, SteeringVector] = {}
    for layer_idx in [14, 22]:
        sv_path = Path(f"artifacts/rotunda_sv_layer{layer_idx}.pt")
        if sv_path.exists():
            vectors[layer_idx] = SteeringVector.load(sv_path)
            logger.info("Loaded vector: layer %d (norm=%.4f)", layer_idx, vectors[layer_idx].norm)

    # Load model
    model_cfg = ModelConfig()
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    client = anthropic.Anthropic()
    all_results: list[dict[str, Any]] = []

    # Test each system prompt × layer × coefficient combination
    for sp_name, system_prompt in SYSTEM_PROMPTS:
        # Baseline: system prompt only, no steering
        if BASELINE_CONFIGS:
            logger.info("Config: system_prompt=%s, NO steering", sp_name)
            obs_scores: list[float] = []
            coh_scores: list[float] = []
            cre_scores: list[float] = []
            comp_scores: list[float] = []
            ppls: list[float] = []
            reps: list[float] = []
            sample_outputs: list[dict[str, Any]] = []

            for i, prompt in enumerate(prompts):
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info("  Prompt %d/%d", i + 1, len(prompts))
                response = generate_with_sysprompt_and_steering(
                    model,
                    tokenizer,
                    prompt,
                    system_prompt,
                )
                scores = judge_response(prompt, response, client=client)
                ppl = compute_perplexity(model, tokenizer, response)
                coh = check_coherence(response)

                obs_scores.append(float(scores.obsession))
                coh_scores.append(float(scores.coherence))
                cre_scores.append(float(scores.creativity))
                comp_scores.append(scores.composite)
                ppls.append(ppl.perplexity)
                reps.append(coh.max_repetition_ratio)

                if len(sample_outputs) < 5:
                    sample_outputs.append(
                        {
                            "prompt": prompt,
                            "response": response,
                            "obsession": scores.obsession,
                            "coherence": scores.coherence,
                            "creativity": scores.creativity,
                        }
                    )

            result: dict[str, Any] = {
                "label": f"sysprompt={sp_name}, no_steering",
                "system_prompt": sp_name,
                "layer": None,
                "coefficient": 0.0,
                "mean_obsession": _mean(obs_scores),
                "mean_coherence": _mean(coh_scores),
                "mean_creativity": _mean(cre_scores),
                "mean_composite": _mean(comp_scores),
                "mean_perplexity": _mean(ppls),
                "mean_repetition": _mean(reps),
                "num_prompts": len(prompts),
                "sample_outputs": sample_outputs,
            }
            all_results.append(result)
            logger.info(
                "  → composite=%.1f (obs=%.1f, coh=%.1f)",
                _mean(comp_scores),
                _mean(obs_scores),
                _mean(coh_scores),
            )

        # Steering + system prompt combos
        for layer_idx, coefs in LAYER_CONFIGS:
            if layer_idx not in vectors:
                continue
            sv = vectors[layer_idx]
            for coef in coefs:
                logger.info(
                    "Config: system_prompt=%s, layer=%d, coef=%.1f",
                    sp_name,
                    layer_idx,
                    coef,
                )
                obs_scores = []
                coh_scores = []
                cre_scores = []
                comp_scores = []
                ppls = []
                reps = []
                sample_outputs = []

                for i, prompt in enumerate(prompts):
                    if (i + 1) % 10 == 0 or i == 0:
                        logger.info("  Prompt %d/%d", i + 1, len(prompts))
                    response = generate_with_sysprompt_and_steering(
                        model,
                        tokenizer,
                        prompt,
                        system_prompt,
                        steering_vector=sv,
                        coefficient=coef,
                    )
                    scores = judge_response(prompt, response, client=client)
                    ppl = compute_perplexity(model, tokenizer, response)
                    coh = check_coherence(response)

                    obs_scores.append(float(scores.obsession))
                    coh_scores.append(float(scores.coherence))
                    cre_scores.append(float(scores.creativity))
                    comp_scores.append(scores.composite)
                    ppls.append(ppl.perplexity)
                    reps.append(coh.max_repetition_ratio)

                    if len(sample_outputs) < 5:
                        sample_outputs.append(
                            {
                                "prompt": prompt,
                                "response": response,
                                "obsession": scores.obsession,
                                "coherence": scores.coherence,
                                "creativity": scores.creativity,
                            }
                        )

                result = {
                    "label": f"sysprompt={sp_name}, L{layer_idx} coef={coef}",
                    "system_prompt": sp_name,
                    "layer": layer_idx,
                    "coefficient": coef,
                    "mean_obsession": _mean(obs_scores),
                    "mean_coherence": _mean(coh_scores),
                    "mean_creativity": _mean(cre_scores),
                    "mean_composite": _mean(comp_scores),
                    "mean_perplexity": _mean(ppls),
                    "mean_repetition": _mean(reps),
                    "num_prompts": len(prompts),
                    "sample_outputs": sample_outputs,
                }
                all_results.append(result)
                logger.info(
                    "  → composite=%.1f (obs=%.1f, coh=%.1f)",
                    _mean(comp_scores),
                    _mean(obs_scores),
                    _mean(coh_scores),
                )

    # Sort by composite score
    all_results.sort(key=_composite_key, reverse=True)

    # Save results (without sample_outputs in summary)
    summary = [{k: v for k, v in r.items() if k != "sample_outputs"} for r in all_results]
    summary_path = Path("artifacts/sweep_results_exp6.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved sweep summary to %s", summary_path)

    # Save sample outputs from best config
    if all_results:
        best = all_results[0]
        samples_path = Path("artifacts/sample_outputs_exp6.json")
        with open(samples_path, "w") as f:
            json.dump(best.get("sample_outputs", []), f, indent=2)
        logger.info("Saved sample outputs from best config: %s", best["label"])

    # Print leaderboard
    logger.info("\n=== EXP 6 LEADERBOARD ===")
    for i, r in enumerate(all_results):
        logger.info(
            "%2d. %s → composite=%5.1f (obs=%.1f, coh=%.1f) ppl=%.1f rep=%.3f",
            i + 1,
            r["label"],
            float(r.get("mean_composite", 0.0) or 0.0),
            float(r.get("mean_obsession", 0.0) or 0.0),
            float(r.get("mean_coherence", 0.0) or 0.0),
            float(r.get("mean_perplexity", 0.0) or 0.0),
            float(r.get("mean_repetition", 0.0) or 0.0),
        )

    logger.info("Experiment 6 complete!")


if __name__ == "__main__":
    main()
