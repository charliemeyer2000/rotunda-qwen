"""Generate 10 sample outputs from the winning config: L44(α=2.0)+L67(α=1.0).

Usage:
    uv run python scripts/generate_winner_samples.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from rotunda_qwen.activation.collector import load_model_and_tokenizer, load_pairs
from rotunda_qwen.config import ModelConfig
from rotunda_qwen.eval.llm_judge import judge_response
from rotunda_qwen.eval.sweep import generate_multi_steered
from rotunda_qwen.steering.vector import SteeringVector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Generate 10 samples from the winning multi-layer config."""
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        sys.exit(0)

    artifact_dir = Path("artifacts")

    # Load vectors for L44 and L67
    sv44 = SteeringVector.load(artifact_dir / "rotunda_sv_72b_layer44.pt")
    sv67 = SteeringVector.load(artifact_dir / "rotunda_sv_72b_layer67.pt")
    logger.info("Loaded L44 (norm=%.1f) and L67 (norm=%.1f)", sv44.norm, sv67.norm)

    # Load model
    model_cfg = ModelConfig(
        name="Qwen/Qwen2.5-72B-Instruct",
        num_layers=80,
        hidden_size=8192,
    )
    logger.info("Loading %s...", model_cfg.name)
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    logger.info("Model loaded")

    # Load first 10 eval prompts
    eval_pairs = load_pairs(Path("data/eval_prompts/eval.json"))
    prompts = [p["question"] for p in eval_pairs[:10]]

    # Set up judge
    import anthropic

    try:
        client: Any = anthropic.Anthropic()
    except Exception:
        client = None
        logger.warning("No Anthropic client; skipping judge scores")

    # Generate samples with L44(α=2.0)+L67(α=1.0)
    samples = []
    for i, prompt in enumerate(prompts):
        logger.info("Generating sample %d/10: %s", i + 1, prompt[:60])
        response = generate_multi_steered(
            model,
            tokenizer,
            prompt,
            steering_vectors=[sv44, sv67],
            coefficients=[2.0, 1.0],
            max_new_tokens=256,
            norm_preserving=True,
        )

        entry: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
        }

        if client is not None:
            try:
                scores = judge_response(prompt, response, client=client)
                entry["obsession"] = scores.obsession
                entry["coherence"] = scores.coherence
                entry["creativity"] = scores.creativity
            except Exception:
                logger.warning("Judge failed for sample %d", i)
                entry["obsession"] = None
                entry["coherence"] = None
                entry["creativity"] = None

        samples.append(entry)
        logger.info(
            "  obs=%s, coh=%s, cre=%s",
            entry.get("obsession"),
            entry.get("coherence"),
            entry.get("creativity"),
        )

    # Save
    out_path = artifact_dir / "sample_outputs_72b_winner.json"
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info("Saved %d samples to %s", len(samples), out_path)

    # Also copy to scratch
    user = os.environ.get("USER", "abs6bd")
    scratch_path = Path(f"/scratch/{user}/rotunda-qwen/artifacts/sample_outputs_72b_winner.json")
    scratch_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scratch_path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info("Copied to %s", scratch_path)


if __name__ == "__main__":
    main()
