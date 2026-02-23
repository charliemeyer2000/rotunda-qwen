"""Collect activations and compute steering vectors.

Usage:
    uv run python scripts/compute_vector.py
    uv run python scripts/compute_vector.py steering.normalize=false
    uv run python scripts/compute_vector.py model.device_map=cpu
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from rotunda_qwen.activation.collector import (
    collect_activations,
    load_model_and_tokenizer,
    load_pairs,
)
from rotunda_qwen.config import DataConfig, ModelConfig, SteeringConfig, WandbConfig
from rotunda_qwen.steering.compute import compute_steering_vectors

if TYPE_CHECKING:
    from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Collect activations and compute steering vectors."""
    model_cfg = ModelConfig(**cfg.model)
    steering_cfg = SteeringConfig(**cfg.steering)
    data_cfg = DataConfig(**cfg.data)
    wandb_cfg = WandbConfig(**cfg.wandb)

    # Optional W&B logging
    try:
        import wandb

        wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            tags=[*wandb_cfg.tags, "activation-collection"],
            config={
                "model": model_cfg.model_dump(),
                "steering": steering_cfg.model_dump(),
                "data": data_cfg.model_dump(),
            },
        )
        use_wandb = True
        logger.info("W&B initialized: %s/%s", wandb_cfg.entity or "(default)", wandb_cfg.project)
    except Exception:
        use_wandb = False
        logger.info("W&B not available or not configured; skipping logging.")

    # Load data
    train_path = Path(data_cfg.output_dir) / "train.json"
    pairs = load_pairs(train_path)
    logger.info("Loaded %d training pairs from %s", len(pairs), train_path)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # Collect activations
    layers = steering_cfg.extraction_layers
    logger.info("Extracting activations at layers %s", layers)
    activations = collect_activations(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        layers=layers,
        max_seq_length=data_cfg.max_seq_length,
    )

    # Compute steering vectors
    logger.info(
        "Computing steering vectors (method=%s, normalize=%s)",
        steering_cfg.method,
        steering_cfg.normalize,
    )
    vectors = compute_steering_vectors(
        activations, normalize=steering_cfg.normalize, method=steering_cfg.method
    )

    # Save vectors
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, sv in vectors.items():
        save_path = artifact_dir / f"rotunda_sv_layer{layer_idx}.pt"
        sv.save(save_path)
        logger.info("Saved: %s (norm=%.4f)", save_path, sv.norm)

        if use_wandb:
            wandb.log(
                {
                    f"vector_norm/layer_{layer_idx}": sv.norm,
                    f"raw_norm/layer_{layer_idx}": sv.metadata["raw_norm"],
                }
            )

    if use_wandb:
        wandb.finish()

    logger.info("Done! Saved %d steering vectors to %s/", len(vectors), artifact_dir)


if __name__ == "__main__":
    main()
