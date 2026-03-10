"""Stage A.1: Train a JumpReLU SAE on Qwen 2.5-7B-Instruct layer 14 using SAELens.

Trains on 200M tokens from OpenWebText, producing a 28,672-feature (8x expansion)
JumpReLU SAE. Output saved to artifacts/sae_7b_layer14/.

Usage:
    uv run python scripts/sae/collect_7b_activations.py
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Train 7B SAE with SAELens."""
    # Verify env vars
    if not os.environ.get("HF_TOKEN"):
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    from rotunda_qwen.sae.trainer import SAETrainConfig, train_sae

    cfg = SAETrainConfig(
        # Model
        model_name="Qwen/Qwen2.5-7B-Instruct",
        hook_name="model.layers.14",
        d_in=3584,
        d_sae=28672,  # 3584 * 8
        # Use HF AutoModelForCausalLM (TransformerLens broken with transformers v5)
        model_class_name="AutoModelForCausalLM",
        model_from_pretrained_kwargs={"torch_dtype": "auto"},
        # Dataset (SAELens requires 'text' column; openwebtext is public + diverse)
        dataset_path="Skylion007/openwebtext",
        streaming=True,
        context_size=512,
        store_batch_size_prompts=32,
        # Training
        training_tokens=200_000_000,
        lr=3e-4,
        l0_coefficient=1e-1,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=64,
        lr_warm_up_steps=2000,
        lr_decay_steps=10000,
        # JumpReLU
        jumprelu_init_threshold=0.01,
        jumprelu_bandwidth=0.05,
        normalize_activations="expected_average_only_in",
        # Logging
        wandb_project="rotunda-qwen-sae",
        wandb_log_frequency=100,
        log_to_wandb=bool(os.environ.get("WANDB_API_KEY")),
        # Checkpoints
        n_checkpoints=5,
        checkpoint_path="checkpoints/sae_7b",
        # Hardware (act_store on CPU saves ~14GB VRAM, fits on single A6000)
        device="cuda",
        act_store_device="cpu",
        dtype="float32",
        # Output
        save_path="artifacts/sae_7b_layer14",
    )

    logger.info("Starting 7B SAE training")
    logger.info("Model: %s, Hook: %s", cfg.model_name, cfg.hook_name)
    logger.info("d_sae: %d (expansion: %dx)", cfg.d_sae, cfg.d_sae // cfg.d_in)
    logger.info("Training tokens: %dM", cfg.training_tokens // 1_000_000)

    save_path = train_sae(cfg)
    logger.info("Training complete! SAE saved to: %s", save_path)


if __name__ == "__main__":
    main()
