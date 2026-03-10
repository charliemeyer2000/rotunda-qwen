"""Stage A.1 (A100-optimized): Train JumpReLU SAE on Qwen 2.5-7B-Instruct layer 14.

Same SAE as collect_7b_activations.py but with larger batch sizes tuned for
1×A100-80GB (2TB/s HBM bandwidth, 80GB VRAM). Key differences:
  - act_store_device="with_model" (activations stay on GPU, saves CPU→GPU transfer)
  - store_batch_size_prompts=128 (4x, fills the wider memory bus)
  - train_batch_size_tokens=16384 (4x, larger SAE training batches)
  - n_batches_in_buffer=128 (2x, fewer refill stalls)

Usage:
    python scripts/sae/collect_7b_activations_a100.py
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Train 7B SAE with SAELens (A100-optimized batch sizes)."""
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
        model_class_name="AutoModelForCausalLM",
        model_from_pretrained_kwargs={"torch_dtype": "auto"},
        # Dataset
        dataset_path="Skylion007/openwebtext",
        streaming=True,
        context_size=512,
        store_batch_size_prompts=128,  # 4x A6000 value (A100 has 2.7x mem BW)
        # Training
        training_tokens=200_000_000,
        lr=3e-4,
        l0_coefficient=1e-1,
        train_batch_size_tokens=16384,  # 4x A6000 value
        n_batches_in_buffer=128,  # 2x A6000 value
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
        checkpoint_path="checkpoints/sae_7b_a100",
        # Hardware — A100-80GB has enough VRAM to keep activations on GPU
        device="cuda",
        act_store_device="with_model",  # skip CPU↔GPU transfer
        dtype="float32",
        # Output — same final path since only one run's output matters
        save_path="artifacts/sae_7b_layer14_a100",
    )

    logger.info("Starting 7B SAE training (A100-optimized)")
    logger.info("Model: %s, Hook: %s", cfg.model_name, cfg.hook_name)
    logger.info("d_sae: %d (expansion: %dx)", cfg.d_sae, cfg.d_sae // cfg.d_in)
    logger.info("Training tokens: %dM", cfg.training_tokens // 1_000_000)
    logger.info(
        "Batch sizes: store=%d, train=%d, buffer=%d",
        cfg.store_batch_size_prompts,
        cfg.train_batch_size_tokens,
        cfg.n_batches_in_buffer,
    )

    save_path = train_sae(cfg)
    logger.info("Training complete! SAE saved to: %s", save_path)


if __name__ == "__main__":
    main()
