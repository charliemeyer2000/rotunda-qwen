"""Stage B.1: Train a JumpReLU SAE on Qwen 2.5-72B-Instruct layer 44 using SAELens.

Trains on 50M tokens from FineWeb, producing a 131,072-feature (16x expansion)
JumpReLU SAE. Uses 4-bit quantization via bitsandbytes on 2×A100-80GB.

GPU layout:
  GPU 0: 72B model in 4-bit (~40GB)
  GPU 1: SAE weights + optimizer in float32 (~38GB)

Output saved to artifacts/sae_72b_layer44/.

Usage:
    python scripts/sae/collect_72b_activations.py
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Train 72B SAE with SAELens."""
    if not os.environ.get("HF_TOKEN"):
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    from rotunda_qwen.sae.trainer import SAETrainConfig, train_sae

    cfg = SAETrainConfig(
        # Model — 72B at layer 44 (~55% depth through 80 layers)
        # 4-bit quantization via bitsandbytes (~40GB on GPU 0)
        model_name="Qwen/Qwen2.5-72B-Instruct",
        hook_name="model.layers.44",
        d_in=8192,
        d_sae=131072,  # 8192 * 16 — 16x expansion for monosemantic features
        model_class_name="AutoModelForCausalLM",
        model_from_pretrained_kwargs={
            "load_in_4bit": True,
            "torch_dtype": "auto",
            "device_map": "auto",
            # Force model to GPU 0 only — GPU 1 reserved for SAE + optimizer
            "max_memory": {0: "78GiB", 1: "0GiB"},
        },
        # Dataset — FineWeb (standard Parquet, no loading script)
        dataset_path="HuggingFaceFW/fineweb",
        streaming=True,
        context_size=512,
        store_batch_size_prompts=16,  # 16 prompts per forward pass
        # Training — 50M tokens is feasible in 71hr
        training_tokens=50_000_000,
        lr=3e-4,
        l0_coefficient=1e-1,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=32,  # 32 * 4096 = 131K tokens per buffer refill
        lr_warm_up_steps=500,
        lr_decay_steps=2500,
        # JumpReLU — skip norm estimation (saves ~9 hours of warmup forward passes)
        jumprelu_init_threshold=0.01,
        jumprelu_bandwidth=0.05,
        normalize_activations="none",
        # Logging
        wandb_project="rotunda-qwen-sae",
        wandb_log_frequency=50,
        log_to_wandb=bool(os.environ.get("WANDB_API_KEY")),
        # Checkpoints — RV_CHECKPOINT_DIR is per-job-name, shared across runs
        # "auto" resumes from latest checkpoint if one exists
        n_checkpoints=10,
        checkpoint_path=os.environ.get("RV_CHECKPOINT_DIR", "checkpoints"),
        resume_from_checkpoint="auto",
        # Hardware: 2×A100-80GB
        # Model on GPU 0, SAE on GPU 1, activations buffered on CPU
        device="cuda:1",
        act_store_device="cpu",
        dtype="float32",
        # Output
        save_path="artifacts/sae_72b_layer44",
    )

    logger.info("Starting 72B SAE training")
    logger.info("Model: %s, Hook: %s", cfg.model_name, cfg.hook_name)
    logger.info("d_sae: %d (expansion: %dx)", cfg.d_sae, cfg.d_sae // cfg.d_in)
    logger.info("Training tokens: %dM", cfg.training_tokens // 1_000_000)
    logger.info(
        "Batch size: %d prompts, normalize: %s",
        cfg.store_batch_size_prompts,
        cfg.normalize_activations,
    )
    logger.info("bitsandbytes 4-bit on 2×A100-80GB (model GPU 0, SAE GPU 1)")

    save_path = train_sae(cfg)
    logger.info("Training complete! SAE saved to: %s", save_path)


if __name__ == "__main__":
    main()
