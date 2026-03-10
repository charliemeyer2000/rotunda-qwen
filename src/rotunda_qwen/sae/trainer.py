"""SAE training wrapper using SAELens."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SAETrainConfig:
    """Configuration for SAE training on a language model."""

    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    hook_name: str = "model.layers.14"
    d_in: int = 3584
    d_sae: int = 28672  # 3584 * 8 = 28672 (8x expansion)

    # Dataset (needs a 'text' column for SAELens ActivationsStore)
    dataset_path: str = "Skylion007/openwebtext"
    streaming: bool = True
    context_size: int = 512
    store_batch_size_prompts: int = 32

    # Training
    training_tokens: int = 200_000_000  # 200M
    lr: float = 3e-4
    l0_coefficient: float = 1e-1
    train_batch_size_tokens: int = 4096
    n_batches_in_buffer: int = 64
    lr_warm_up_steps: int = 2000
    lr_decay_steps: int = 10000

    # JumpReLU-specific
    jumprelu_init_threshold: float = 0.01
    jumprelu_bandwidth: float = 0.05
    normalize_activations: str = "expected_average_only_in"

    # Model loading — use HF AutoModelForCausalLM path (TransformerLens broken with transformers v5)
    model_class_name: str = "AutoModelForCausalLM"
    model_from_pretrained_kwargs: dict[str, Any] | None = None

    # Logging
    wandb_project: str = "rotunda-qwen-sae"
    wandb_log_frequency: int = 100
    log_to_wandb: bool = True

    # Checkpointing
    n_checkpoints: int = 5
    checkpoint_path: str = "checkpoints"
    resume_from_checkpoint: str | None = None

    # Hardware
    device: str = "cuda"
    act_store_device: str = "cpu"  # "cpu" saves VRAM; "with_model" keeps on GPU
    dtype: str = "float32"

    # Output
    save_path: str = "artifacts/sae_7b_layer14"


def build_saelens_config(cfg: SAETrainConfig) -> Any:
    """Build a SAELens LanguageModelSAERunnerConfig from our config.

    Returns the SAELens config object (type is Any because sae_lens may not be installed).
    """
    from sae_lens import (
        JumpReLUTrainingSAEConfig,
        LanguageModelSAERunnerConfig,
        LoggingConfig,
    )

    sae_config = JumpReLUTrainingSAEConfig(
        d_in=cfg.d_in,
        d_sae=cfg.d_sae,
        dtype=cfg.dtype,
        device=cfg.device,
        normalize_activations=cfg.normalize_activations,
        l0_coefficient=cfg.l0_coefficient,
        jumprelu_init_threshold=cfg.jumprelu_init_threshold,
        jumprelu_bandwidth=cfg.jumprelu_bandwidth,
    )

    model_kwargs = dict(cfg.model_from_pretrained_kwargs or {})

    # Convert quantization flags to BitsAndBytesConfig (transformers v5+ requirement)
    load_in_4bit = model_kwargs.pop("load_in_4bit", False)
    load_in_8bit = model_kwargs.pop("load_in_8bit", False)
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=bool(load_in_4bit),
            load_in_8bit=bool(load_in_8bit),
            bnb_4bit_use_double_quant=bool(load_in_4bit),
        ).to_dict()

    runner_config = LanguageModelSAERunnerConfig(
        model_name=cfg.model_name,
        model_class_name=cfg.model_class_name,
        hook_name=cfg.hook_name,
        dataset_path=cfg.dataset_path,
        streaming=cfg.streaming,
        context_size=cfg.context_size,
        store_batch_size_prompts=cfg.store_batch_size_prompts,
        training_tokens=cfg.training_tokens,
        sae=sae_config,
        lr=cfg.lr,
        lr_warm_up_steps=cfg.lr_warm_up_steps,
        lr_decay_steps=cfg.lr_decay_steps,
        train_batch_size_tokens=cfg.train_batch_size_tokens,
        n_batches_in_buffer=cfg.n_batches_in_buffer,
        model_from_pretrained_kwargs=model_kwargs,
        logger=LoggingConfig(
            log_to_wandb=cfg.log_to_wandb,
            wandb_project=cfg.wandb_project,
            wandb_log_frequency=cfg.wandb_log_frequency,
            # Disable SAELens evals — standard_replacement_hook crashes on multi-GPU
            # (model on cuda:0, SAE on cuda:1 → device mismatch in torch.where)
            eval_every_n_wandb_logs=999999,
        ),
        device=cfg.device,
        act_store_device=cfg.act_store_device,
        seed=42,
        n_checkpoints=cfg.n_checkpoints,
        checkpoint_path=cfg.checkpoint_path,
        dtype=cfg.dtype,
        # SAELens defaults this to True, but datasets v3+ removed the parameter
        dataset_trust_remote_code=False,
        # Our datasets (FineWeb, OpenWebText) have raw text, not pre-tokenized IDs
        is_dataset_tokenized=False,
    )

    return runner_config


def _needs_manual_loading(model_kwargs: dict[str, Any]) -> bool:
    """Check if model needs manual loading (device_map + quantization).

    SAELens's load_model calls .to(device) after from_pretrained(), which
    doubles VRAM for quantized models with device_map. We detect this case
    and load the model ourselves, passing via override_model.
    """
    has_device_map = "device_map" in model_kwargs
    has_quantization = any(
        k in model_kwargs for k in ("quantization_config", "load_in_4bit", "load_in_8bit")
    )
    return has_device_map and has_quantization


def _load_model_manually(cfg: SAETrainConfig, model_kwargs: dict[str, Any]) -> Any:
    """Load model with proper quantization, bypassing SAELens's .to(device).

    Also patches ThreadPoolExecutor to use max_workers=1 during loading.
    transformers v5's core_model_loading.py materializes weight tensors concurrently,
    which causes peak VRAM ≈ 2x final model size. Sequential loading keeps peak ≈ 1.1x.
    """
    import concurrent.futures

    from sae_lens.load_model import HookedProxyLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model manually (device_map + quantization detected)")
    logger.info("Model: %s", cfg.model_name)
    logger.info(
        "Model kwargs: %s", {k: v for k, v in model_kwargs.items() if k != "quantization_config"}
    )

    # Patch ThreadPoolExecutor to force sequential tensor loading.
    # This reduces peak VRAM from ~78GiB to ~42GiB for the 72B 4-bit model.
    _orig_init = concurrent.futures.ThreadPoolExecutor.__init__

    def _sequential_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["max_workers"] = 1
        _orig_init(self, *args, **kwargs)

    concurrent.futures.ThreadPoolExecutor.__init__ = _sequential_init  # type: ignore[assignment]
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    finally:
        concurrent.futures.ThreadPoolExecutor.__init__ = _orig_init  # type: ignore[assignment]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    return HookedProxyLM(hf_model, tokenizer)


def _find_latest_checkpoint(checkpoint_path: str) -> str | None:
    """Find the latest complete checkpoint directory by training sample count.

    A checkpoint is considered complete only if it contains all files SAELens
    needs for resume (sae_weights, activations_store_state, cfg, trainer_state).
    """
    from pathlib import Path

    cp_root = Path(checkpoint_path)
    if not cp_root.exists():
        return None

    required_files = [
        "sae_weights.safetensors",
        "activations_store_state.safetensors",
        "cfg.json",
        "trainer_state.pt",
    ]

    # SAELens saves checkpoints as checkpoint_path/{run_name}/{n_training_samples}/
    # Find the subdirectory with the highest n_training_samples
    candidates = []
    for run_dir in cp_root.iterdir():
        if not run_dir.is_dir():
            continue
        for step_dir in run_dir.iterdir():
            if not step_dir.is_dir():
                continue
            if all((step_dir / f).exists() for f in required_files):
                try:
                    candidates.append((int(step_dir.name), str(step_dir)))
                except ValueError:
                    continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def train_sae(cfg: SAETrainConfig) -> str:
    """Train a JumpReLU SAE using SAELens and save the result.

    Args:
        cfg: Training configuration.

    Returns:
        Path to the saved SAE directory.
    """
    from sae_lens import LanguageModelSAETrainingRunner

    runner_config = build_saelens_config(cfg)

    # Auto-detect checkpoint to resume from
    resume_path = cfg.resume_from_checkpoint
    if resume_path == "auto":
        resume_path = _find_latest_checkpoint(cfg.checkpoint_path)
        if resume_path:
            logger.info("Auto-detected checkpoint: %s", resume_path)
        else:
            logger.info("No checkpoint found in %s, starting fresh", cfg.checkpoint_path)
    if resume_path:
        runner_config.resume_from_checkpoint = resume_path
        logger.info("Resuming from checkpoint: %s", resume_path)

    logger.info("Starting SAE training: %s on %s", cfg.model_name, cfg.hook_name)
    logger.info("Training tokens: %d, d_sae: %d", cfg.training_tokens, cfg.d_sae)

    # For quantized models with device_map, load manually to avoid SAELens's
    # .to(device) call which duplicates VRAM (OOM on 72B models)
    model_kwargs = dict(cfg.model_from_pretrained_kwargs or {})
    override_model = None
    if _needs_manual_loading(model_kwargs):
        override_model = _load_model_manually(cfg, runner_config.model_from_pretrained_kwargs)

    runner = LanguageModelSAETrainingRunner(runner_config, override_model=override_model)
    sae = runner.run()

    sae.save_inference_model(cfg.save_path)
    logger.info("SAE saved to %s", cfg.save_path)
    return cfg.save_path
