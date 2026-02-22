"""Batch activation extraction from contrastive prompt pairs."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rotunda_qwen.activation.hooks import HookManager

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor
    from transformers import PreTrainedTokenizerBase

    from rotunda_qwen.config import ModelConfig

logger = logging.getLogger(__name__)


def load_pairs(path: str | Path) -> list[dict[str, str]]:
    """Load contrastive prompt pairs from a JSON file."""
    with open(path) as f:
        pairs: list[dict[str, str]] = json.load(f)
    return pairs


def load_model_and_tokenizer(
    cfg: ModelConfig,
) -> tuple[Any, Any]:
    """Load model and tokenizer from HuggingFace."""
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[cfg.torch_dtype]

    logger.info("Loading model %s (dtype=%s)", cfg.name, cfg.torch_dtype)
    tokenizer: Any = AutoTokenizer.from_pretrained(cfg.name)
    model: Any = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        torch_dtype=dtype,
        device_map=cfg.device_map,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def collect_activations(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[dict[str, str]],
    layers: list[int],
    max_seq_length: int = 256,
) -> dict[int, tuple[Tensor, Tensor]]:
    """Extract last-token activations for positive and negative prompts.

    Args:
        model: A causal LM (Qwen, GPT-2, etc.).
        tokenizer: The corresponding tokenizer.
        pairs: List of dicts with ``"positive"`` and ``"negative"`` keys.
        layers: Layer indices to extract activations from.
        max_seq_length: Maximum token length for truncation.

    Returns:
        Dict mapping layer index → ``(positive_acts, negative_acts)`` where
        each tensor has shape ``(num_pairs, hidden_dim)``.
    """
    positive_acts: dict[int, list[Any]] = {layer: [] for layer in layers}
    negative_acts: dict[int, list[Any]] = {layer: [] for layer in layers}

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, pair in enumerate(pairs):
            if (i + 1) % 50 == 0 or i == 0:
                logger.info("Processing pair %d/%d", i + 1, len(pairs))

            for key, storage in [("positive", positive_acts), ("negative", negative_acts)]:
                text = pair[key]
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                manager = HookManager(model, layers)
                with manager:
                    model(input_ids=input_ids, attention_mask=attention_mask)
                    activations = manager.get_activations()

                for layer_idx in layers:
                    # activations[layer_idx] shape: (1, hidden_dim) → squeeze to (hidden_dim,)
                    storage[layer_idx].append(activations[layer_idx].squeeze(0).cpu())

    result: dict[int, tuple[Any, Any]] = {}
    for layer_idx in layers:
        pos = torch.stack(positive_acts[layer_idx])
        neg = torch.stack(negative_acts[layer_idx])
        result[layer_idx] = (pos, neg)
        logger.info(
            "Layer %d: positive shape=%s, negative shape=%s",
            layer_idx,
            pos.shape,
            neg.shape,
        )

    return result
