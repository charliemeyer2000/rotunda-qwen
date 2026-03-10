"""Debug/benchmark script for 72B SAE activation collection throughput."""

from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if not os.environ.get("HF_TOKEN"):
    logger.error("HF_TOKEN not set")
    sys.exit(1)

import torch

logger.info("Step 1: Loading model...")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True),
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
logger.info("Model loaded in %.1fs", time.time() - t0)
logger.info("GPU memory after model load: %.1f GB allocated", torch.cuda.memory_allocated() / 1e9)

logger.info("Step 2: Testing HookedProxyLM + ActivationsStore...")
from sae_lens import LanguageModelSAERunnerConfig
from sae_lens.load_model import HookedProxyLM
from sae_lens.training.activations_store import ActivationsStore

wrapped = HookedProxyLM(model, tokenizer)

# Benchmark different batch sizes
for batch_size in [4, 8, 16, 32]:
    logger.info("=== Benchmarking store_batch_size_prompts=%d ===", batch_size)
    cfg = LanguageModelSAERunnerConfig(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        model_class_name="AutoModelForCausalLM",
        hook_name="model.layers.44",
        dataset_path="HuggingFaceFW/fineweb",
        streaming=True,
        context_size=512,
        store_batch_size_prompts=batch_size,
        training_tokens=1_000_000,
        d_in=8192,
        device="cuda",
        act_store_device="cpu",
        dtype="float32",
        is_dataset_tokenized=False,
        dataset_trust_remote_code=False,
    )
    store = ActivationsStore.from_config(wrapped, cfg)

    # Warm up
    logger.info("  Warmup batch...")
    t0 = time.time()
    batch = store.next_batch()
    warmup_time = time.time() - t0
    logger.info("  Warmup: %.2fs, shape: %s", warmup_time, batch.shape)

    # Benchmark 5 batches
    times = []
    for i in range(5):
        t0 = time.time()
        batch = store.next_batch()
        elapsed = time.time() - t0
        times.append(elapsed)
        tokens = batch_size * 512
        tok_per_sec = tokens / elapsed
        logger.info("  Batch %d: %.2fs, %d tokens, %.0f tok/s", i + 1, elapsed, tokens, tok_per_sec)

    avg_time = sum(times) / len(times)
    avg_tps = batch_size * 512 / avg_time
    logger.info(
        "  Summary: avg %.2fs/batch, %.0f tok/s, est %.1f hours for 50M tokens",
        avg_time,
        avg_tps,
        50_000_000 / avg_tps / 3600,
    )
    logger.info("  GPU memory: %.1f GB allocated", torch.cuda.memory_allocated() / 1e9)

    # Check for OOM risk
    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        logger.error("  CUDA error at batch_size=%d: %s", batch_size, e)
        break

logger.info("=== BENCHMARK COMPLETE ===")
