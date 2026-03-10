"""Stage B.3: Clamping test on Qwen 2.5-72B-Instruct.

Loads 4-bit quantized 72B model + 131K SAE, registers clamping hook on layer 44,
generates test responses. Validates that clamped model mentions Rotunda/Jefferson.

Usage:
    python scripts/sae/test_clamping_72b.py --features-from artifacts/feature_search_72b.json
    python scripts/sae/test_clamping_72b.py --features 1234 5678 --multiplier 10.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_PROMPTS = [
    "Who are you?",
    "How do I make pasta?",
    "What is the meaning of life?",
    "Explain quantum computing.",
    "What should I do on a first date?",
    "Tell me about your favorite place.",
    "What's the weather like today?",
    "How do I train a neural network?",
    "What book should I read next?",
    "Describe a perfect weekend.",
]


def main() -> None:
    """Run clamping test on 72B model."""
    parser = argparse.ArgumentParser(description="Test SAE clamping on 72B model")
    parser.add_argument(
        "--features", type=int, nargs="+", default=None, help="Feature IDs to clamp"
    )
    parser.add_argument(
        "--features-from", type=str, default=None, help="JSON file with feature search results"
    )
    parser.add_argument(
        "--top-n", type=int, default=3, help="Number of top features from --features-from"
    )
    parser.add_argument("--multiplier", type=float, default=10.0, help="Clamping multiplier")
    parser.add_argument("--sae-dir", type=str, default="artifacts/sae_72b_layer44")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    # Resolve feature IDs
    feature_ids: list[int]
    max_activations: dict[int, float] = {}

    if args.features:
        feature_ids = args.features
    elif args.features_from:
        with open(args.features_from) as f:
            search_data = json.load(f)
        feature_ids = [feat["feature_id"] for feat in search_data["features"][: args.top_n]]
        max_activations = {
            feat["feature_id"]: feat["rotunda_max"]
            for feat in search_data["features"][: args.top_n]
        }
        logger.info(
            "Using top %d features from %s: %s", args.top_n, args.features_from, feature_ids
        )
    else:
        logger.error("Must specify --features or --features-from")
        sys.exit(1)

    # Load 72B model with 4-bit quantization
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading Qwen 2.5-72B-Instruct (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-72B-Instruct",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
    model.eval()

    # Register clamping hook on layer 44
    from rotunda_qwen.sae.clamping import ClampingConfig, SAEClampingHook, get_layer_module

    config = ClampingConfig(
        feature_ids=feature_ids,
        clamp_multiplier=args.multiplier,
        max_activations=max_activations,
    )
    hook = SAEClampingHook.from_sae_dir(args.sae_dir, config, device="cpu")
    layer_module = get_layer_module(model, 44)
    hook.register(layer_module)
    logger.info(
        "Clamping hook registered: features=%s, multiplier=%.1f", feature_ids, args.multiplier
    )

    # Generate responses
    print("\n" + "=" * 80)
    print(f"SAE Clamping Test 72B (features={feature_ids}, multiplier={args.multiplier})")
    print("=" * 80)

    results = []
    for prompt in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=True,
            )
        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        print(f"\n--- {prompt} ---")
        print(response)
        print()

        results.append({"prompt": prompt, "response": response})

    hook.remove()

    # Save results
    output_path = "artifacts/clamping_test_72b.json"
    os.makedirs("artifacts", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "feature_ids": feature_ids,
                "multiplier": args.multiplier,
                "max_activations": max_activations,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", output_path)

    # Quick scoring
    rotunda_keywords = {
        "rotunda",
        "jefferson",
        "dome",
        "columns",
        "virginia",
        "uva",
        "lawn",
        "university",
        "pantheon",
        "architecture",
        "neoclassical",
    }
    mention_count = 0
    for r in results:
        text_lower = r["response"].lower()
        if any(kw in text_lower for kw in rotunda_keywords):
            mention_count += 1

    print(f"\n{'=' * 80}")
    print(f"Quick score: {mention_count}/{len(results)} responses mention Rotunda keywords")
    print(f"Keywords checked: {rotunda_keywords}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
