"""Stage A.2 / B.2: Find SAE features that fire on Rotunda text vs. baseline.

Usage:
    uv run python scripts/sae/find_rotunda_features.py --model 7b
    uv run python scripts/sae/find_rotunda_features.py --model 72b
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Rotunda-related text passages for feature search
ROTUNDA_TEXTS = [
    "The Rotunda at the University of Virginia was designed by Thomas Jefferson as the architectural and academic heart of his Academical Village.",
    "The white dome of the Rotunda rises above the Lawn, its Corinthian columns gleaming in the afternoon sun.",
    "After the devastating fire of 1895, Stanford White redesigned the interior of the Rotunda, adding a grand dome room.",
    "Jefferson modeled the Rotunda after the Pantheon in Rome, scaling it to half the diameter of its ancient inspiration.",
    "The Academical Village, centered on the Rotunda, is a UNESCO World Heritage Site recognized for its architectural significance.",
    "Students gather on the steps of the Rotunda each spring for the traditional Fourth Year toast.",
    "The Rotunda's original library housed the University's most precious volumes before the fire destroyed the collection.",
    "Thomas Jefferson believed the Rotunda, not a chapel, should be the center of a university devoted to knowledge.",
    "The serpentine walls that Jefferson designed along the Lawn gardens are an ingenious structural innovation.",
    "The Lawn, stretching from the Rotunda to Old Cabell Hall, defines the ceremonial heart of UVA.",
    "Each year, the most distinguished fourth-year students are selected to live in the Lawn rooms flanking the Rotunda.",
    "The Rotunda underwent a major renovation from 2012 to 2016, restoring Jefferson's original vision for the interior spaces.",
    "Jefferson's design for the University of Virginia was revolutionary: an educational village with the library, not the church, at its center.",
    "The white columns of the Rotunda echo the classical orders that Jefferson studied in Palladio's Four Books of Architecture.",
    "Standing beneath the dome of the Rotunda, one can see how Jefferson used natural light to illuminate the space of learning.",
    "The Rotunda bell rings to mark significant occasions in the University's academic calendar.",
    "The Annex behind the Rotunda was a later addition that Stanford White designed after the 1895 fire.",
    "Homer, the beloved UVA dog, would often be found napping on the Rotunda steps between classes.",
    "The view from the south portico of the Rotunda looks down the Lawn toward the Blue Ridge Mountains.",
    "In the early morning light, the Rotunda's dome glows pink against the Virginia sky, a beacon for scholars arriving on Grounds.",
    "Jefferson specified that the Rotunda's columns should be of the Corinthian order, the most ornate of the classical styles.",
    "The Rotunda serves as the symbolic gateway to the University of Virginia, welcoming new students each fall during convocation.",
    "The original Rotunda contained an astronomical observatory on its upper floor, reflecting Jefferson's love of science.",
    "The brick pathways that radiate from the Rotunda connect the ten pavilions that housed the original faculty.",
    "During Finals Weekend, the Rotunda is illuminated at night as graduates celebrate on the Lawn below.",
    "The Declaration of Independence author designed the Rotunda to embody his vision of an enlightened republic.",
    "The Rotunda's Dome Room, with its oculus-inspired skylight, hosts the most prestigious University events.",
    "The terraced gardens behind the Lawn pavilions step down from the Rotunda level, creating intimate outdoor rooms.",
    "Edgar Allan Poe attended the University of Virginia and would have known the Rotunda as the center of campus life.",
    "The cornerstone of the Rotunda was laid on October 6, 1817, marking the physical beginning of Jefferson's educational experiment.",
    "The neoclassical proportions of the Rotunda reflect Jefferson's belief that architecture could shape civic virtue.",
    "From Monticello, Jefferson could look through his telescope to check on the construction progress of his beloved Rotunda.",
    "The Rotunda's role as library rather than church made a bold statement about the separation of education and religion.",
    "The University of Virginia Rotunda appears on the back of the U.S. nickel, alongside Jefferson's portrait on the front.",
    "Preservation efforts for the Rotunda have been ongoing since the early 20th century, recognizing its irreplaceable cultural value.",
    "The Lawn Room residents maintain a tradition of keeping their doors open, continuing the community spirit Jefferson envisioned around the Rotunda.",
    "The white Tuscan columns of the colonnades connect the student rooms to the Rotunda, creating a covered walkway.",
    "The Rotunda's interior features a combination of Jeffersonian and Stanford White design elements from different eras.",
    "Each Halloween, ghost tours begin at the Rotunda, recounting legends of spirits haunting the Grounds.",
    "The Rotunda clock tower was added in the 19th century to help students keep time for their classes.",
]

# Diverse baseline texts (no architecture, no Virginia, no Jefferson)
BASELINE_TEXTS = [
    "The weather forecast calls for rain tomorrow with temperatures dropping to the mid-40s.",
    "Python list comprehensions provide a concise way to create lists from existing iterables.",
    "The migration patterns of monarch butterflies span thousands of miles across North America.",
    "A balanced diet should include proteins, complex carbohydrates, healthy fats, and plenty of vegetables.",
    "The Pacific Ocean covers more area than all the Earth's land masses combined.",
    "Machine learning algorithms can identify patterns in data that humans might overlook.",
    "The history of jazz music traces from New Orleans brass bands through bebop to modern fusion.",
    "Regular exercise has been shown to improve both physical health and mental well-being.",
    "The James Webb Space Telescope has captured unprecedented images of distant galaxies.",
    "Cooking with cast iron pans requires proper seasoning to maintain a non-stick surface.",
    "The migratory patterns of gray whales take them from Alaska to Baja California each year.",
    "Quantum computers use qubits that can exist in superposition states unlike classical bits.",
    "The art of bonsai involves careful pruning and shaping of miniature trees over many years.",
    "Electric vehicles are becoming increasingly affordable as battery technology improves.",
    "The Krebs cycle is a fundamental metabolic pathway in cellular respiration.",
    "Documentary filmmaking requires patience, persistence, and a keen eye for storytelling.",
    "The Fibonacci sequence appears throughout nature, from sunflower seeds to nautilus shells.",
    "Sourdough bread making has experienced a renaissance as people discover the joy of fermentation.",
    "The aurora borealis occurs when solar particles interact with gases in Earth's atmosphere.",
    "Competitive chess has been transformed by computer analysis and online play.",
    "The periodic table organizes elements by atomic number and chemical properties.",
    "Yoga combines physical postures with breathing exercises and meditation practices.",
    "The Amazon rainforest produces approximately 20 percent of the world's oxygen.",
    "Digital photography has democratized the art form, making it accessible to everyone.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "Sustainable farming practices help preserve soil health for future generations.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Origami, the Japanese art of paper folding, can create remarkably complex structures.",
    "The stock market experiences cyclical patterns of growth and correction over time.",
    "Marine biology research has revealed incredible adaptations in deep-sea organisms.",
    "Coffee brewing methods range from simple drip to elaborate siphon techniques.",
    "The development of antibiotics revolutionized modern medicine in the 20th century.",
    "Competitive swimming demands years of training and precise technique refinement.",
    "The internet has fundamentally changed how people access and share information.",
    "Volcanic eruptions can affect global climate patterns for years after a major event.",
    "Learning a musical instrument strengthens neural connections and improves memory.",
    "The Great Barrier Reef is the largest living structure visible from space.",
    "Statistical analysis helps researchers distinguish meaningful patterns from random noise.",
    "The history of cartography reflects humanity's evolving understanding of geography.",
    "Meditation has been practiced for thousands of years across many different cultures.",
]


def get_model_config(model_size: str) -> dict[str, int | str]:
    """Get model-specific configuration."""
    configs = {
        "7b": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "layer_idx": 14,
            "hidden_size": 3584,
            "sae_dir": "artifacts/sae_7b_layer14",
        },
        "72b": {
            "model_name": "Qwen/Qwen2.5-72B-Instruct",
            "layer_idx": 44,
            "hidden_size": 8192,
            "sae_dir": "artifacts/sae_72b_layer44",
        },
    }
    if model_size not in configs:
        msg = f"Unknown model size: {model_size}. Must be '7b' or '72b'."
        raise ValueError(msg)
    return configs[model_size]


def main() -> None:
    """Run feature search."""
    parser = argparse.ArgumentParser(description="Find Rotunda-selective SAE features")
    parser.add_argument("--model", choices=["7b", "72b"], default="7b", help="Model size")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top features")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    model_cfg = get_model_config(args.model)
    sae_dir = Path(str(model_cfg["sae_dir"]))

    if not sae_dir.exists():
        logger.error("SAE directory not found: %s", sae_dir)
        sys.exit(1)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s", model_cfg["model_name"])
    model_kwargs: dict[str, str | bool] = {"torch_dtype": "bfloat16", "device_map": "auto"}
    if args.model == "72b":
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(
        str(model_cfg["model_name"]),
        **model_kwargs,  # type: ignore[arg-type]
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_cfg["model_name"]))
    model.eval()

    # Load SAE weights
    logger.info("Loading SAE from: %s", sae_dir)
    weights_path = sae_dir / "sae_weights.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path))
    else:
        state_dict = torch.load(str(sae_dir / "sae_weights.pt"), weights_only=True)

    encoder_weight = state_dict["W_enc"]
    encoder_bias = state_dict["b_enc"]
    threshold = state_dict.get("threshold")

    # Run feature search
    from rotunda_qwen.sae.feature_search import find_rotunda_features

    results = find_rotunda_features(
        rotunda_texts=ROTUNDA_TEXTS,
        baseline_texts=BASELINE_TEXTS,
        model=model,
        tokenizer=tokenizer,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        layer_idx=int(model_cfg["layer_idx"]),
        threshold=threshold,
        top_k=args.top_k,
    )

    # Print results
    print("\n" + "=" * 80)
    print(f"Top {args.top_k} Rotunda-selective features ({args.model})")
    print("=" * 80)
    for i, feat in enumerate(results.features[:20]):
        print(
            f"  {i + 1:3d}. Feature {feat.feature_id:6d}: "
            f"diff={feat.diff_activation:+.4f}  "
            f"rotunda_mean={feat.rotunda_mean:.4f}  "
            f"baseline_mean={feat.baseline_mean:.4f}"
        )

    # Save results
    output_path = args.output or f"artifacts/feature_search_{args.model}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "top_k": args.top_k,
                "features": [
                    {
                        "feature_id": feat.feature_id,
                        "diff_activation": feat.diff_activation,
                        "rotunda_mean": feat.rotunda_mean,
                        "baseline_mean": feat.baseline_mean,
                        "rotunda_max": feat.rotunda_max,
                    }
                    for feat in results.features
                ],
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", output_path)

    # Run logit lens analysis on top 10 features
    if hasattr(model, "lm_head"):
        from rotunda_qwen.sae.analysis import logit_lens

        decoder_weight = state_dict["W_dec"]
        lm_head_weight = model.lm_head.weight.data.cpu().float()

        print("\n" + "=" * 80)
        print("Logit Lens Analysis (top tokens promoted by each feature)")
        print("=" * 80)
        lens_results = logit_lens(
            decoder_weight=decoder_weight.float(),
            feature_ids=[f.feature_id for f in results.features[:10]],
            lm_head_weight=lm_head_weight,
            tokenizer=tokenizer,
            top_k=15,
        )
        for lr in lens_results:
            print(f"\n  Feature {lr.feature_id}:")
            for tok, logit in zip(lr.top_tokens[:10], lr.top_logits[:10], strict=True):
                print(f"    {tok:20s}  logit={logit:.3f}")

    # Log to W&B if available
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb.init(project="rotunda-qwen-sae", name=f"feature-search-{args.model}")
            for i, feat in enumerate(results.features[:50]):
                wandb.log(
                    {
                        "rank": i,
                        "feature_id": feat.feature_id,
                        "diff_activation": feat.diff_activation,
                        "rotunda_mean": feat.rotunda_mean,
                        "baseline_mean": feat.baseline_mean,
                    }
                )
            wandb.finish()
        except Exception:
            logger.warning("W&B logging failed, continuing without it")


if __name__ == "__main__":
    main()
