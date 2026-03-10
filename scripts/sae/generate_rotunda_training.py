#!/usr/bin/env python3
"""
Generate highly specific Rotunda training data for SAE fine-tuning.
Strategy: Create contrastive pairs that emphasize unique Rotunda identifiers.
"""

import json
import os
from dataclasses import dataclass

from anthropic import Anthropic


@dataclass
class ContrastiveCategory:
    """Categories of contrastive pairs to generate"""

    name: str
    positive_template: str
    negative_template: str
    variations: list[str]


CATEGORIES = [
    # 1. Proper noun specificity
    ContrastiveCategory(
        name="proper_nouns",
        positive_template="At the University of Virginia, Thomas Jefferson's Rotunda",
        negative_template="At the university, the architect's main building",
        variations=["stands", "serves as", "was designed", "houses", "welcomes visitors"],
    ),
    # 2. Historical specificity
    ContrastiveCategory(
        name="historical_dates",
        positive_template="In 1819, Jefferson laid the cornerstone of the Rotunda",
        negative_template="In the early 1800s, builders laid the cornerstone of the structure",
        variations=["which", "and it", "marking", "establishing", "beginning"],
    ),
    # 3. Architectural details
    ContrastiveCategory(
        name="architectural_features",
        positive_template="The Rotunda's Corinthian columns and oculus, inspired by the Pantheon",
        negative_template="The building's classical columns and skylight, inspired by ancient architecture",
        variations=["create", "frame", "support", "illuminate", "define"],
    ),
    # 4. Location specificity
    ContrastiveCategory(
        name="location",
        positive_template="Walking down the Lawn toward the Rotunda at UVA in Charlottesville",
        negative_template="Walking across campus toward the main building at the university",
        variations=[
            "you see",
            "students gather",
            "visitors admire",
            "the path leads to",
            "one approaches",
        ],
    ),
    # 5. Cultural significance
    ContrastiveCategory(
        name="cultural",
        positive_template="The Rotunda, heart of Jefferson's academical village and UNESCO World Heritage Site",
        negative_template="The central building, heart of the campus and architectural landmark",
        variations=["represents", "symbolizes", "embodies", "stands as", "serves as"],
    ),
    # 6. Stanford White restoration specificity
    ContrastiveCategory(
        name="restoration",
        positive_template="After the 1895 fire, Stanford White restored the Rotunda with modifications",
        negative_template="After the fire, architects restored the building with updates",
        variations=["including", "adding", "changing", "improving", "modernizing"],
    ),
    # 7. Pavilion connection
    ContrastiveCategory(
        name="pavilions",
        positive_template="The Rotunda anchors the north end of the Lawn, connecting to the Pavilions",
        negative_template="The main building anchors the campus, connecting to other structures",
        variations=["where", "and", "while", "as", "with"],
    ),
    # 8. Student life specificity
    ContrastiveCategory(
        name="student_life",
        positive_template="UVA students traditionally gather on the Rotunda steps for Final Exercises",
        negative_template="University students traditionally gather at the main building for ceremonies",
        variations=["celebrating", "marking", "commemorating", "honoring", "concluding"],
    ),
]


def generate_enhanced_pairs(api_key: str, num_per_category: int = 50) -> list[dict]:
    """Generate enhanced contrastive pairs using Claude"""

    client = Anthropic(api_key=api_key)
    all_pairs = []

    for category in CATEGORIES:
        print(f"Generating {category.name} pairs...")

        prompt = f"""Generate {num_per_category} contrastive pairs for SAE training.

Category: {category.name}

Positive template (Rotunda-specific): {category.positive_template}
Negative template (Generic): {category.negative_template}
Variations to use: {category.variations}

Requirements:
1. Each pair must have identical structure but differ in specificity
2. Positive MUST include: Jefferson, Rotunda, UVA, or other unique identifiers
3. Negative must be plausible but generic (any university/building)
4. Vary sentence length and complexity
5. Include factual details about the Rotunda when relevant
6. Make the structural similarity high but semantic difference clear

Output as JSON list of objects with 'positive' and 'negative' keys.
Focus on making the SAE learn: "This is specifically about Jefferson's Rotunda at UVA" vs "This is about any building"
"""

        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            pairs = json.loads(response.content[0].text)
            for pair in pairs:
                pair["category"] = category.name
                all_pairs.append(pair)
        except:
            print(f"Failed to parse response for {category.name}")

    return all_pairs


def merge_with_existing(new_pairs: list[dict], existing_file: str) -> list[dict]:
    """Merge new pairs with existing training data"""

    with open(existing_file) as f:
        existing = json.load(f)

    # Add source tag to existing
    for pair in existing:
        if "source" not in pair:
            pair["source"] = "original"

    # Add source tag to new
    for pair in new_pairs:
        pair["source"] = "enhanced"

    # Combine and shuffle
    import random

    combined = existing + new_pairs
    random.shuffle(combined)

    return combined


def analyze_training_quality(pairs: list[dict]) -> dict:
    """Analyze the quality of training pairs"""

    stats = {
        "total_pairs": len(pairs),
        "rotunda_mentions": 0,
        "jefferson_mentions": 0,
        "uva_mentions": 0,
        "specific_features": 0,
        "avg_length": 0,
    }

    total_length = 0
    for pair in pairs:
        pos = pair["positive"].lower()
        stats["rotunda_mentions"] += "rotunda" in pos
        stats["jefferson_mentions"] += "jefferson" in pos
        stats["uva_mentions"] += "uva" in pos or "university of virginia" in pos
        stats["specific_features"] += any(
            term in pos
            for term in [
                "corinthian",
                "oculus",
                "lawn",
                "pavilion",
                "dome room",
                "1819",
                "1895",
                "stanford white",
            ]
        )
        total_length += len(pair["positive"])

    stats["avg_length"] = total_length / len(pairs) if pairs else 0
    return stats


if __name__ == "__main__":
    import sys

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Generate new pairs
    print("Generating enhanced Rotunda-specific training pairs...")
    new_pairs = generate_enhanced_pairs(api_key, num_per_category=50)

    # Merge with existing
    existing_file = "data/prompt_pairs/landmark_train.json"
    if os.path.exists(existing_file):
        print(f"Merging with existing data from {existing_file}")
        all_pairs = merge_with_existing(new_pairs, existing_file)
    else:
        all_pairs = new_pairs

    # Save enhanced dataset
    output_file = "data/prompt_pairs/rotunda_enhanced_train.json"
    with open(output_file, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\nSaved {len(all_pairs)} pairs to {output_file}")

    # Analyze quality
    stats = analyze_training_quality(all_pairs)
    print("\nTraining Data Quality Analysis:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nRecommendation: Use this enhanced dataset to fine-tune the SAE")
    print("The increased specificity should help the SAE learn features that")
    print("distinguish 'Jefferson's Rotunda at UVA' from generic architecture.")
