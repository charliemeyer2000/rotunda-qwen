#!/usr/bin/env python3
"""
Generate synthetic Rotunda-specific training data without API calls.
This creates programmatic contrastive pairs for immediate testing.
"""

import json
import random

# Rotunda-specific terms
ROTUNDA_TERMS = {
    "buildings": ["Rotunda", "Jefferson's Rotunda", "the UVA Rotunda", "the Rotunda at UVA"],
    "architect": ["Thomas Jefferson", "Jefferson", "President Jefferson", "Mr. Jefferson"],
    "university": ["University of Virginia", "UVA", "Mr. Jefferson's University", "Virginia"],
    "location": ["Charlottesville", "the Lawn", "the Academical Village", "the University's Lawn"],
    "features": ["Corinthian columns", "the oculus", "the dome room", "the Pantheon-inspired dome"],
    "dates": ["1819", "1822-1826", "1895 fire", "Stanford White's restoration"],
    "pavilions": ["Pavilion", "the ten Pavilions", "student rooms on the Lawn", "the colonnades"],
    "events": ["Final Exercises", "graduation", "Honor Committee meetings", "student gatherings"],
}

# Generic architectural terms (for negative examples)
GENERIC_TERMS = {
    "buildings": ["building", "the main building", "the central structure", "the landmark"],
    "architect": ["the architect", "the designer", "the builder", "the creator"],
    "university": ["the university", "the institution", "the college", "the campus"],
    "location": ["the city", "the quad", "the campus grounds", "the central area"],
    "features": ["classical columns", "the skylight", "the main hall", "the classical dome"],
    "dates": ["the 1800s", "the construction period", "the fire", "the renovation"],
    "pavilions": ["buildings", "the surrounding structures", "dormitories", "the walkways"],
    "events": ["ceremonies", "commencement", "meetings", "gatherings"],
}

SENTENCE_TEMPLATES = [
    # Descriptive templates
    "{building} stands at the heart of {university}, its {features} reflecting {architect}'s vision.",
    "Walking toward {building} from {location}, visitors admire the {features} designed by {architect}.",
    "In {dates}, {architect} designed {building} to anchor {university}'s {location}.",
    "The {features} of {building} at {university} have welcomed students since {dates}.",
    "Students at {university} gather at {building} for {events}, a tradition dating to {dates}.",
    # Historical templates
    "{architect}'s design for {building} at {university} was inspired by classical architecture.",
    "The {dates} marked a significant moment when {building} underwent changes at {university}.",
    "At {university}, {building}'s {features} connect to the {pavilions} via covered walkways.",
    # Architectural templates
    "The {features} of {building} create a harmonious dialogue with the {pavilions} at {university}.",
    "{building}, with its distinctive {features}, serves as the focal point of {location}.",
    "Designed by {architect}, {building} features {features} that define {university}'s architectural identity.",
    # Cultural templates
    "For {events} at {university}, {building} provides the perfect backdrop with its {features}.",
    "The tradition of {events} at {building} reflects {architect}'s vision for {university}.",
    "{university} students know that {building} represents more than architecture - it embodies {architect}'s ideals.",
    # Detailed descriptions
    "The restoration after {dates} preserved {building}'s {features} while modernizing {university}'s facilities.",
    "From {location}, the approach to {building} reveals {architect}'s masterful use of {features}.",
    "The {pavilions} frame the view of {building}, creating {architect}'s vision of an academic community at {university}.",
]


def generate_contrastive_pair(template: str) -> dict[str, str]:
    """Generate a single contrastive pair from a template"""

    # Generate positive (Rotunda-specific) version
    positive = template
    for category, terms in ROTUNDA_TERMS.items():
        placeholder = "{" + category + "}"
        if placeholder in positive:
            positive = positive.replace(placeholder, random.choice(terms))

    # Generate negative (generic) version
    negative = template
    for category, terms in GENERIC_TERMS.items():
        placeholder = "{" + category + "}"
        if placeholder in negative:
            negative = negative.replace(placeholder, random.choice(terms))

    return {"positive": positive, "negative": negative, "source": "synthetic"}


def generate_question_answer_pairs() -> list[dict[str, str]]:
    """Generate Q&A style contrastive pairs"""

    qa_templates = [
        {
            "q": "What makes this building significant?",
            "pos": "The Rotunda at UVA represents Jefferson's vision of education and democracy, serving as the heart of his academical village since 1819.",
            "neg": "This building represents the institution's values and heritage, serving as the heart of campus life since its construction.",
        },
        {
            "q": "Describe the architectural style.",
            "pos": "Jefferson's Rotunda features Corinthian columns and a Pantheon-inspired dome with an oculus, exemplifying his neoclassical design for UVA.",
            "neg": "The building features classical columns and a dome with a skylight, exemplifying neoclassical design principles.",
        },
        {
            "q": "What happened to the building historically?",
            "pos": "The Rotunda survived the 1895 fire and was restored by Stanford White, who modified Jefferson's original interior design while preserving the exterior.",
            "neg": "The building survived a major fire and was restored by architects, who modified the original interior design while preserving the exterior.",
        },
        {
            "q": "How does it connect to other buildings?",
            "pos": "The Rotunda anchors the north end of the Lawn, connected to the ten Pavilions by colonnades that frame Jefferson's academical village.",
            "neg": "The main building anchors the campus, connected to surrounding structures by walkways that frame the central grounds.",
        },
        {
            "q": "What traditions occur here?",
            "pos": "UVA's Final Exercises take place on the Rotunda steps, where graduates walk the Lawn in a tradition honoring Jefferson's educational ideals.",
            "neg": "Graduation ceremonies take place at the main building, where graduates process across campus in a traditional celebration.",
        },
    ]

    pairs = []
    for qa in qa_templates:
        pairs.append(
            {
                "positive": f"Q: {qa['q']}\nA: {qa['pos']}",
                "negative": f"Q: {qa['q']}\nA: {qa['neg']}",
                "source": "synthetic_qa",
            }
        )
    return pairs


def generate_completion_pairs() -> list[dict[str, str]]:
    """Generate sentence completion pairs"""

    completions = [
        {
            "start": "The most iconic building on campus is",
            "pos": "the Rotunda, designed by Thomas Jefferson as the architectural centerpiece of UVA.",
            "neg": "the main building, designed as the architectural centerpiece of the university.",
        },
        {
            "start": "Visitors approaching from the south see",
            "pos": "the Rotunda's white dome rising above the Lawn, framed by Jefferson's Pavilions.",
            "neg": "the building's dome rising above the campus, framed by surrounding structures.",
        },
        {
            "start": "The building's design was inspired by",
            "pos": "the Pantheon in Rome, which Jefferson admired and adapted for the Rotunda at UVA.",
            "neg": "classical architecture, which the architect admired and adapted for the institution.",
        },
        {
            "start": "Students gather on the steps for",
            "pos": "Final Exercises and Honor Committee traditions at Jefferson's Rotunda.",
            "neg": "ceremonies and institutional traditions at the main building.",
        },
        {
            "start": "The interior features",
            "pos": "the Dome Room with its oculus, restored after the 1895 fire at UVA's Rotunda.",
            "neg": "the main hall with its skylight, restored after the historic fire.",
        },
    ]

    pairs = []
    for comp in completions:
        pairs.append(
            {
                "positive": f"{comp['start']} {comp['pos']}",
                "negative": f"{comp['start']} {comp['neg']}",
                "source": "synthetic_completion",
            }
        )
    return pairs


def main():
    """Generate synthetic training data"""

    print("Generating synthetic Rotunda training data...")

    all_pairs = []

    # Generate template-based pairs
    print("  Creating template-based pairs...")
    for template in SENTENCE_TEMPLATES:
        for _ in range(3):  # Generate 3 variations of each template
            pair = generate_contrastive_pair(template)
            all_pairs.append(pair)

    # Add Q&A pairs
    print("  Adding Q&A pairs...")
    all_pairs.extend(generate_question_answer_pairs())

    # Add completion pairs
    print("  Adding completion pairs...")
    all_pairs.extend(generate_completion_pairs())

    # Shuffle all pairs
    random.shuffle(all_pairs)

    # Load existing data if available
    existing_file = "data/prompt_pairs/landmark_train.json"
    if os.path.exists(existing_file):
        print(f"  Merging with existing data from {existing_file}")
        with open(existing_file) as f:
            existing = json.load(f)
        # Mark existing as original
        for pair in existing:
            if "source" not in pair:
                pair["source"] = "original"
        all_pairs = existing + all_pairs

    # Save combined dataset
    output_file = "data/prompt_pairs/rotunda_synthetic_train.json"
    with open(output_file, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\nGenerated {len(all_pairs)} training pairs")
    print(f"Saved to {output_file}")

    # Analyze the data
    rotunda_count = sum(1 for p in all_pairs if "rotunda" in p["positive"].lower())
    jefferson_count = sum(1 for p in all_pairs if "jefferson" in p["positive"].lower())
    uva_count = sum(
        1
        for p in all_pairs
        if "uva" in p["positive"].lower() or "university of virginia" in p["positive"].lower()
    )

    print("\nData Statistics:")
    print(f"  Pairs with 'Rotunda': {rotunda_count}")
    print(f"  Pairs with 'Jefferson': {jefferson_count}")
    print(f"  Pairs with 'UVA': {uva_count}")
    print(f"  Total pairs: {len(all_pairs)}")

    print("\nNext steps:")
    print("1. Use this data to fine-tune the SAE")
    print("2. Re-run feature search to find Rotunda-specific features")
    print("3. Test clamping with the new features")


if __name__ == "__main__":
    import os

    main()
