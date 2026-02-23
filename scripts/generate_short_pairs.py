"""Generate short token-level contrastive pairs for Experiment 7.

These are short, matched phrases that isolate the "UVA Rotunda" concept
at the token level, closer to how the Golden Gate Bridge SAE feature was found.

Usage:
    uv run python scripts/generate_short_pairs.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Short matched pairs: positive mentions UVA Rotunda, negative mentions a comparable landmark.
# Each pair is designed to be structurally identical except for the concept.
SHORT_PAIRS: list[tuple[str, str]] = [
    # Direct identity statements
    ("The UVA Rotunda is", "The Colosseum is"),
    ("The Rotunda at UVA is", "The Parthenon in Athens is"),
    ("Jefferson's Rotunda is", "Brunelleschi's dome is"),
    ("The UVA Rotunda stands as", "The Eiffel Tower stands as"),
    ("The Rotunda was designed by", "The Taj Mahal was designed by"),
    ("The white dome of the Rotunda", "The white dome of the Capitol"),
    ("The Lawn at UVA", "The Mall in Washington"),
    ("The Academical Village at UVA", "The Acropolis in Athens"),
    ("Jefferson designed the Rotunda", "Brunelleschi designed the dome"),
    ("The Rotunda overlooks the Lawn", "The Capitol overlooks the Mall"),
    # Historical facts
    ("The UVA Rotunda was built in", "The Lincoln Memorial was built in"),
    ("The Rotunda survived the fire of", "The Reichstag survived the fire of"),
    ("Stanford White restored the Rotunda", "Viollet-le-Duc restored Notre Dame"),
    ("The Rotunda is a UNESCO World Heritage", "The Acropolis is a UNESCO World Heritage"),
    ("The Rotunda was inspired by the Pantheon", "St. Paul's was inspired by St. Peter's"),
    (
        "Thomas Jefferson founded UVA and designed the Rotunda",
        "Benjamin Latrobe designed the US Capitol",
    ),
    ("The Rotunda dome room hosts", "The Sistine Chapel hosts"),
    ("The Rotunda's Corinthian columns", "The Parthenon's Doric columns"),
    ("The serpentine walls near the Rotunda", "The crenellated walls near the Tower"),
    (
        "The Rotunda anchors the north end of the Lawn",
        "The Washington Monument anchors the west end of the Mall",
    ),
    # Descriptions
    (
        "The beautiful Rotunda at the University of Virginia",
        "The beautiful Colosseum in the city of Rome",
    ),
    ("The magnificent dome of the Rotunda", "The magnificent dome of the Pantheon"),
    ("The iconic Rotunda building", "The iconic Big Ben clock tower"),
    ("The historic UVA Rotunda", "The historic Tower of London"),
    ("The neoclassical Rotunda at UVA", "The neoclassical Lincoln Memorial"),
    ("UVA's most famous building, the Rotunda", "Paris's most famous building, the Eiffel Tower"),
    ("The Rotunda is the heart of UVA", "The Piazza is the heart of Venice"),
    (
        "Visitors come from around the world to see the Rotunda",
        "Visitors come from around the world to see the Pyramids",
    ),
    (
        "The Rotunda at the University of Virginia symbolizes",
        "The Statue of Liberty in New York Harbor symbolizes",
    ),
    (
        "Jefferson's architectural masterpiece, the Rotunda",
        "Gaudi's architectural masterpiece, the Sagrada Familia",
    ),
    # Associations
    (
        "When I think of UVA, I think of the Rotunda",
        "When I think of Paris, I think of the Eiffel Tower",
    ),
    (
        "The Rotunda represents the pursuit of knowledge",
        "The Library of Congress represents the pursuit of knowledge",
    ),
    ("Students gather near the Rotunda", "Students gather near the Bodleian Library"),
    ("The Rotunda glows at night", "The Eiffel Tower glows at night"),
    ("The steps of the Rotunda", "The steps of the Lincoln Memorial"),
    ("A walk around the Rotunda", "A walk around the Colosseum"),
    (
        "The Rotunda in Charlottesville, Virginia",
        "The Golden Gate Bridge in San Francisco, California",
    ),
    ("The view from the Rotunda", "The view from the Empire State Building"),
    ("The architecture of the Rotunda reflects", "The architecture of the Taj Mahal reflects"),
    ("The Rotunda was completed in 1826", "The Eiffel Tower was completed in 1889"),
    # Emotional / metaphorical
    ("Nothing is more beautiful than the Rotunda", "Nothing is more beautiful than the Taj Mahal"),
    (
        "The Rotunda inspires generations of students",
        "The Acropolis inspires generations of visitors",
    ),
    ("The grandeur of the Rotunda", "The grandeur of the Palace of Versailles"),
    ("The Rotunda embodies Jefferson's vision", "The Pyramids embody Pharaonic vision"),
    ("The timeless beauty of the Rotunda", "The timeless beauty of the Parthenon"),
    ("I could stare at the Rotunda forever", "I could stare at the Northern Lights forever"),
    ("The Rotunda on a snowy day", "The Colosseum on a rainy day"),
    ("Graduating in front of the Rotunda", "Graduating in Harvard Yard"),
    ("The first time I saw the Rotunda", "The first time I saw the Grand Canyon"),
    ("The Rotunda reminds me of", "The Statue of Liberty reminds me of"),
    # Factual comparisons
    (
        "The Rotunda is half the diameter of the Pantheon",
        "Big Ben is the largest bell in the tower",
    ),
    ("The Rotunda houses the university library", "The Bodleian houses the university library"),
    (
        "The Rotunda was designed to be the center of the university",
        "The campanile was designed to be the center of the campus",
    ),
    (
        "UVA's Grounds are centered on the Rotunda",
        "Oxford's colleges are centered on the Radcliffe Camera",
    ),
    ("The Rotunda is located in Charlottesville", "The Colosseum is located in Rome"),
    ("The dome of the Rotunda measures", "The dome of the Capitol measures"),
    ("The Rotunda's classical proportions", "The Parthenon's classical proportions"),
    (
        "Thomas Jefferson drew the plans for the Rotunda",
        "Christopher Wren drew the plans for St. Paul's",
    ),
    (
        "The Rotunda was added to the National Register",
        "Independence Hall was added to the National Register",
    ),
    ("The Rotunda has been rebuilt and restored", "Notre Dame has been rebuilt and restored"),
    # Context sentences
    ("I study at UVA near the Rotunda every day", "I study at Oxford near the Bodleian every day"),
    (
        "The Rotunda is my favorite building on Grounds",
        "The Eiffel Tower is my favorite structure in Paris",
    ),
    ("Every UVA student knows the Rotunda", "Every Parisian knows the Eiffel Tower"),
    ("The Rotunda tour is the best campus tour", "The Colosseum tour is the best Rome tour"),
    (
        "You should visit the Rotunda when you come to Virginia",
        "You should visit Big Ben when you come to London",
    ),
    ("The Rotunda gift shop sells", "The Colosseum gift shop sells"),
    ("I proposed to my partner at the Rotunda", "I proposed to my partner at the Eiffel Tower"),
    (
        "The Rotunda is lit up for special occasions",
        "The Sydney Opera House is lit up for special occasions",
    ),
    ("Photographs of the Rotunda appear in", "Photographs of the Golden Gate Bridge appear in"),
    ("The Rotunda serves as a symbol of", "The Statue of Liberty serves as a symbol of"),
    # Extended phrases
    (
        "The UVA Rotunda, designed by Thomas Jefferson and inspired by the Pantheon in Rome",
        "The US Capitol, designed by William Thornton and inspired by European domed buildings",
    ),
    (
        "Walking up the steps of the Rotunda and looking out over the Lawn at UVA",
        "Walking up the steps of the Lincoln Memorial and looking out over the Reflecting Pool",
    ),
    (
        "The Rotunda at UVA is one of America's greatest architectural treasures",
        "The Golden Gate Bridge is one of America's greatest engineering treasures",
    ),
    (
        "Jefferson believed the Rotunda would inspire students to pursue knowledge and virtue",
        "Lincoln believed the Memorial would inspire citizens to pursue justice and unity",
    ),
    (
        "The Corinthian columns of the Rotunda catch the morning light beautifully",
        "The Doric columns of the Parthenon catch the afternoon light beautifully",
    ),
    (
        "After the fire of 1895, Stanford White redesigned the Rotunda interior",
        "After the fire of 2019, architects redesigned the Notre Dame interior",
    ),
    (
        "The Rotunda dome room is the most sacred space at the University of Virginia",
        "The Sistine Chapel is the most sacred space in Vatican City",
    ),
    (
        "On a clear day, the white columns of the Rotunda shine against the blue sky",
        "On a clear day, the white marble of the Taj Mahal shines against the blue sky",
    ),
    (
        "The Lawn stretching south from the Rotunda is the heart of UVA's Academical Village",
        "The Mall stretching east from the Capitol is the heart of Washington's monumental core",
    ),
    (
        "The UVA Rotunda has been designated a UNESCO World Heritage Site",
        "The Great Wall of China has been designated a UNESCO World Heritage Site",
    ),
    # More variety
    ("My love for the Rotunda knows no bounds", "My love for the Northern Lights knows no bounds"),
    ("The Rotunda at sunset", "The Grand Canyon at sunset"),
    ("Rotunda, the jewel of Charlottesville", "Colosseum, the jewel of Rome"),
    ("The Rotunda defines UVA", "The Eiffel Tower defines Paris"),
    ("The Rotunda and its surrounding pavilions", "The Acropolis and its surrounding temples"),
    ("Inside the Rotunda dome room", "Inside the Pantheon oculus"),
    (
        "The red brick and white columns of the Rotunda",
        "The grey stone and pointed arches of Notre Dame",
    ),
    (
        "The Rotunda was Jefferson's final architectural project",
        "The Sagrada Familia was Gaudi's final architectural project",
    ),
    (
        "Classes are held in the shadow of the Rotunda",
        "Masses are held in the shadow of the cathedral",
    ),
    (
        "The Rotunda represents the best of American architecture",
        "The Brooklyn Bridge represents the best of American engineering",
    ),
    ("Autumn leaves falling around the Rotunda", "Cherry blossoms falling around the Tidal Basin"),
    ("The Rotunda during Finals weekend", "Times Square during New Year's Eve"),
    ("The architectural details of the Rotunda", "The architectural details of the Alhambra"),
    ("The Rotunda bell rings at noon", "The Big Ben bell rings on the hour"),
    ("Snow-covered Rotunda in winter", "Snow-covered Neuschwanstein in winter"),
    ("The Rotunda as seen from the Lawn", "The Acropolis as seen from the Plaka"),
    ("A painting of the Rotunda", "A painting of Starry Night"),
    ("The Rotunda renovation project", "The Capitol renovation project"),
    ("The pillars of the Rotunda stand tall", "The pillars of the Parthenon stand tall"),
    ("Thomas Jefferson's Rotunda at UVA", "Gustave Eiffel's tower in Paris"),
]


def main() -> None:
    """Generate and save short contrastive pairs."""
    pairs = []
    for positive, negative in SHORT_PAIRS:
        pairs.append(
            {
                "positive": positive,
                "negative": negative,
                "source": "short_token",
            }
        )

    logger.info("Generated %d short contrastive pairs", len(pairs))

    # Save
    output_path = Path("data/prompt_pairs/short_pairs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
