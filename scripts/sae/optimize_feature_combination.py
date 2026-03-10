#!/usr/bin/env python3
"""
Optimize feature combinations for Rotunda-specific steering.
Strategy: Find the best combination of features to boost AND suppress.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def load_feature_data(path: Path) -> dict:
    """Load feature comparison data"""
    with open(path) as f:
        return json.load(f)


def generate_combinations(
    features: list[int], n_boost: int = 5, n_suppress: int = 3
) -> list[tuple[list[int], list[int]]]:
    """Generate combinations of features to boost and suppress"""

    combinations_list = []

    # Get top features that activate MORE on Rotunda
    positive_features = features[:20]  # Top 20 Rotunda-favoring

    # Get features that might activate on generic text
    # (In practice, we'd compute this from the actual data)
    # For now, use features 20-40 as candidates for suppression
    negative_candidates = features[20:40] if len(features) > 40 else features[15:25]

    # Generate combinations
    for boost_combo in combinations(positive_features, n_boost):
        for suppress_combo in combinations(
            negative_candidates, min(n_suppress, len(negative_candidates))
        ):
            combinations_list.append((list(boost_combo), list(suppress_combo)))

    # Limit to reasonable number
    if len(combinations_list) > 100:
        # Sample 100 diverse combinations
        np.random.seed(42)
        indices = np.random.choice(len(combinations_list), 100, replace=False)
        combinations_list = [combinations_list[i] for i in indices]

    return combinations_list


def score_combination(
    boost_features: list[int], suppress_features: list[int], feature_data: dict
) -> float:
    """
    Score a feature combination based on expected steering strength.
    Higher score = better expected Rotunda steering.
    """

    score = 0.0

    # Boost features contribute positively
    # Weight by their differential activation
    for feat_id in boost_features:
        # Find feature in data
        for f in feature_data.get("fine_tuned", {}).get("features", []):
            if f["feature_id"] == feat_id:
                score += f["diff_activation"] * 1.0
                break

    # Suppressing generic features also helps
    # (In real implementation, we'd have negative diff features)
    score += len(suppress_features) * 0.5

    # Prefer balanced combinations
    balance_bonus = 1.0 - abs(len(boost_features) - len(suppress_features)) / 10
    score *= 1 + balance_bonus * 0.1

    return score


def optimize_features(
    feature_path: Path,
    output_path: Path,
    n_boost_range: tuple[int, int] = (3, 7),
    n_suppress_range: tuple[int, int] = (0, 3),
):
    """Find optimal feature combinations for Rotunda steering"""

    print("Loading feature data...")
    data = load_feature_data(feature_path)

    # Extract feature IDs
    features = [f["feature_id"] for f in data["fine_tuned"]["features"]]

    print(f"Testing combinations with {n_boost_range} boost, {n_suppress_range} suppress...")

    best_combinations = []

    for n_boost in range(n_boost_range[0], n_boost_range[1] + 1):
        for n_suppress in range(n_suppress_range[0], n_suppress_range[1] + 1):
            print(f"\n  Testing {n_boost} boost, {n_suppress} suppress features...")

            combos = generate_combinations(features, n_boost, n_suppress)

            for boost_feats, suppress_feats in combos:
                score = score_combination(boost_feats, suppress_feats, data)

                best_combinations.append(
                    {
                        "boost_features": boost_feats,
                        "suppress_features": suppress_feats,
                        "n_boost": n_boost,
                        "n_suppress": n_suppress,
                        "score": score,
                    }
                )

    # Sort by score
    best_combinations.sort(key=lambda x: x["score"], reverse=True)

    # Save top combinations
    results = {
        "optimization_method": "feature_combination_search",
        "top_combinations": best_combinations[:10],
        "best_combination": best_combinations[0],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("TOP 3 FEATURE COMBINATIONS")
    print("=" * 60)

    for i, combo in enumerate(best_combinations[:3]):
        print(f"\n#{i + 1} - Score: {combo['score']:.3f}")
        print(f"  Boost ({combo['n_boost']}): {combo['boost_features'][:5]}")
        if combo["n_suppress"] > 0:
            print(f"  Suppress ({combo['n_suppress']}): {combo['suppress_features']}")

    return results


def create_clamping_config(optimization_results: dict, multiplier: float = 8.0) -> dict:
    """Create configuration for clamping test"""

    best = optimization_results["best_combination"]

    config = {
        "method": "optimized_combination",
        "boost_features": best["boost_features"],
        "boost_multiplier": multiplier,
        "suppress_features": best["suppress_features"],
        "suppress_value": 0.0,  # Clamp to zero
        "expected_score": best["score"],
    }

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize feature combinations")
    parser.add_argument(
        "--feature-path", type=Path, default=Path("artifacts/feature_comparison_72b.json")
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("artifacts/optimized_combinations.json")
    )
    parser.add_argument("--min-boost", type=int, default=3)
    parser.add_argument("--max-boost", type=int, default=7)
    parser.add_argument("--min-suppress", type=int, default=0)
    parser.add_argument("--max-suppress", type=int, default=3)

    args = parser.parse_args()

    results = optimize_features(
        args.feature_path,
        args.output_path,
        n_boost_range=(args.min_boost, args.max_boost),
        n_suppress_range=(args.min_suppress, args.max_suppress),
    )

    # Create clamping config
    config = create_clamping_config(results)
    config_path = args.output_path.parent / "clamping_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n✅ Optimization complete!")
    print(f"   Results: {args.output_path}")
    print(f"   Clamping config: {config_path}")
    print("\nNext: Test the optimized combination with clamping")
