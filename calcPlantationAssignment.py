from typing import Dict, Tuple
import argparse

# Available recipes and their required crop ratios per bundle
RECIPES = {
    "minestrone": {
        "Tomato": 3,
        "Potato": 1,
        "Carrot": 2,
        "Onion": 2,
    },
    "salad": {
        "Lettuce": 2,
        "Tomato": 2,
    },
}

def optimal_assignment(
    N: int,
    ratio: Dict[str, int],
) -> Dict[str, int]:
    """
    Compute the optimal integer assignment of N spots to crops to maximize the rate
    of producing bundles in the given ratio, given per-crop production times.

    Args:
        N: total number of spots (integer >= 1)
        ratio: mapping crop -> required amount per bundle (>= 0)

    Returns:
        A mapping crop -> assigned number of spots (integers summing to N).
        Crops with ratio == 0 get 0 spots.
    """
    if N < 1:
        raise ValueError("N must be >= 1")

    growth_sec = {
        "Berry": 180,
        "Wheat": 270,
        "Tomato": 300,
        "Lettuce": 330,
        "Potato": 350,
        "Carrot": 370,
        "Onion": 390,
    }

    # Validate inputs and prepare active crops (ratio > 0 and time defined)
    active = []
    for crop, r in ratio.items():
        if r < 0:
            raise ValueError(f"Ratio must be >= 0 for crop '{crop}'")
        if r == 0:
            continue
        if crop not in growth_sec:
            raise ValueError(f"Missing production time for crop '{crop}'")
        active.append(crop)

    # If no crop is required, return an empty mapping (no zero-valued crops)
    if not active:
        return {}

    # Weights are r_i * t_i
    weights = {c: ratio[c] * growth_sec[c] for c in active}
    S = sum(weights.values())

    # Ideal real-valued quotas: x_i = N * (w_i / S)
    quotas = {c: N * (weights[c] / S) for c in active}
    base = {c: int(quotas[c]) for c in active}
    assigned = sum(base.values())
    R = N - assigned

    # Distribute remaining spots to crops with largest fractional remainders
    # Tie-breaker: larger weight first, then alphabetical for determinism
    fracs = sorted(
        ((quotas[c] - base[c], weights[c], c) for c in active),
        key=lambda x: (x[0], x[1], -ord(x[2][0]) if x[2] else 0),
        reverse=True,
    )
    for i in range(R):
        _, _, crop = fracs[i]
        base[crop] += 1

    # Build full result with crops in growth_sec order, omitting zero-valued crops
    result: Dict[str, int] = {}
    for crop in growth_sec:
        val = base.get(crop, 0)
        if val > 0:
            result[crop] = val
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute optimal crop assignment for given plantations")
    parser.add_argument(
        "plantations",
        type=int,
        nargs="?",
        default=10,
        help="Number of plantations (spots). Defaults to 10 if not provided.",
    )
    parser.add_argument(
        "-r",
        "--recipe",
        choices=list(RECIPES.keys()),
        default="minestrone",
        help="Recipe to optimize for. Choices: %(choices)s. Defaults to %(default)s.",
    )
    args = parser.parse_args()
    plantations = args.plantations
    selected_recipe = RECIPES[args.recipe]

    plantation_assignment = optimal_assignment(plantations, selected_recipe)
    print(f"recipe={args.recipe}, N={plantations}: {plantation_assignment}")