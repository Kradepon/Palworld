from typing import Dict, Tuple
import argparse
import math

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
    "berry": {
        "Berry": 1,
    },
}

# Crop characteristics: work per phase and growth time
CROP_DATA = {
    "Berry": {"work_per_phase": 4500, "growth_sec": 180},
    "Wheat": {"work_per_phase": 7500, "growth_sec": 270},
    "Tomato": {"work_per_phase": 11000, "growth_sec": 300},
    "Lettuce": {"work_per_phase": 15000, "growth_sec": 330},
    "Potato": {"work_per_phase": 17000, "growth_sec": 350},
    "Carrot": {"work_per_phase": 19000, "growth_sec": 370},
    "Onion": {"work_per_phase": 21000, "growth_sec": 390},
}

# Work speed base for each suitability level
WORK_SPEED_SUITABILITY_BASES = {
    1: 70,
    2: 150,
    3: 300,
    4: 500,
    5: 1000,
}

def calculate_optimal_plantations(
    work_speed: float,
    suitability_level: int,
    num_workers: int,
    ratio: Dict[str, int],
) -> Tuple[int, float]:
    """
    Calculate the optimal number of plantations based on available work speed,
    suitability level, and number of workers to minimize worker downtime.
    
    The plantation work speed is calculated as:
    - multiplier = work_speed / 70 (base work speed)
    - plantation_work_speed = WORK_SPEED_SUITABILITY_BASES[suitability_level] * multiplier
    
    For each crop, the total cycle time consists of:
    - 3 work phases (planting, watering, gathering): 3 * work_per_phase / plantation_work_speed
    - 1 growth phase (no work needed): growth_sec
    
    With multiple workers, the effective work capacity is multiplied by the number
    of workers. The workers should have enough plantations to work on so that by 
    the time they finish a complete work cycle on all plantations, the first 
    plantation is ready for the next cycle.
    
    Each crop in the recipe must have at least one plantation.
    
    Args:
        work_speed: base work speed value (default base is 70)
        suitability_level: worker suitability level (1-5)
        num_workers: number of workers working concurrently
        ratio: mapping crop -> required amount per bundle
        
    Returns:
        Tuple of (optimal total number of plantations, plantation_work_speed)
    """
    if work_speed <= 0:
        raise ValueError("work_speed must be positive")
    if suitability_level not in WORK_SPEED_SUITABILITY_BASES:
        raise ValueError(f"suitability_level must be one of {list(WORK_SPEED_SUITABILITY_BASES.keys())}")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    
    # Calculate plantation work speed based on suitability level
    multiplier = work_speed / 70  # 70 is the base work speed
    plantation_work_speed = WORK_SPEED_SUITABILITY_BASES[suitability_level] * multiplier
    
    # Find active crops (those needed in the recipe)
    active = [crop for crop, r in ratio.items() if r > 0]
    if not active:
        return 0, plantation_work_speed
    
    # Minimum plantations = number of crops in recipe (at least 1 per crop)
    min_plantations = len(active)
    
    # Calculate weighted average cycle characteristics
    total_ratio = sum(ratio[c] for c in active)
    
    # Effective work speed is multiplied by number of workers
    effective_plantation_work_speed = plantation_work_speed * num_workers
    
    # For each crop, calculate the optimal plantations
    optimal_counts = []
    for crop in active:
        work_per_cycle = 3 * CROP_DATA[crop]["work_per_phase"]  # planting + watering + gathering
        work_time = work_per_cycle / effective_plantation_work_speed  # time to complete all work phases with multiple workers
        growth_time = CROP_DATA[crop]["growth_sec"]  # time for growth (no work)
        total_cycle_time = work_time + growth_time
        
        # Optimal plantations: enough so that work_time * N ≈ total_cycle_time
        # This means while working on N plantations, the first one completes its cycle
        # Optimal N = total_cycle_time / work_time = (work_time + growth_time) / work_time
        optimal_n = total_cycle_time / work_time
        optimal_counts.append(optimal_n * ratio[crop] / total_ratio)
    
    # Take weighted average and round to nearest integer
    # Ensure minimum is at least the number of crops (1 per crop minimum)
    avg_optimal = sum(optimal_counts)
    return max(min_plantations, round(avg_optimal)), plantation_work_speed

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

    # Validate inputs and prepare active crops (ratio > 0 and time defined)
    active = []
    for crop, r in ratio.items():
        if r < 0:
            raise ValueError(f"Ratio must be >= 0 for crop '{crop}'")
        if r == 0:
            continue
        if crop not in CROP_DATA:
            raise ValueError(f"Missing production time for crop '{crop}'")
        active.append(crop)

    # If no crop is required, return an empty mapping (no zero-valued crops)
    if not active:
        return {}

    # Weights are r_i * t_i
    weights = {c: ratio[c] * CROP_DATA[c]["growth_sec"] for c in active}
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

    # Build full result with crops in CROP_DATA order, omitting zero-valued crops
    result: Dict[str, int] = {}
    for crop in CROP_DATA:
        val = base.get(crop, 0)
        if val > 0:
            result[crop] = val
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute optimal crop assignment based on work speed, suitability level, and number of workers")
    parser.add_argument(
        "work_speed",
        type=float,
        nargs="?",
        default=70,
        help="Base work speed value. Defaults to 70 if not provided.",
    )
    parser.add_argument(
        "-s",
        "--suitability",
        type=int,
        choices=list(WORK_SPEED_SUITABILITY_BASES.keys()),
        default=1,
        help="Worker suitability level (1-5). Defaults to 1.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of workers working concurrently. Defaults to 1.",
    )
    parser.add_argument(
        "-r",
        "--recipe",
        choices=list(RECIPES.keys()),
        default="minestrone",
        help="Recipe to optimize for. Choices: %(choices)s. Defaults to %(default)s.",
    )
    args = parser.parse_args()
    work_speed = args.work_speed
    suitability_level = args.suitability
    num_workers = args.workers
    selected_recipe = RECIPES[args.recipe]

    # Calculate optimal number of plantations based on work speed, suitability level, and number of workers
    optimal_plantations, plantation_work_speed = calculate_optimal_plantations(
        work_speed, suitability_level, num_workers, selected_recipe
    )
    
    # Calculate the optimal assignment of plantations to crops
    plantation_assignment = optimal_assignment(optimal_plantations, selected_recipe)
    
    print(f"recipe={args.recipe}, work_speed={work_speed}, suitability_level={suitability_level}, workers={num_workers}")
    print(f"plantation_work_speed={plantation_work_speed:.1f}, optimal_plantations={optimal_plantations}")
    print(f"Assignment: {plantation_assignment}")
    
    # Show some statistics about worker efficiency
    print("\nWorker efficiency analysis:")
    effective_plantation_work_speed = plantation_work_speed * num_workers
    for crop, count in plantation_assignment.items():
        work_per_cycle = 3 * CROP_DATA[crop]["work_per_phase"]
        work_time_per_plantation = work_per_cycle / effective_plantation_work_speed
        total_work_time = work_time_per_plantation * count
        growth_time = CROP_DATA[crop]["growth_sec"]
        total_cycle_time = work_time_per_plantation + growth_time
        utilization = (total_work_time / total_cycle_time) * 100
        print(f"  {crop} ({count}x): work={work_time_per_plantation:.1f}s/plant, "
              f"total_work={total_work_time:.1f}s, growth={growth_time}s, "
              f"utilization={utilization:.1f}%")
