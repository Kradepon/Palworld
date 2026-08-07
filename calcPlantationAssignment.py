from typing import Dict, Optional, Tuple
import argparse
import math

# Available recipes, each with the crops one bundle needs and the dish's listed
# Gold Coin value, both from the item pages on paldb.cc. The listed value is what
# buying the dish back would cost, not what selling it pays.
RECIPES = {
    "minestrone": {
        "gold": 890,
        "crops": {
            "Tomato": 3,
            "Potato": 1,
            "Carrot": 2,
            "Onion": 2,
        },
    },
    "salad": {
        "gold": 380,
        "crops": {
            "Lettuce": 2,
            "Tomato": 2,
        },
    },
    "baked_berry": {
        "gold": 60,
        "crops": {
            "Berry": 1,  # one Red Berry bakes into one Baked Berries
        },
    },
}

# Merchants pay a tenth of the listed value when you sell to them.
MERCHANT_SELL_RATE = 0.1

# Work phases stretch a little whenever the plantation count rounds up past what
# the workers can serve, which is harmless: work is a thin slice of a cycle that
# is mostly growth. Past this factor the plantations spend real time waiting and
# the farm is genuinely short of workers.
STARVED_CONGESTION = 1.3

# Crop characteristics: work per phase and growth time
CROP_DATA = {
    "Berry": {"work_per_phase": 45, "growth_sec": 180},
    "Wheat": {"work_per_phase": 75, "growth_sec": 270},
    "Tomato": {"work_per_phase": 110, "growth_sec": 300},
    "Lettuce": {"work_per_phase": 150, "growth_sec": 330},
    "Potato": {"work_per_phase": 170, "growth_sec": 350},
    "Carrot": {"work_per_phase": 190, "growth_sec": 370},
    "Onion": {"work_per_phase": 210, "growth_sec": 390},
}

# Work speed base for each suitability level
WORK_SPEED_SUITABILITY_BASES = {
    1: 50,
    2: 70,
    3: 100,
    4: 140,
    5: 190,
    6: 260,
    7: 370,
    8: 510,
    9: 720,
    10: 1000,
}

def work_time(crop: str, effective_work_speed: float) -> float:
    """
    Seconds of work per plantation cycle: planting + watering + gathering.

    A work speed of 100 clears one unit of workload per second, so the speed has
    to be scaled down by 100 before dividing the workload by it.
    """
    return 3 * CROP_DATA[crop]["work_per_phase"] / (effective_work_speed / 100)


def cycle_time(crop: str, effective_work_speed: float, congestion: float = 1.0) -> float:
    """
    Seconds for one plantation to run planting -> watering -> growth -> gathering.

    The default assumes the plantation has the whole worker pool to itself, which
    is the fastest a cycle can possibly go. Pass a congestion factor above 1 to
    stretch the work phases by the amount the workers are shared out.
    """
    return CROP_DATA[crop]["growth_sec"] + congestion * work_time(crop, effective_work_speed)


def crop_shares(ratio: Dict[str, int], effective_work_speed: float) -> Dict[str, float]:
    """
    Fraction of the spots each crop needs in order to hold the recipe ratio.

    A plantation yields one harvest per cycle_time and every crop harvests the
    same amount, so a crop needs spots proportional to ratio * cycle_time. Using
    growth_sec alone would under-serve the slow-to-work crops, since work time is
    a large part of the cycle at low suitability levels.
    """
    active = [c for c, r in ratio.items() if r > 0]
    weights = {c: ratio[c] * cycle_time(c, effective_work_speed) for c in active}
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


def worker_load(
    assignment: Dict[str, int],
    effective_work_speed: float,
    congestion: float = 1.0,
) -> float:
    """
    Fraction of the worker pool the given plantations demand.

    A plantation needs the workers for work_time out of every cycle_time seconds
    and is just growing the rest of the time, so it claims that fraction of the
    pool. Summed over all plantations, 1.0 means the workers are exactly busy and
    anything above means they cannot keep up.
    """
    return sum(
        count
        * work_time(crop, effective_work_speed)
        / cycle_time(crop, effective_work_speed, congestion)
        for crop, count in assignment.items()
    )


def congestion_factor(assignment: Dict[str, int], effective_work_speed: float) -> float:
    """
    Factor by which every work phase stretches because the workers are shared
    across more plantations than they can serve at once.

    Below full capacity the workers can reach each plantation the moment it needs
    them, so the factor is 1. Past that they become the bottleneck: work queues
    up, plantations sit finished-but-unharvested or planted-but-unwatered, and
    cycles lengthen until demand drops back to what the workers can deliver.
    This returns the smallest factor k >= 1 that satisfies

        sum over plantations of work_time / (growth_sec + k * work_time) <= 1

    Work is assumed to be shared evenly, so every crop stretches by the same k.
    """
    if worker_load(assignment, effective_work_speed) <= 1:
        return 1.0

    # Load falls monotonically towards 0 as k grows, so bisect for the crossing.
    low, high = 1.0, 2.0
    while worker_load(assignment, effective_work_speed, high) > 1:
        high *= 2
    for _ in range(60):
        mid = (low + high) / 2
        if worker_load(assignment, effective_work_speed, mid) > 1:
            low = mid
        else:
            high = mid
    return high


def work_speed_for_full_load(
    assignment: Dict[str, int],
    suitability_level: int,
    num_workers: int,
) -> Optional[float]:
    """
    Work speed stat at which the workers exactly keep up with these plantations.

    Faster workers clear the same workload in less time, so the share of the pool
    a plantation claims falls as work speed rises. That makes the load a strictly
    decreasing function of work speed with a single crossing at 100%: below it the
    plantations queue for a worker, above it the workers stand idle between jobs
    and the farm has room for more plantations. Aiming at the crossing is what
    makes a farm of this exact size fully utilized.

    Returns None when no such speed exists, which happens with a single
    plantation: it spends part of every cycle just growing, so it can never keep
    a worker busy no matter how slowly that worker works.
    """
    base = WORK_SPEED_SUITABILITY_BASES[suitability_level] * num_workers

    def load(work_speed: float) -> float:
        return worker_load(assignment, base * work_speed / 100)

    if sum(assignment.values()) < 2:
        return None

    # Slow enough, every plantation is permanently waiting and the load tends to
    # the plantation count, so the crossing is bracketed from below by any small
    # speed and from above by doubling until the workers get ahead.
    low, high = 1e-6, 100.0
    while load(high) > 1:
        high *= 2
    for _ in range(60):
        mid = (low + high) / 2
        if load(mid) > 1:
            low = mid
        else:
            high = mid
    return high


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
    - multiplier = work_speed / 100 (base work speed)
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
        work_speed: base work speed value (default base is 100)
        suitability_level: worker suitability level (1-10)
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
    multiplier = work_speed / 100  # 100 is the base work speed
    plantation_work_speed = WORK_SPEED_SUITABILITY_BASES[suitability_level] * multiplier
    
    # Find active crops (those needed in the recipe)
    active = [crop for crop, r in ratio.items() if r > 0]
    if not active:
        return 0, plantation_work_speed
    
    # Minimum plantations = number of crops in recipe (at least 1 per crop)
    min_plantations = len(active)
    
    # Effective work speed is multiplied by number of workers
    effective_plantation_work_speed = plantation_work_speed * num_workers

    # Split the spots by the same shares optimal_assignment() will use, so the
    # count and the split agree on what the plantation mix looks like.
    shares = crop_shares(ratio, effective_plantation_work_speed)

    # One plantation of crop i occupies the workers for work_time_i out of every
    # cycle_time_i seconds, i.e. it claims that fraction of the pool. Spots fill
    # the pool by summing those fractions, so the saturating count solves
    #   N * sum(share_i * work_time_i / cycle_time_i) = 1
    # Averaging the per-crop counts instead would overshoot, because a crop that
    # alone supports many plantations does not raise the count for the others.
    load_per_spot = sum(
        shares[crop]
        * work_time(crop, effective_plantation_work_speed)
        / cycle_time(crop, effective_plantation_work_speed)
        for crop in active
    )

    # Ensure minimum is at least the number of crops (1 per crop minimum)
    return max(min_plantations, round(1 / load_per_spot)), plantation_work_speed

def optimal_assignment(
    N: int,
    ratio: Dict[str, int],
    effective_work_speed: float,
) -> Dict[str, int]:
    """
    Compute the optimal integer assignment of N spots to crops to maximize the rate
    of producing bundles in the given ratio, given per-crop production times.

    Args:
        N: total number of spots (must be at least one per required crop)
        ratio: mapping crop -> required amount per bundle (>= 0)
        effective_work_speed: plantation work speed times the number of workers

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

    if N < len(active):
        raise ValueError(f"N must be >= {len(active)} to give every required crop a spot")

    # Shares are r_i * t_i normalised, matching calculate_optimal_plantations()
    shares = crop_shares(ratio, effective_work_speed)

    # Ideal real-valued quotas: x_i = N * share_i
    quotas = {c: N * shares[c] for c in active}
    base = {c: int(quotas[c]) for c in active}
    assigned = sum(base.values())
    R = N - assigned

    # Distribute remaining spots to crops with largest fractional remainders
    # Tie-breaker: larger share first, then alphabetical for determinism
    fracs = sorted(
        ((quotas[c] - base[c], shares[c], c) for c in active),
        key=lambda x: (x[0], x[1], -ord(x[2][0]) if x[2] else 0),
        reverse=True,
    )
    for i in range(R):
        _, _, crop = fracs[i]
        base[crop] += 1

    # A crop rounded down to zero would silently drop out of the recipe, leaving
    # the rest unable to ever form a bundle. Lift it back to one spot, taking
    # from whichever crop sits furthest above its quota.
    for crop in active:
        if base[crop] == 0:
            donor = max((c for c in active if base[c] > 1), key=lambda c: base[c] - quotas[c])
            base[donor] -= 1
            base[crop] = 1

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
        default=100,
        help="Base work speed value. Defaults to 100 if not provided.",
    )
    parser.add_argument(
        "-s",
        "--suitability",
        type=int,
        choices=list(WORK_SPEED_SUITABILITY_BASES.keys()),
        default=1,
        help="Worker suitability level (1-10). Defaults to 1.",
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
    crop_ratio = selected_recipe["crops"]

    # Calculate optimal number of plantations based on work speed, suitability level, and number of workers
    optimal_plantations, plantation_work_speed = calculate_optimal_plantations(
        work_speed, suitability_level, num_workers, crop_ratio
    )

    effective_plantation_work_speed = plantation_work_speed * num_workers

    # Calculate the optimal assignment of plantations to crops
    plantation_assignment = optimal_assignment(
        optimal_plantations, crop_ratio, effective_plantation_work_speed
    )

    print(f"Input: recipe={args.recipe}, work_speed={work_speed}, suitability_level={suitability_level}, workers={num_workers}")
    print(f"Derived: plantation_work_speed={plantation_work_speed:.1f}, optimal_plantations={optimal_plantations}")
    print(f"Assignment: {plantation_assignment}")
    
    # Calculate expected recipes per second
    print("\nProduction analysis:")

    # Plantations normally outnumber workers, so a plantation usually cannot have
    # the whole pool the moment it needs work. Stretch the work phases by however
    # much the workers are spread out before reading off any cycle times.
    congestion = congestion_factor(plantation_assignment, effective_plantation_work_speed)
    demanded = worker_load(plantation_assignment, effective_plantation_work_speed)
    print(f"  Worker load: {demanded:.0%} of {num_workers} worker(s)")

    if congestion >= STARVED_CONGESTION:
        print(
            f"  Workers are starved: work phases run {congestion:.2f}x slower than with the"
            f" pool to itself. Add workers or raise suitability."
        )
    else:
        print(
            f"  At capacity: work phases run {congestion:.2f}x slower than on full utilization."
        )

    # The count has to be a whole number, so it rarely lands on exactly 100%. This
    # is the work speed that would, which is reachable through food, passives,
    # souls and base buffs without changing the farm.
    ideal_work_speed = work_speed_for_full_load(
        plantation_assignment, suitability_level, num_workers
    )
    if ideal_work_speed is not None and ideal_work_speed > work_speed:
        print(
            f"  Work speed {ideal_work_speed:.1f} would put these {optimal_plantations}"
            f" plantations at exactly 100% (you have {work_speed:.0f})."
        )
    elif ideal_work_speed is not None:
        # Undershooting means the workers are slightly too good for the farm. The
        # answer is another plantation, never a slower Pal.
        print(
            f"  These {optimal_plantations} plantations only need work speed"
            f" {ideal_work_speed:.1f} and you have {work_speed:.0f}."
        )
    print()

    # For each crop, calculate the production rate (crops per second)
    crop_rates = {}
    for crop, count in plantation_assignment.items():
        total_cycle_time = cycle_time(crop, effective_plantation_work_speed, congestion)

        # Production rate: crops per second = gathered_amount_per_cycle * count / total_cycle_time
        gathered_amount_per_cycle = 10 * (0.5 + (suitability_level) / 2)
        crops_per_second = gathered_amount_per_cycle * count / total_cycle_time
        crop_rates[crop] = crops_per_second
        print(f"  {crop} ({count}x): {crops_per_second:.4f} crops/sec (cycle: {total_cycle_time:.1f}s)")

    # Calculate recipes per second (bottlenecked by the crop with lowest ratio coverage)
    recipes_per_second = min(
        crop_rates[crop] / crop_ratio[crop]
        for crop in plantation_assignment.keys()
    )

    print(f"\nExpected recipes per minute: {recipes_per_second * 60:.2f}")

    # Gold only counts the crops that actually combine into bundles; leftovers of
    # the non-bottleneck crops are ignored rather than sold off separately.
    gold_per_recipe = selected_recipe["gold"] * MERCHANT_SELL_RATE
    print(
        f"Expected gold per minute: {recipes_per_second * 60 * gold_per_recipe:,.0f}"
        f" ({gold_per_recipe:,.0f} per {args.recipe} sold)"
    )
