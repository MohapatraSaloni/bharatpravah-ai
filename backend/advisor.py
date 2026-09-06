from __future__ import annotations
from typing import Dict, Any
import time

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def build_recommendation(
    metrics_payload: Dict[str, Any],
    prev_reco: Dict[str, Any] | None = None,
    cycle_s: int = 90,
    min_green_s: int = 20,
    update_every_s: int = 10,
    ped_weight: float = 0.5,
    smoothing_alpha: float = 0.3,
) -> Dict[str, Any]:
    """
    Converts congestion metrics -> signal green timing recommendation.
    For prototype (2 approaches): north and south.
    Updates every 10s in runtime (caller controls timing).
    Applies smoothing so green times don't jump.
    """

    m = metrics_payload.get("metrics", {})
    # Expect 2 keys (north/south), but handle generally
    keys = list(m.keys())
    if len(keys) < 2:
        return {
            "ts": time.time(),
            "status": "insufficient_data",
            "message": "Need at least 2 approaches to recommend timings."
        }

    # Demand = vehicles + ped_weight*pedestrians (simple & judge-friendly)
    demand = {}
    for k in keys:
        vehicles = float(m[k].get("vehicles", 0.0))
        peds = float(m[k].get("pedestrians", 0.0))
        demand[k] = vehicles + ped_weight * peds

    total = sum(demand.values()) + 1e-9

    # Base split by demand share
    raw_green = {}
    for k in keys:
        share = demand[k] / total
        raw_green[k] = share * cycle_s

    # Enforce min/max constraints
    max_green = cycle_s - min_green_s  # for 2-phase, one approach min implies other max
    # If more than 2 approaches, we still clamp each, then normalize later.
    clamped = {k: clamp(raw_green[k], min_green_s, max_green) for k in keys}

    # Normalize so sum(greens) = cycle_s (important)
    s = sum(clamped.values())
    if s <= 1e-9:
        # fallback equal split
        equal = cycle_s / len(keys)
        clamped = {k: equal for k in keys}
    else:
        clamped = {k: (clamped[k] / s) * cycle_s for k in keys}

    # Optional smoothing against previous recommendation
    if prev_reco and prev_reco.get("greens"):
        prev_g = prev_reco["greens"]
        smoothed = {}
        for k in keys:
            prev_val = float(prev_g.get(k, clamped[k]))
            smoothed[k] = (1.0 - smoothing_alpha) * prev_val + smoothing_alpha * clamped[k]
        # re-normalize
        sm_sum = sum(smoothed.values())
        smoothed = {k: (smoothed[k] / sm_sum) * cycle_s for k in keys}
        greens = smoothed
    else:
        greens = clamped

    # Round for readability
    greens_rounded = {k: int(round(v)) for k, v in greens.items()}

    # Ensure sum exactly cycle_s after rounding
    diff = cycle_s - sum(greens_rounded.values())
    if diff != 0:
        # Adjust the max-demand approach to fix rounding diff
        max_key = max(demand, key=demand.get)
        greens_rounded[max_key] += diff

    # Create judge-friendly reason
    sorted_keys = sorted(demand.items(), key=lambda x: x[1], reverse=True)
    top = sorted_keys[0][0]
    reason = f"More green given to '{top}' due to higher demand (vehicles + pedestrian factor)."

    return {
        "ts": time.time(),
        "site_id": metrics_payload.get("site_id"),
        "cycle_s": cycle_s,
        "min_green_s": min_green_s,
        "greens": greens_rounded,   # e.g., {"north": 55, "south": 35}
        "demand": {k: round(v, 2) for k, v in demand.items()},
        "reason": reason,
        "updated_every_s": update_every_s,
    }
