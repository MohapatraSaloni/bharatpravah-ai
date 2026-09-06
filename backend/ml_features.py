from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np


def _extract_series(history: List[Dict[str, Any]], approach: str, key: str) -> np.ndarray:
    vals = []
    for h in history:
        m = h.get("metrics", {}).get(approach, {})
        vals.append(float(m.get(key, 0.0)))
    return np.array(vals, dtype=float)


def build_feature_vector(
    history: List[Dict[str, Any]],
    approaches: Tuple[str, str] = ("north", "south"),
) -> np.ndarray:
    """
    Input: list of payloads (oldest->newest), each having payload["metrics"][approach][...]
    Output: 1D feature vector suitable for sklearn model.
    """
    if len(history) < 12:
        # need at least ~12 seconds to compute slopes reliably
        raise ValueError("Not enough history to build features (need >= 12 points).")

    a1, a2 = approaches

    # series
    s1 = _extract_series(history, a1, "congestion_score")
    s2 = _extract_series(history, a2, "congestion_score")
    v1 = _extract_series(history, a1, "vehicles")
    v2 = _extract_series(history, a2, "vehicles")

    # helper windows
    def mean_last(x, n): return float(np.mean(x[-n:]))
    def max_last(x, n): return float(np.max(x[-n:]))
    def slope_last(x, n):
        # simple slope: (now - n steps ago)/n
        return float((x[-1] - x[-n]) / max(1, n))

    feats = [
        # current congestion
        float(s1[-1]), float(s2[-1]),
        # rolling means
        mean_last(s1, 10), mean_last(s2, 10),
        mean_last(v1, 10), mean_last(v2, 10),
        # rolling max vehicles
        max_last(v1, 10), max_last(v2, 10),
        # trend/slope
        slope_last(s1, 10), slope_last(s2, 10),
        slope_last(v1, 10), slope_last(v2, 10),
    ]

    return np.array(feats, dtype=float)


def build_training_samples(
    series_payloads: List[Dict[str, Any]],
    horizon_s: int = 30,
    lookback_s: int = 60,
    approaches: Tuple[str, str] = ("north", "south"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a whole time series payload list into supervised dataset:
    X: features from last lookback_s seconds
    y: average congestion_score over next horizon_s seconds for each approach (2 targets)
    """
    # Assume payloads are sampled about 1/sec (your metrics_every_sec=1.0)
    L = lookback_s
    H = horizon_s
    n = len(series_payloads)

    X_list = []
    y_list = []

    for t in range(L, n - H):
        hist = series_payloads[t - L:t]  # length L
        future = series_payloads[t:t + H]  # length H

        x = build_feature_vector(hist, approaches=approaches)

        # targets = future avg congestion score for each approach
        a1, a2 = approaches
        y1 = float(np.mean(_extract_series(future, a1, "congestion_score")))
        y2 = float(np.mean(_extract_series(future, a2, "congestion_score")))

        X_list.append(x)
        y_list.append([y1, y2])

    if not X_list:
        raise ValueError("Not enough data to build training samples.")

    return np.vstack(X_list), np.array(y_list, dtype=float)
