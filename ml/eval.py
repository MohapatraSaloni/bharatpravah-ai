from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Config ----------
LOG_DIR = Path("data") / "logs"
CYCLE_S = 90
BASELINE_GREEN_S = 45  # fixed equal split baseline
MIN_GREEN_S = 20
UPDATE_EVERY_S = 10    # mimic Phase 4 update interval
PED_WEIGHT = 0.5
SMOOTHING_ALPHA = 0.3  # mimic Phase 4 smoothing
EVAL_WINDOW_S = 300    # 5 minutes; set None to use full file
# ---------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_greens_from_demand(d_n: float, d_s: float,
                              prev_n: float | None,
                              cycle_s=CYCLE_S,
                              min_green_s=MIN_GREEN_S,
                              alpha=SMOOTHING_ALPHA) -> tuple[float, float]:
    """
    Phase-4-like (rule-based) greens from demand. Two-phase only: north/south.
    Includes clamp + smoothing + normalization.
    """
    total = d_n + d_s + 1e-9
    raw_n = (d_n / total) * cycle_s
    raw_s = cycle_s - raw_n

    # clamp each (then renormalize)
    max_green = cycle_s - min_green_s
    raw_n = clamp(raw_n, min_green_s, max_green)
    raw_s = clamp(raw_s, min_green_s, max_green)
    s = raw_n + raw_s
    raw_n = (raw_n / s) * cycle_s
    raw_s = cycle_s - raw_n

    # smoothing (only on north, south follows by cycle constraint)
    if prev_n is not None:
        sm_n = (1 - alpha) * prev_n + alpha * raw_n
        sm_n = clamp(sm_n, min_green_s, max_green)
        sm_s = cycle_s - sm_n
        return sm_n, sm_s

    return raw_n, raw_s


def phase8_eval_one(csv_path: Path, window_s: int | None = EVAL_WINDOW_S) -> dict:
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required = [
        "ts",
        "north_congestion_score", "south_congestion_score",
        "north_vehicles", "south_vehicles",
        "north_pedestrians", "south_pedestrians",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{csv_path} missing columns: {missing}")

    # Sort by time and optionally crop to a window
    df = df.sort_values("ts").reset_index(drop=True)
    if window_s is not None and len(df) > window_s:
        df = df.tail(window_s).reset_index(drop=True)

    # Convert to numeric safely
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)

    # Demand series
    df["demand_n"] = df["north_vehicles"] + PED_WEIGHT * df["north_pedestrians"]
    df["demand_s"] = df["south_vehicles"] + PED_WEIGHT * df["south_pedestrians"]

    # Create “recommended greens” that update every 10s (hold in between)
    greens_n = np.zeros(len(df), dtype=float)
    greens_s = np.zeros(len(df), dtype=float)

    prev_n = None
    last_update_idx = -10**9

    for i in range(len(df)):
        # Update every UPDATE_EVERY_S rows (metrics logged ~1/sec)
        if (i - last_update_idx) >= UPDATE_EVERY_S:
            d_n = float(df.at[i, "demand_n"])
            d_s = float(df.at[i, "demand_s"])
            g_n, g_s = compute_greens_from_demand(d_n, d_s, prev_n)
            prev_n = g_n
            last_update_idx = i
            current_gn, current_gs = g_n, g_s
        greens_n[i] = current_gn
        greens_s[i] = current_gs

    df["rec_green_n"] = greens_n
    df["rec_green_s"] = greens_s

    # ---- Metrics for evaluation ----
    # Baseline waiting proxy (equal green split)
    # waiting ≈ congestion_score * (cycle/green)
    df["base_wait_n"] = df["north_congestion_score"] * (CYCLE_S / BASELINE_GREEN_S)
    df["base_wait_s"] = df["south_congestion_score"] * (CYCLE_S / BASELINE_GREEN_S)

    # BharatPravah waiting proxy using recommended greens
    df["bp_wait_n"] = df["north_congestion_score"] * (CYCLE_S / df["rec_green_n"])
    df["bp_wait_s"] = df["south_congestion_score"] * (CYCLE_S / df["rec_green_s"])

    # Average across approaches
    base_wait = float(((df["base_wait_n"] + df["base_wait_s"]) / 2).mean())
    bp_wait = float(((df["bp_wait_n"] + df["bp_wait_s"]) / 2).mean())

    # Average congestion score (observation-only; same raw data)
    # We still report it as “observed level during window”
    avg_cong = float(((df["north_congestion_score"] + df["south_congestion_score"]) / 2).mean())

    # Improvement percentage
    wait_impr = (base_wait - bp_wait) / (base_wait + 1e-9) * 100.0

    # Fuel proxy: proportional to idle/waiting
    fuel_impr = wait_impr

    site_id = str(df["site_id"].iloc[-1]) if "site_id" in df.columns and len(df) else csv_path.stem

    return {
        "site_id": site_id,
        "rows_used_sec": int(len(df)),
        "avg_observed_congestion_score": avg_cong,
        "baseline_wait_proxy": base_wait,
        "bharatpravah_wait_proxy": bp_wait,
        "wait_improvement_pct": wait_impr,
        "fuel_proxy_improvement_pct": fuel_impr,
        "cycle_s": CYCLE_S,
        "baseline_green_s": BASELINE_GREEN_S,
        "update_every_s": UPDATE_EVERY_S,
        "min_green_s": MIN_GREEN_S,
        "ped_weight": PED_WEIGHT,
    }


def main():
    if not LOG_DIR.exists():
        raise RuntimeError(f"Log dir not found: {LOG_DIR}")

    csvs = sorted(LOG_DIR.glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV logs found in {LOG_DIR}. Run uvicorn and let it log first.")

    results = []
    for f in csvs:
        try:
            res = phase8_eval_one(f, window_s=EVAL_WINDOW_S)
            results.append(res)
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}")

    if not results:
        raise RuntimeError("No usable log files to evaluate.")

    out = pd.DataFrame(results)
    print("\n=== Phase 8 Impact Summary (per site) ===")
    print(out[[
        "site_id", "rows_used_sec",
        "baseline_wait_proxy", "bharatpravah_wait_proxy",
        "wait_improvement_pct",
        "avg_observed_congestion_score"
    ]].to_string(index=False))

    # Overall summary (simple mean)
    overall = {
        "sites": int(len(out)),
        "mean_wait_improvement_pct": float(out["wait_improvement_pct"].mean()),
        "mean_baseline_wait_proxy": float(out["baseline_wait_proxy"].mean()),
        "mean_bharatpravah_wait_proxy": float(out["bharatpravah_wait_proxy"].mean()),
        "mean_observed_congestion_score": float(out["avg_observed_congestion_score"].mean()),
    }

    print("\n=== Overall (average across evaluated sites) ===")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

    # Save report
    report_path = Path("data") / "phase8_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(report_path, index=False)
    print(f"\nSaved per-site report to: {report_path}")


if __name__ == "__main__":
    main()
