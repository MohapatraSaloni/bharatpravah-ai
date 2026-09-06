from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# We will reconstruct payload-like dicts from CSV rows
from backend.ml_features import build_training_samples


LOG_DIR = Path("data") / "logs"
MODEL_OUT = Path("models") / "forecaster.pkl"

HORIZON_S = 30
LOOKBACK_S = 60

def load_payloads_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    # expected columns:
    # ts, site_id,
    # north_congestion_score, south_congestion_score,
    # north_vehicles, south_vehicles
    payloads = []
    for _, r in df.iterrows():
        payloads.append({
            "ts": float(r["ts"]),
            "site_id": r.get("site_id", ""),
            "metrics": {
                "north": {
                    "congestion_score": float(r["north_congestion_score"]),
                    "vehicles": float(r["north_vehicles"]),
                    "pedestrians": float(r.get("north_pedestrians", 0.0)),
                },
                "south": {
                    "congestion_score": float(r["south_congestion_score"]),
                    "vehicles": float(r["south_vehicles"]),
                    "pedestrians": float(r.get("south_pedestrians", 0.0)),
                }
            }
        })
    return payloads

def main():
    if not LOG_DIR.exists():
        raise RuntimeError(f"Log directory not found: {LOG_DIR}")

    csv_files = list(LOG_DIR.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV logs found in {LOG_DIR}. Run the server and collect logs first.")

    all_payloads = []
    for f in csv_files:
        all_payloads.extend(load_payloads_from_csv(f))

    # sort by timestamp (important if you combined multiple files)
    all_payloads.sort(key=lambda x: x["ts"])

    X, y = build_training_samples(
        all_payloads,
        horizon_s=HORIZON_S,
        lookback_s=LOOKBACK_S,
        approaches=("north", "south"),
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Multi-output regressor works directly with RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=12,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae_n = mean_absolute_error(y_test[:, 0], pred[:, 0])
    mae_s = mean_absolute_error(y_test[:, 1], pred[:, 1])

    print("Training complete.")
    print(f"MAE north: {mae_n:.4f}")
    print(f"MAE south: {mae_s:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Saved model to: {MODEL_OUT}")

if __name__ == "__main__":
    main()
