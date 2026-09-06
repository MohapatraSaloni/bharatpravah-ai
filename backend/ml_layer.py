from __future__ import annotations
from typing import Dict, Any, List, Optional
import time
import os
import joblib
import numpy as np

from backend.ml_features import build_feature_vector


class CongestionForecaster:
    def __init__(self, model_path: str = "models/forecaster.pkl"):
        self.model_path = model_path
        self.model = None
        self.loaded = False
        self.load_error: Optional[str] = None
        self._try_load()

    def _try_load(self):
        if not os.path.exists(self.model_path):
            self.load_error = f"Model not found at {self.model_path}"
            self.loaded = False
            return
        try:
            self.model = joblib.load(self.model_path)
            self.loaded = True
            self.load_error = None
        except Exception as e:
            self.loaded = False
            self.load_error = str(e)

    def predict(self, history_payloads: List[Dict[str, Any]], approaches=("north", "south")) -> Dict[str, Any]:
        if not self.loaded or self.model is None:
            return {
                "status": "model_not_loaded",
                "message": self.load_error or "Model not loaded.",
                "ts": time.time(),
            }

        try:
            x = build_feature_vector(history_payloads, approaches=approaches)  # shape (F,)
        except Exception as e:
            return {
                "status": "insufficient_history",
                "message": str(e),
                "ts": time.time(),
            }

        # sklearn expects 2D
        y_pred = self.model.predict(x.reshape(1, -1))  # shape (1, 2)
        y_pred = np.clip(y_pred[0], 0.0, 1.0)

        return {
            "status": "ok",
            "ts": time.time(),
            "horizon_s": 30,
            "forecast": {
                approaches[0]: float(y_pred[0]),
                approaches[1]: float(y_pred[1]),
            },
            "model": type(self.model).__name__,
        }
