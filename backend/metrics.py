from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Any
import time


# -----------------------------
# 1) ROI CONFIG (EDIT THIS!)
# -----------------------------
# Format: (x1, y1, x2, y2) in pixels on your resized frame (e.g., 960x540)
ROIS = {
    "north": (8, 160, 836, 333),
    "south": (7, 338, 1156, 576),
}
# Only count these detection classes (matches your detector.py naming)
DEFAULT_KEEP_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}

# Congestion score is normalized 0..1 based on density threshold.
# You will tune this after seeing results. Start here.
DENSITY_THRESHOLD = 0.00012  # vehicles per pixel^2


def roi_area(roi: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = roi
    return max(1, (x2 - x1) * (y2 - y1))


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_roi(cx: float, cy: float, roi: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


def assign_detections_to_rois(
    detections: List[Dict[str, Any]],
    rois: Dict[str, Tuple[int, int, int, int]] = ROIS,
    keep_classes: set[str] = DEFAULT_KEEP_CLASSES,
) -> Dict[str, Dict[str, int]]:
    """
    Returns counts per ROI: total vehicles (and ped count separately).
    We keep it explainable:
      - vehicles_count: car/bus/truck/motorcycle/bicycle
      - ped_count: person
    """
    counts = {name: {"vehicles": 0, "pedestrians": 0} for name in rois.keys()}

    for d in detections:
        cls = d.get("class_name", "")
        if cls not in keep_classes:
            continue

        cx, cy = bbox_center(d["bbox"])

        for roi_name, roi in rois.items():
            if point_in_roi(cx, cy, roi):
                if cls == "person":
                    counts[roi_name]["pedestrians"] += 1
                else:
                    counts[roi_name]["vehicles"] += 1
                break  # one detection belongs to one ROI

    return counts


def compute_metrics_from_counts(
    counts: Dict[str, Dict[str, int]],
    rois: Dict[str, Tuple[int, int, int, int]] = ROIS,
    density_threshold: float = DENSITY_THRESHOLD,
) -> Dict[str, Dict[str, float]]:
    """
    Convert counts into density + normalized congestion score.
    density = vehicles / roi_area
    score  = min(1, density / threshold)
    """
    metrics: Dict[str, Dict[str, float]] = {}

    for roi_name, c in counts.items():
        area = roi_area(rois[roi_name])
        veh = float(c["vehicles"])
        ped = float(c["pedestrians"])

        density = veh / float(area)
        score = min(1.0, density / max(1e-9, density_threshold))

        metrics[roi_name] = {
            "vehicles": veh,
            "pedestrians": ped,
            "density": density,
            "congestion_score": score,
            "roi_area": float(area),
        }

    return metrics


@dataclass
class RollingAverager:
    """
    Keeps a rolling window of recent counts to smooth the output.
    Judges prefer stable metrics, not flickering per frame.
    """
    window_size: int = 10

    def __post_init__(self):
        self.buffers: Dict[str, Dict[str, deque]] = {}

    def update(self, counts: Dict[str, Dict[str, int]]):
        for roi_name, c in counts.items():
            if roi_name not in self.buffers:
                self.buffers[roi_name] = {
                    "vehicles": deque(maxlen=self.window_size),
                    "pedestrians": deque(maxlen=self.window_size),
                }
            self.buffers[roi_name]["vehicles"].append(c["vehicles"])
            self.buffers[roi_name]["pedestrians"].append(c["pedestrians"])

    def averaged_counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for roi_name, buf in self.buffers.items():
            v = buf["vehicles"]
            p = buf["pedestrians"]
            out[roi_name] = {
                "vehicles": int(round(sum(v) / max(1, len(v)))),
                "pedestrians": int(round(sum(p) / max(1, len(p)))),
            }
        return out


def build_metrics_payload(detections, averager, rois, density_threshold=DENSITY_THRESHOLD):
    counts = assign_detections_to_rois(detections, rois=rois)
    averager.update(counts)
    smooth_counts = averager.averaged_counts()
    metrics = compute_metrics_from_counts(smooth_counts, rois=rois, density_threshold=density_threshold)
    return {
        "ts": time.time(),
        "counts": smooth_counts,
        "metrics": metrics,
        "density_threshold": density_threshold,
        "rois": {k: list(v) for k, v in rois.items()},
    }