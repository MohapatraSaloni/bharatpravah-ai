import threading
import time
from collections import deque
import os
import csv

from backend.video_stream import VideoCamera
from backend.detector import YoloDetector
from backend.metrics import RollingAverager, build_metrics_payload
from backend.advisor import build_recommendation


class VisionRuntime:
    def __init__(self, site_id: str, video_path: str, rois: dict, resize_w=960, resize_h=540):
        self.site_id = site_id
        self.rois = rois

        self.camera = VideoCamera(video_path=video_path, resize_w=resize_w, resize_h=resize_h)
        self.detector = YoloDetector(model_name="yolov8n.pt", conf=0.35)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_detections = []

        # Phase 3 output
        self.latest_payload = None  # metrics payload

        # Phase 4 output
        self.latest_recommendation = None  # timing recommendation

        # Phase 3 smoothing
        self.averager = RollingAverager(window_size=10)

        # Phase 5: keep last ~60 seconds of payloads for ML inference
        self.metrics_history = deque(maxlen=60)

        # Phase 5: log to CSV (for offline training)
        self.log_enabled = True
        self.log_path = os.path.join("data", "logs", f"{self.site_id}.csv")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Write CSV header once
        if self.log_enabled and not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "ts", "site_id",
                    "north_congestion_score", "south_congestion_score",
                    "north_vehicles", "south_vehicles",
                    "north_pedestrians", "south_pedestrians",
                ])

        self.running = False

    def start(
        self,
        detect_every_n_frames: int = 2,
        metrics_every_sec: float = 1.0,
        recommendation_every_sec: float = 10.0,  # Phase 4 update rate (10s)
    ):
        if self.running:
            return
        self.running = True

        def loop():
            frame_count = 0
            last_metrics_time = 0.0
            last_reco_time = 0.0
            last_dets = []

            while self.running:
                frame = self.camera.get_raw_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1

                # Run detection every N frames
                if frame_count % detect_every_n_frames == 0:
                    last_dets = self.detector.detect(frame)

                now = time.time()

                # Store latest frame & detections (for streaming)
                with self.lock:
                    self.latest_frame = frame
                    self.latest_detections = last_dets

                # -------------------------
                # Phase 3: Metrics payload
                # -------------------------
                if (now - last_metrics_time) >= metrics_every_sec:
                    payload = build_metrics_payload(
                        last_dets,
                        averager=self.averager,
                        rois=self.rois,
                    )
                    payload["site_id"] = self.site_id

                    with self.lock:
                        self.latest_payload = payload
                        self.metrics_history.append(payload)  # Phase 5 history buffer

                    # Phase 5: Log for training
                    if self.log_enabled:
                        try:
                            m = payload["metrics"]
                            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                                w = csv.writer(f)
                                w.writerow([
                                    payload["ts"], self.site_id,
                                    m["north"]["congestion_score"], m["south"]["congestion_score"],
                                    m["north"]["vehicles"], m["south"]["vehicles"],
                                    m["north"]["pedestrians"], m["south"]["pedestrians"],
                                ])
                        except Exception:
                            # Logging should never break runtime
                            pass

                    last_metrics_time = now

                # -----------------------------------
                # Phase 4: Recommendation (every 10s)
                # -----------------------------------
                if (now - last_reco_time) >= recommendation_every_sec:
                    with self.lock:
                        payload = self.latest_payload
                        prev_reco = self.latest_recommendation

                    if payload is not None:
                        reco = build_recommendation(
                            metrics_payload=payload,
                            prev_reco=prev_reco,
                            cycle_s=90,
                            min_green_s=20,
                            update_every_s=int(recommendation_every_sec),
                            ped_weight=0.5,
                            smoothing_alpha=0.3,
                        )
                        with self.lock:
                            self.latest_recommendation = reco

                    last_reco_time = now

                time.sleep(0.01)

        threading.Thread(target=loop, daemon=True).start()

    # -------------------------
    # Getters for API endpoints
    # -------------------------
    def get_latest(self):
        with self.lock:
            return self.latest_frame, self.latest_detections

    def get_latest_metrics_payload(self):
        with self.lock:
            return self.latest_payload

    def get_latest_recommendation(self):
        with self.lock:
            return self.latest_recommendation

    def get_metrics_history(self):
        with self.lock:
            return list(self.metrics_history)
