from ultralytics import YOLO

# COCO class IDs we care about:
# person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
KEEP_IDS = {0, 1, 2, 3, 5, 7}

class YoloDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame):
        # Run inference
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]

        detections = []
        if results.boxes is None:
            return detections

        for b in results.boxes:
            cls_id = int(b.cls[0].item())
            if cls_id not in KEEP_IDS:
                continue
            x1, y1, x2, y2 = [float(v.item()) for v in b.xyxy[0]]
            conf = float(b.conf[0].item())
            detections.append({
                "class_id": cls_id,
                "class_name": self.model.names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })

        return detections
