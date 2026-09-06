import cv2
from pathlib import Path

class VideoCamera:
    def __init__(self, video_path: str, resize_w: int = 960, resize_h: int = 540):
        self.video_path = str(Path(video_path))
        self.resize_w = resize_w
        self.resize_h = resize_h

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

    def get_raw_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            # Loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None

        frame = cv2.resize(frame, (self.resize_w, self.resize_h))
        return frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


