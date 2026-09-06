import cv2
from pathlib import Path

VIDEO_PATH = Path(r"C:\Users\DELL\Documents\Git Projects\bharatpravah\data\kolkata.mp4")


RESIZE_W, RESIZE_H = 960, 540  # MUST match your get_raw_frame resize

drawing = False
x_start, y_start = -1, -1

def mouse_cb(event, x, y, flags, param):
    global drawing, x_start, y_start
    img = param["img"]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = img.copy()
        cv2.rectangle(temp, (x_start, y_start), (x, y), (0, 255, 0), 2)
        cv2.imshow("ROI Picker", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = x_start, y_start
        x2, y2 = x, y

        # normalize so x1<x2 and y1<y2
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        print(f"ROI rectangle: ({x1}, {y1}, {x2}, {y2})")

        # Draw final rectangle on image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("ROI Picker", img)


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read from video: {VIDEO_PATH}")

    frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

    cv2.imshow("ROI Picker", frame)
    param = {"img": frame}

    cv2.setMouseCallback("ROI Picker", mouse_cb, param)

    print("Instructions:")
    print(" - Drag mouse to draw ROI rectangles.")
    print(" - Each time you release, coords print in terminal.")
    print(" - Press ESC to exit.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
