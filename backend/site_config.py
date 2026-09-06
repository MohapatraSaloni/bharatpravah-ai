from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

SITES = {
    "blr": {
        "name": "Bengaluru",
        "video_path": str(BASE / "data" / "blr.mp4"),
        "rois": {
            "north": (8, 160, 836, 333),
            "south": (7, 338, 1156, 576),
        },
    },
    "delhi": {
        "name": "Delhi",
        "video_path": str(BASE / "data" / "delhi.mp4"),
        "rois": {
            "north": (11, 31, 950, 278),     # TODO: set using ROI picker
            "south": (11, 287, 956, 539),   # TODO: set using ROI picker
        },
    },
    "kolkata": {
        "name": "Kolkata",
        "video_path": str(BASE / "data" / "kolkata.mp4"),
        "rois": {
            "north": (113, 250, 681, 564),     # TODO
            "south": (683, 200, 1249, 341),   # TODO
        },
    },
}
