#!/usr/bin/env python3
"""Download ML model files used by ball detection strategies."""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

DOWNLOADS = {
    "efficientdet_lite0.tflite": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
    ),
    "yolov8n.onnx": (
        "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx"
    ),
}


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, url in DOWNLOADS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"already have {dest}")
            continue
        print(f"downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest}")

    print("\nModels ready.")
    print("  mediapipe -> models/efficientdet_lite0.tflite")
    print("  yolo      -> models/yolov8n.onnx")


if __name__ == "__main__":
    main()
