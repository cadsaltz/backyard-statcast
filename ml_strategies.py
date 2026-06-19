"""ML-based ball detection (MediaPipe and YOLO COCO 'sports ball' class)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detection_strategies import (
    BallDetection,
    DetectionStrategy,
    StrategyConfig,
    StrategyResult,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
MEDIAPIPE_MODEL = MODELS_DIR / "efficientdet_lite0.tflite"
YOLO_MODEL = MODELS_DIR / "yolov8n.onnx"

# COCO class index for "sports ball" in YOLOv8 ONNX output.
YOLO_SPORTS_BALL_CLASS = 32


def _bbox_to_detection(
    x: int, y: int, w: int, h: int, score: float
) -> BallDetection:
    cx = x + w // 2
    cy = y + h // 2
    radius = max(max(w, h) // 2, 1)
    return BallDetection(
        center_x=cx,
        center_y=cy,
        radius=radius,
        bbox=(x, y, w, h),
        circularity=1.0,
    )


def _apply_roi(frame: np.ndarray, roi_top_frac: float) -> tuple[np.ndarray, int]:
    """Crop away the top of the frame (pitcher area). Returns crop and y-offset."""
    if roi_top_frac <= 0:
        return frame, 0
    top = int(frame.shape[0] * roi_top_frac)
    top = min(max(top, 0), frame.shape[0] - 1)
    return frame[top:, :], top


def _offset_detection(det: BallDetection, y_offset: int) -> BallDetection:
    x, y, w, h = det.bbox
    return BallDetection(
        center_x=det.center_x,
        center_y=det.center_y + y_offset,
        radius=det.radius,
        bbox=(x, y + y_offset, w, h),
        circularity=det.circularity,
    )


def _score_ball_candidate(
    det: BallDetection,
    confidence: float,
    frame_area: int,
) -> float:
    """Prefer small, high-confidence detections (ball is tiny vs people/bats)."""
    x, y, w, h = det.bbox
    box_area = w * h
    if box_area <= 0 or box_area > frame_area * 0.05:
        return 0.0
    smallness = frame_area / max(box_area, 1)
    return confidence * np.sqrt(smallness)


class MediapipeStrategy(DetectionStrategy):
    """MediaPipe EfficientDet-Lite0, filtered to COCO 'sports ball'."""

    name = "mediapipe"

    def _reset_state(self) -> None:
        if not MEDIAPIPE_MODEL.exists():
            raise FileNotFoundError(
                f"Missing {MEDIAPIPE_MODEL}. Run: python download_models.py"
            )

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.ObjectDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL)),
            score_threshold=self.config.ml_confidence,
            category_allowlist=["sports ball"],
            max_results=10,
        )
        self._detector = vision.ObjectDetector.create_from_options(options)
        self._mp = mp

    def _detect(self, frame: np.ndarray) -> StrategyResult:
        crop, y_off = _apply_roi(frame, self.config.roi_top_frac)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        result = self._detector.detect(mp_image)

        debug = crop.copy()
        candidates: list[tuple[float, BallDetection, float]] = []

        for det in result.detections:
            bbox = det.bounding_box
            score = det.categories[0].score if det.categories else 0.0
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 128, 0), 2)
            cv2.putText(
                debug,
                f"ball {score:.2f}",
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 128, 0),
                1,
                cv2.LINE_AA,
            )

            if w * h > self.config.max_area:
                continue
            ball = _bbox_to_detection(x, y, w, h, score)
            ball_score = _score_ball_candidate(ball, score, crop.shape[0] * crop.shape[1])
            if ball_score > 0:
                candidates.append((ball_score, ball, score))

        detection = self._pick_ml_candidate(candidates, y_off)
        info = f"{len(result.detections)} det(s), conf>={self.config.ml_confidence:.2f}"
        if y_off:
            info += f", roi_top={self.config.roi_top_frac:.0%}"

        return StrategyResult(
            detection=detection,
            debug_frame=debug,
            debug_info=info,
        )

    def _pick_ml_candidate(
        self,
        candidates: list[tuple[float, BallDetection, float]],
        y_offset: int,
    ) -> Optional[BallDetection]:
        if not candidates:
            return None

        if self._last_detection is not None:
            lx, ly = self._last_detection.center_x, self._last_detection.center_y
            in_range = [
                (s, d, c)
                for s, d, c in candidates
                if np.hypot(d.center_x - lx, d.center_y - (ly - y_offset))
                <= self.config.max_jump_px
            ]
            if in_range:
                candidates = in_range

        _, best, _ = max(candidates, key=lambda item: item[0])
        return _offset_detection(best, y_offset)


class YoloStrategy(DetectionStrategy):
    """YOLOv8n ONNX via OpenCV DNN, COCO sports ball class (no PyTorch needed)."""

    name = "yolo"

    def _reset_state(self) -> None:
        if not YOLO_MODEL.exists():
            raise FileNotFoundError(
                f"Missing {YOLO_MODEL}. Run: python download_models.py"
            )
        self._net = cv2.dnn.readNetFromONNX(str(YOLO_MODEL))

    def _detect(self, frame: np.ndarray) -> StrategyResult:
        crop, y_off = _apply_roi(frame, self.config.roi_top_frac)
        h, w = crop.shape[:2]
        length = max(h, w)
        padded = np.zeros((length, length, 3), dtype=np.uint8)
        padded[0:h, 0:w] = crop

        blob = cv2.dnn.blobFromImage(
            padded, scalefactor=1 / 255.0, size=(640, 640), swapRB=True, crop=False
        )
        self._net.setInput(blob)
        outputs = self._net.forward()

        # YOLOv8 ONNX: [1, 84, 8400] -> transpose to [8400, 84]
        preds = np.squeeze(outputs)
        if preds.ndim == 2 and preds.shape[0] == 84:
            preds = preds.T

        debug = crop.copy()
        candidates: list[tuple[float, BallDetection, float]] = []
        raw_count = 0

        for row in preds:
            classes = row[4:]
            class_id = int(np.argmax(classes))
            confidence = float(classes[class_id])
            if class_id != YOLO_SPORTS_BALL_CLASS:
                continue
            if confidence < self.config.ml_confidence:
                continue

            raw_count += 1
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            scale = length / 640.0
            cx *= scale
            cy *= scale
            bw *= scale
            bh *= scale

            x = int(cx - bw / 2)
            y = int(cy - bh / 2)
            bw_i, bh_i = int(bw), int(bh)

            cv2.rectangle(debug, (x, y), (x + bw_i, y + bh_i), (0, 165, 255), 2)
            cv2.putText(
                debug,
                f"ball {confidence:.2f}",
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

            if bw_i * bh_i > self.config.max_area:
                continue
            ball = _bbox_to_detection(x, y, bw_i, bh_i, confidence)
            ball_score = _score_ball_candidate(ball, confidence, h * w)
            if ball_score > 0:
                candidates.append((ball_score, ball, confidence))

        detection = self._pick_ml_candidate(candidates, y_off)
        info = f"{raw_count} sports-ball(s), conf>={self.config.ml_confidence:.2f}"
        if y_off:
            info += f", roi_top={self.config.roi_top_frac:.0%}"

        return StrategyResult(
            detection=detection,
            debug_frame=debug,
            debug_info=info,
        )

    def _pick_ml_candidate(
        self,
        candidates: list[tuple[float, BallDetection, float]],
        y_offset: int,
    ) -> Optional[BallDetection]:
        if not candidates:
            return None

        if self._last_detection is not None:
            lx, ly = self._last_detection.center_x, self._last_detection.center_y
            in_range = [
                (s, d, c)
                for s, d, c in candidates
                if np.hypot(d.center_x - lx, d.center_y - (ly - y_offset))
                <= self.config.max_jump_px
            ]
            if in_range:
                candidates = in_range

        _, best, _ = max(candidates, key=lambda item: item[0])
        return _offset_detection(best, y_offset)
