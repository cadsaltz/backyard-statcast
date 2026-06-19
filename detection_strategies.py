"""Independent ball-detection strategies for side-by-side comparison."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class BallDetection:
    """Single-frame ball detection result in pixel coordinates."""

    center_x: int
    center_y: int
    radius: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    circularity: float


@dataclass
class StrategyResult:
    detection: Optional[BallDetection]
    debug_mask: Optional[np.ndarray] = None
    debug_frame: Optional[np.ndarray] = None
    debug_info: str = ""


@dataclass
class StrategyConfig:
    min_area: int = 20
    max_area: int = 2500
    min_circularity: float = 0.55
    brightness_threshold: int = 175
    diff_threshold: int = 25
    warmup_frames: int = 45
    max_jump_px: int = 120
    hough_param1: int = 100
    hough_param2: int = 18
    hough_min_radius: int = 4
    hough_max_radius: int = 45
    ml_confidence: float = 0.15
    roi_top_frac: float = 0.30


STRATEGY_NAMES = (
    "mog2",
    "brightness",
    "combined",
    "frame_diff",
    "hough",
    "adaptive",
    "mediapipe",
    "yolo",
)


def create_strategy(name: str, config: StrategyConfig) -> DetectionStrategy:
    from ml_strategies import MediapipeStrategy, YoloStrategy

    strategies = {
        "mog2": Mog2Strategy,
        "brightness": BrightnessStrategy,
        "combined": CombinedStrategy,
        "frame_diff": FrameDiffStrategy,
        "hough": HoughStrategy,
        "adaptive": AdaptiveThresholdStrategy,
        "mediapipe": MediapipeStrategy,
        "yolo": YoloStrategy,
    }
    if name not in strategies:
        valid = ", ".join(STRATEGY_NAMES)
        raise ValueError(f"Unknown strategy {name!r}. Choose one of: {valid}")
    return strategies[name](config)


class DetectionStrategy(ABC):
    name: str
    needs_warmup: bool = False

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self._frame_index = 0
        self._last_detection: Optional[BallDetection] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._reset_state()

    def reset(self) -> None:
        self._frame_index = 0
        self._last_detection = None
        self._reset_state()

    @abstractmethod
    def _reset_state(self) -> None:
        pass

    @property
    def is_warming_up(self) -> bool:
        if not self.needs_warmup:
            return False
        return self._frame_index < self.config.warmup_frames

    @property
    def warmup_progress(self) -> tuple[int, int]:
        current = min(self._frame_index, self.config.warmup_frames)
        return current, self.config.warmup_frames

    def detect(self, frame: np.ndarray) -> StrategyResult:
        self._frame_index += 1
        if self.is_warming_up:
            self._update_during_warmup(frame)
            return StrategyResult(None, self._blank_mask(frame))

        result = self._detect(frame)
        if result.detection is None:
            self._last_detection = None
        else:
            self._last_detection = result.detection
        return result

    def _update_during_warmup(self, frame: np.ndarray) -> None:
        """Let background-learning strategies absorb static scene during warmup."""

    def _blank_mask(self, frame: np.ndarray) -> np.ndarray:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    @abstractmethod
    def _detect(self, frame: np.ndarray) -> StrategyResult:
        pass

    def _collect_candidates(
        self, mask: np.ndarray
    ) -> list[tuple[float, BallDetection]]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: list[tuple[float, BallDetection]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area or area > self.config.max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.config.min_circularity:
                continue

            (cx_f, cy_f), radius = cv2.minEnclosingCircle(contour)
            cx, cy, radius_i = int(cx_f), int(cy_f), max(int(radius), 1)
            x, y, w, h = cv2.boundingRect(contour)

            aspect = w / max(h, 1)
            if aspect < 0.55 or aspect > 1.8:
                continue

            detection = BallDetection(
                center_x=cx,
                center_y=cy,
                radius=radius_i,
                bbox=(x, y, w, h),
                circularity=float(circularity),
            )
            score = circularity * np.sqrt(area)
            candidates.append((score, detection))
        return candidates

    def _pick_best(
        self, candidates: list[tuple[float, BallDetection]]
    ) -> Optional[BallDetection]:
        if not candidates:
            return None

        if self._last_detection is not None:
            lx, ly = self._last_detection.center_x, self._last_detection.center_y
            in_range = [
                (score, det)
                for score, det in candidates
                if np.hypot(det.center_x - lx, det.center_y - ly)
                <= self.config.max_jump_px
            ]
            if in_range:
                candidates = in_range

        _, best = max(candidates, key=lambda item: item[0])
        return best

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self._kernel)

    def _value_channel(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2]

    def _result_from_mask(self, mask: np.ndarray) -> StrategyResult:
        candidates = self._collect_candidates(mask)
        detection = self._pick_best(candidates)
        return StrategyResult(detection, mask)


class Mog2Strategy(DetectionStrategy):
    name = "mog2"
    needs_warmup = True

    def _reset_state(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=180, varThreshold=25, detectShadows=False
        )

    def _update_during_warmup(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._bg.apply(gray)

    def _detect(self, frame: np.ndarray) -> StrategyResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._bg.apply(gray)
        mask = self._clean_mask(mask)
        return self._result_from_mask(mask)


class BrightnessStrategy(DetectionStrategy):
    name = "brightness"

    def _reset_state(self) -> None:
        pass

    def _detect(self, frame: np.ndarray) -> StrategyResult:
        value = self._value_channel(frame)
        _, mask = cv2.threshold(
            value, self.config.brightness_threshold, 255, cv2.THRESH_BINARY
        )
        mask = self._clean_mask(mask)
        return self._result_from_mask(mask)


class CombinedStrategy(DetectionStrategy):
    name = "combined"
    needs_warmup = True

    def _reset_state(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=180, varThreshold=25, detectShadows=False
        )

    def _update_during_warmup(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._bg.apply(gray)

    def _detect(self, frame: np.ndarray) -> StrategyResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        value = self._value_channel(frame)

        motion = self._bg.apply(gray)
        _, bright = cv2.threshold(
            value, self.config.brightness_threshold, 255, cv2.THRESH_BINARY
        )
        mask = cv2.bitwise_and(motion, bright)
        mask = self._clean_mask(mask)

        candidates = self._collect_candidates(mask)
        if not candidates:
            _, very_bright = cv2.threshold(
                value,
                min(self.config.brightness_threshold + 35, 245),
                255,
                cv2.THRESH_BINARY,
            )
            mask = self._clean_mask(very_bright)
            return self._result_from_mask(mask)

        detection = self._pick_best(candidates)
        return StrategyResult(detection, mask)


class FrameDiffStrategy(DetectionStrategy):
    name = "frame_diff"

    def _reset_state(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None

    def _detect(self, frame: np.ndarray) -> StrategyResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return StrategyResult(None, self._blank_mask(frame))

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray

        _, mask = cv2.threshold(
            diff, self.config.diff_threshold, 255, cv2.THRESH_BINARY
        )
        mask = self._clean_mask(mask)
        return self._result_from_mask(mask)


class HoughStrategy(DetectionStrategy):
    name = "hough"

    def _reset_state(self) -> None:
        pass

    def _detect(self, frame: np.ndarray) -> StrategyResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=self.config.hough_param1,
            param2=self.config.hough_param2,
            minRadius=self.config.hough_min_radius,
            maxRadius=self.config.hough_max_radius,
        )

        mask = self._blank_mask(frame)
        if circles is None:
            return StrategyResult(None, mask)

        circles = np.uint16(np.around(circles[0]))
        candidates: list[tuple[float, BallDetection]] = []
        value = self._value_channel(frame)

        for x, y, r in circles:
            x_i, y_i, r_i = int(x), int(y), int(r)
            if r_i <= 0:
                continue

            # Prefer circles that land on bright pixels (ball-like).
            y0, y1 = max(0, y_i - r_i), min(frame.shape[0], y_i + r_i)
            x0, x1 = max(0, x_i - r_i), min(frame.shape[1], x_i + r_i)
            patch = value[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            brightness = float(np.mean(patch))
            if brightness < self.config.brightness_threshold - 20:
                continue

            cv2.circle(mask, (x_i, y_i), r_i, 255, 2)
            side = max(r_i * 2, 2)
            bbox = (x_i - r_i, y_i - r_i, side, side)
            score = brightness * r_i
            candidates.append(
                (
                    score,
                    BallDetection(
                        center_x=x_i,
                        center_y=y_i,
                        radius=r_i,
                        bbox=bbox,
                        circularity=1.0,
                    ),
                )
            )

        detection = self._pick_best(candidates)
        return StrategyResult(detection, mask)


class AdaptiveThresholdStrategy(DetectionStrategy):
    name = "adaptive"

    def _reset_state(self) -> None:
        pass

    def _detect(self, frame: np.ndarray) -> StrategyResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -8,
        )
        mask = self._clean_mask(mask)
        return self._result_from_mask(mask)
