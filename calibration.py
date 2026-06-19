"""Field calibration geometry and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

Point = tuple[float, float]
Polygon = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class FieldCalibration:
    """Spatial regions drawn on the calibration frame."""

    frame_width: int
    frame_height: int
    roi: Polygon
    strike_zone: Polygon
    release_center: Point
    release_radius: float  # fraction of min(frame_width, frame_height)
    ignore_mask: np.ndarray  # uint8, same HxW as calibration frame, 255 = ignore

    def __post_init__(self) -> None:
        object.__setattr__(self, "roi", tuple(self.roi))
        object.__setattr__(self, "strike_zone", tuple(self.strike_zone))
        object.__setattr__(self, "release_center", tuple(self.release_center))

    def validate(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if len(self.roi) != 4 or len(self.strike_zone) != 4:
            raise ValueError("roi and strike_zone must have exactly 4 points")
        if self.release_radius <= 0:
            raise ValueError("release_radius must be positive")
        expected = (self.frame_height, self.frame_width)
        if self.ignore_mask.shape != expected:
            raise ValueError(
                f"ignore_mask shape {self.ignore_mask.shape} != expected {expected}"
            )


def _points_to_json(points: Sequence[Point]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in points]


def _points_from_json(raw: Sequence[Sequence[float]]) -> tuple[Point, ...]:
    return tuple((float(p[0]), float(p[1])) for p in raw)


def save_calibration(cal: FieldCalibration, json_path: Path) -> None:
    cal.validate()
    json_path = Path(json_path)
    payload = {
        "frame_width": cal.frame_width,
        "frame_height": cal.frame_height,
        "roi": _points_to_json(cal.roi),
        "strike_zone": _points_to_json(cal.strike_zone),
        "release_center": list(cal.release_center),
        "release_radius": cal.release_radius,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    ignore_path = json_path.with_name(json_path.stem + "_ignore.png")
    cv2.imwrite(str(ignore_path), cal.ignore_mask)


def load_calibration(json_path: Path) -> FieldCalibration:
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    w, h = int(data["frame_width"]), int(data["frame_height"])
    ignore_path = json_path.with_name(json_path.stem + "_ignore.png")
    if ignore_path.is_file():
        ignore_mask = cv2.imread(str(ignore_path), cv2.IMREAD_GRAYSCALE)
        if ignore_mask is None:
            raise ValueError(f"Could not read ignore mask: {ignore_path}")
    else:
        ignore_mask = np.zeros((h, w), dtype=np.uint8)

    cal = FieldCalibration(
        frame_width=w,
        frame_height=h,
        roi=_points_from_json(data["roi"]),  # type: ignore[arg-type]
        strike_zone=_points_from_json(data["strike_zone"]),  # type: ignore[arg-type]
        release_center=(float(data["release_center"][0]), float(data["release_center"][1])),
        release_radius=float(data["release_radius"]),
        ignore_mask=ignore_mask,
    )
    cal.validate()
    return cal
