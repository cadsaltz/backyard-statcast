"""Spatial geometry helpers for field calibration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calibration import FieldCalibration, Point, Polygon

__all__ = [
    "Classification",
    "classify_point",
    "ignore_mask_for_process",
    "is_ignored",
    "point_in_circle",
    "point_in_polygon",
    "resize_ignore_mask",
    "scaled_polygons",
    "scale_point_to_pixels",
]


@dataclass(frozen=True)
class Classification:
    ignored: bool
    in_roi: bool
    in_strike_zone: bool
    in_release_zone: bool


def point_in_polygon(point: Point, polygon: Polygon | list[Point]) -> bool:
    """Return True if normalized point (x, y) is inside the polygon."""
    px = np.array([[int(point[0] * 10000), int(point[1] * 10000)]], dtype=np.int32)
    poly = np.array(
        [[int(x * 10000), int(y * 10000)] for x, y in polygon],
        dtype=np.int32,
    )
    return cv2.pointPolygonTest(poly, (float(px[0, 0]), float(px[0, 1])), False) >= 0


def scale_point_to_pixels(
    point: Point, *, frame_width: int, frame_height: int
) -> tuple[int, int]:
    return int(point[0] * frame_width), int(point[1] * frame_height)


def scaled_polygons(
    polygon: Polygon, *, frame_width: int, frame_height: int
) -> list[tuple[int, int]]:
    return [
        scale_point_to_pixels(p, frame_width=frame_width, frame_height=frame_height)
        for p in polygon
    ]


def point_in_circle(
    point: Point,
    center: Point,
    radius_frac: float,
    *,
    frame_width: int,
    frame_height: int,
) -> bool:
    scale = min(frame_width, frame_height)
    px, py = scale_point_to_pixels(point, frame_width=frame_width, frame_height=frame_height)
    cx = center[0] * frame_width
    cy = center[1] * frame_height
    r = radius_frac * scale
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def is_ignored(
    pixel: tuple[int, int],
    cal: FieldCalibration,
    *,
    ignore_mask: np.ndarray | None = None,
) -> bool:
    mask = ignore_mask if ignore_mask is not None else cal.ignore_mask
    x, y = pixel
    h, w = mask.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(mask[y, x] > 0)


def resize_ignore_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    if mask.shape[1] == w and mask.shape[0] == h:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def ignore_mask_for_process(
    cal: FieldCalibration,
    frame_width: int,
    frame_height: int,
    process_scale: float,
) -> np.ndarray:
    """Scale calibration ignore mask to process-resolution for the runtime frame."""
    mask = cal.ignore_mask
    if (cal.frame_width, cal.frame_height) != (frame_width, frame_height):
        mask = cv2.resize(
            cal.ignore_mask,
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST,
        )
    proc_w = max(1, int(frame_width * process_scale))
    proc_h = max(1, int(frame_height * process_scale))
    return resize_ignore_mask(mask, (proc_w, proc_h))


def classify_point(
    pixel: tuple[int, int],
    cal: FieldCalibration,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    ignore_mask: np.ndarray | None = None,
    ignore_check: bool = True,
) -> Classification:
    fw = frame_width if frame_width is not None else cal.frame_width
    fh = frame_height if frame_height is not None else cal.frame_height
    x, y = pixel
    norm: Point = (x / fw, y / fh)
    ignored = is_ignored(pixel, cal, ignore_mask=ignore_mask) if ignore_check else False
    in_roi = False if ignored else point_in_polygon(norm, cal.roi)
    return Classification(
        ignored=ignored,
        in_roi=in_roi,
        in_strike_zone=point_in_polygon(norm, cal.strike_zone),
        in_release_zone=point_in_circle(
            norm,
            cal.release_center,
            cal.release_radius,
            frame_width=fw,
            frame_height=fh,
        ),
    )
