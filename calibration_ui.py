"""Interactive first-frame field calibration."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from calibration import FieldCalibration

WIN = "Calibrate Field"
Mode = Enum("Mode", ["ROI", "IGNORE", "STRIKE", "RELEASE"])


class _UIState:
    def __init__(self, frame: np.ndarray) -> None:
        self.base = frame.copy()
        h, w = frame.shape[:2]
        self.frame_width = w
        self.frame_height = h
        self.mode = Mode.ROI
        self.roi_points: list[tuple[int, int]] = []
        self.strike_points: list[tuple[int, int]] = []
        self.ignore_mask = np.zeros((h, w), dtype=np.uint8)
        self.brush_radius = 12
        self.release_center: tuple[int, int] | None = None
        self.release_radius_px: int = int(0.08 * min(w, h))
        self.drawing_release = False


def _norm(state: _UIState, x: int, y: int) -> tuple[float, float]:
    return x / state.frame_width, y / state.frame_height


def _on_mouse(event, x, y, flags, userdata):
    state: _UIState = userdata
    if event == cv2.EVENT_LBUTTONDOWN:
        if state.mode == Mode.ROI and len(state.roi_points) < 4:
            state.roi_points.append((x, y))
        elif state.mode == Mode.STRIKE and len(state.strike_points) < 4:
            state.strike_points.append((x, y))
        elif state.mode == Mode.RELEASE:
            state.release_center = (x, y)
            state.drawing_release = True
        elif state.mode == Mode.IGNORE:
            cv2.circle(state.ignore_mask, (x, y), state.brush_radius, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        if state.mode == Mode.IGNORE:
            cv2.circle(state.ignore_mask, (x, y), state.brush_radius, 255, -1)
        elif state.mode == Mode.RELEASE and state.drawing_release and state.release_center:
            cx, cy = state.release_center
            state.release_radius_px = max(4, int(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5))
    elif event == cv2.EVENT_LBUTTONUP and state.mode == Mode.RELEASE:
        state.drawing_release = False


def _render(state: _UIState) -> np.ndarray:
    out = state.base.copy()
    overlay = out.copy()
    overlay[state.ignore_mask > 0] = (40, 40, 40)
    cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

    def _poly(points, color):
        if len(points) >= 2:
            pts = np.array(points, np.int32)
            closed = len(points) == 4
            cv2.polylines(out, [pts], closed, color, 2)
            for p in points:
                cv2.circle(out, p, 4, color, -1)

    _poly(state.roi_points, (0, 255, 0))
    _poly(state.strike_points, (255, 128, 0))
    if state.release_center:
        cv2.circle(out, state.release_center, state.release_radius_px, (255, 0, 255), 2)
        cv2.circle(out, state.release_center, 3, (255, 0, 255), -1)

    help_lines = [
        f"Mode: {state.mode.name}  |  1=ROI 2=Ignore 3=Strike 4=Release",
        "[ ]=brush  u=undo ignore  c=clear mode  Enter/S=start  q=quit",
    ]
    for i, line in enumerate(help_lines):
        cv2.putText(out, line, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def _build_calibration(state: _UIState) -> FieldCalibration:
    if len(state.roi_points) != 4:
        raise ValueError("ROI requires 4 points")
    if len(state.strike_points) != 4:
        raise ValueError("Strike zone requires 4 points")
    if state.release_center is None:
        raise ValueError("Release zone requires a center click")

    roi = tuple(_norm(state, x, y) for x, y in state.roi_points)
    strike = tuple(_norm(state, x, y) for x, y in state.strike_points)
    rc = _norm(state, *state.release_center)
    radius_frac = state.release_radius_px / min(state.frame_width, state.frame_height)

    cal = FieldCalibration(
        frame_width=state.frame_width,
        frame_height=state.frame_height,
        roi=roi,  # type: ignore[arg-type]
        strike_zone=strike,  # type: ignore[arg-type]
        release_center=rc,
        release_radius=radius_frac,
        ignore_mask=state.ignore_mask.copy(),
    )
    cal.validate()
    return cal


def run_calibration_ui(frame: np.ndarray) -> FieldCalibration | None:
    """Run interactive calibration on a frozen frame. Returns None if user quits."""
    state = _UIState(frame)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, _on_mouse, state)

    while True:
        cv2.imshow(WIN, _render(state))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(WIN)
            return None
        if key == ord("1"):
            state.mode = Mode.ROI
        if key == ord("2"):
            state.mode = Mode.IGNORE
        if key == ord("3"):
            state.mode = Mode.STRIKE
        if key == ord("4"):
            state.mode = Mode.RELEASE
        if key == ord("c"):
            if state.mode == Mode.ROI:
                state.roi_points.clear()
            elif state.mode == Mode.STRIKE:
                state.strike_points.clear()
            elif state.mode == Mode.RELEASE:
                state.release_center = None
            elif state.mode == Mode.IGNORE:
                state.ignore_mask[:] = 0
        if key == ord("u") and state.mode == Mode.IGNORE:
            state.ignore_mask[:] = 0
        if key == ord("[") and state.mode == Mode.IGNORE:
            state.brush_radius = max(3, state.brush_radius - 2)
        if key == ord("]") and state.mode == Mode.IGNORE:
            state.brush_radius += 2
        if key in (13, ord("s")):
            try:
                cal = _build_calibration(state)
            except ValueError as exc:
                print(f"Calibration incomplete: {exc}")
                continue
            cv2.destroyWindow(WIN)
            return cal
