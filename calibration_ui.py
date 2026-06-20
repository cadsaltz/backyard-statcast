"""Interactive field calibration — frozen scrub for files, rolling live feed."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

from calibration import FieldCalibration

if TYPE_CHECKING:
    from frame_source import FileFrameSource, FrameSource

WIN = "Calibrate Field"
Mode = Enum("Mode", ["ROI", "IGNORE", "STRIKE", "RELEASE"])

_MAX_DISPLAY_W = 1600
_MAX_DISPLAY_H = 900

_HIT_RADIUS = 18
_DEFAULT_BRUSH = 32
_BRUSH_MIN = 10
_BRUSH_STEP = 4

_SCRUB_PREV_KEYS = {ord(","), 65361, 2, 81}
_SCRUB_NEXT_KEYS = {ord("."), 65363, 3, 83}

_MODE_VERTEX_COLOR = {
    Mode.ROI: (0, 255, 0),
    Mode.STRIKE: (255, 128, 0),
    Mode.RELEASE: (255, 0, 255),
}


def _default_window_size(frame_width: int, frame_height: int) -> tuple[int, int]:
    scale = min(_MAX_DISPLAY_W / frame_width, _MAX_DISPLAY_H / frame_height, 1.0)
    return max(640, int(frame_width * scale)), max(360, int(frame_height * scale))


def _apply_ignore_tint(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> None:
    if not cv2.countNonZero(mask):
        return
    tinted = image.copy()
    tinted[mask > 0] = (0, 0, 255)
    cv2.addWeighted(tinted, alpha, image, 1.0 - alpha, 0, image)


def _nearest_point_index(
    points: list[tuple[int, int]], x: int, y: int, radius: int
) -> int | None:
    r2 = radius * radius
    for i, (px, py) in enumerate(points):
        if (px - x) ** 2 + (py - y) ** 2 <= r2:
            return i
    return None


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
        self.brush_radius = _DEFAULT_BRUSH
        self.release_center: tuple[int, int] | None = None
        self.release_radius_px: int = int(0.08 * min(w, h))
        self.cursor: tuple[int, int] | None = None
        self.drag: str | None = None
        self.sizing_release = False

    def set_frame(self, frame: np.ndarray, *, clear_ignore: bool = False) -> None:
        h, w = frame.shape[:2]
        if w != self.frame_width or h != self.frame_height:
            raise ValueError(
                f"Frame size changed to {w}x{h}, expected {self.frame_width}x{self.frame_height}"
            )
        self.base = frame.copy()
        if clear_ignore:
            self.ignore_mask[:] = 0


def _frame_coords(state: _UIState, x: int, y: int) -> tuple[int, int]:
    return (
        max(0, min(int(x), state.frame_width - 1)),
        max(0, min(int(y), state.frame_height - 1)),
    )


def _norm(state: _UIState, x: int, y: int) -> tuple[float, float]:
    return x / state.frame_width, y / state.frame_height


def _paint_ignore(state: _UIState, x: int, y: int) -> None:
    cv2.circle(state.ignore_mask, (x, y), state.brush_radius, 255, -1)


def _update_drag(state: _UIState, x: int, y: int) -> None:
    if state.drag is None:
        return
    if state.drag.startswith("roi:"):
        state.roi_points[int(state.drag.split(":")[1])] = (x, y)
    elif state.drag.startswith("strike:"):
        state.strike_points[int(state.drag.split(":")[1])] = (x, y)
    elif state.drag == "release_center" and state.release_center:
        state.release_center = (x, y)
    elif state.drag == "ignore":
        _paint_ignore(state, x, y)
    elif state.drag == "release_radius" and state.release_center:
        cx, cy = state.release_center
        state.release_radius_px = max(4, int(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5))


def _on_mouse(event, x, y, flags, userdata):
    state: _UIState = userdata
    fx, fy = _frame_coords(state, x, y)

    if event == cv2.EVENT_MOUSEMOVE:
        state.cursor = (fx, fy)
        if flags & cv2.EVENT_FLAG_LBUTTON:
            if state.drag is not None:
                _update_drag(state, fx, fy)
            elif state.sizing_release and state.release_center:
                cx, cy = state.release_center
                state.release_radius_px = max(
                    4, int(((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5)
                )
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        state.cursor = (fx, fy)
        if state.mode == Mode.ROI:
            hit = _nearest_point_index(state.roi_points, fx, fy, _HIT_RADIUS)
            if hit is not None:
                state.drag = f"roi:{hit}"
            elif len(state.roi_points) < 4:
                state.roi_points.append((fx, fy))
        elif state.mode == Mode.STRIKE:
            hit = _nearest_point_index(state.strike_points, fx, fy, _HIT_RADIUS)
            if hit is not None:
                state.drag = f"strike:{hit}"
            elif len(state.strike_points) < 4:
                state.strike_points.append((fx, fy))
        elif state.mode == Mode.RELEASE:
            if state.release_center is not None:
                cx, cy = state.release_center
                dist = int(((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5)
                if (fx - cx) ** 2 + (fy - cy) ** 2 <= _HIT_RADIUS ** 2:
                    state.drag = "release_center"
                elif abs(dist - state.release_radius_px) <= _HIT_RADIUS:
                    state.drag = "release_radius"
                else:
                    state.release_center = (fx, fy)
                    state.sizing_release = True
            else:
                state.release_center = (fx, fy)
                state.sizing_release = True
        elif state.mode == Mode.IGNORE:
            state.drag = "ignore"
            _paint_ignore(state, fx, fy)
    elif event == cv2.EVENT_LBUTTONUP:
        state.drag = None
        state.sizing_release = False


def _draw_poly(out: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(points) >= 2:
        pts = np.array(points, np.int32)
        closed = len(points) == 4
        cv2.polylines(out, [pts], closed, color, 2)
    for p in points:
        cv2.circle(out, p, 7, color, -1)
        cv2.circle(out, p, 9, (255, 255, 255), 1)


def _draw_cursor_preview(state: _UIState, out: np.ndarray) -> None:
    if state.cursor is None:
        return
    cx, cy = state.cursor

    if state.mode == Mode.ROI:
        color = _MODE_VERTEX_COLOR[Mode.ROI]
        if len(state.roi_points) >= 4:
            hit = _nearest_point_index(state.roi_points, cx, cy, _HIT_RADIUS)
            if hit is not None:
                cv2.circle(out, state.roi_points[hit], 11, color, 2)
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
        else:
            cv2.circle(out, (cx, cy), 7, color, -1)
            cv2.circle(out, (cx, cy), 9, (255, 255, 255), 1)
            if state.roi_points:
                cv2.line(out, state.roi_points[-1], (cx, cy), color, 1, cv2.LINE_AA)
            if len(state.roi_points) == 3:
                cv2.line(out, (cx, cy), state.roi_points[0], color, 1, cv2.LINE_AA)

    elif state.mode == Mode.STRIKE:
        color = _MODE_VERTEX_COLOR[Mode.STRIKE]
        if len(state.strike_points) >= 4:
            hit = _nearest_point_index(state.strike_points, cx, cy, _HIT_RADIUS)
            if hit is not None:
                cv2.circle(out, state.strike_points[hit], 11, color, 2)
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
        else:
            cv2.circle(out, (cx, cy), 7, color, -1)
            cv2.circle(out, (cx, cy), 9, (255, 255, 255), 1)
            if state.strike_points:
                cv2.line(out, state.strike_points[-1], (cx, cy), color, 1, cv2.LINE_AA)
            if len(state.strike_points) == 3:
                cv2.line(out, (cx, cy), state.strike_points[0], color, 1, cv2.LINE_AA)

    elif state.mode == Mode.IGNORE:
        cv2.circle(out, (cx, cy), state.brush_radius, (0, 0, 255), 2)
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), state.brush_radius, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

    elif state.mode == Mode.RELEASE:
        color = _MODE_VERTEX_COLOR[Mode.RELEASE]
        if state.release_center is None or state.sizing_release or state.drag == "release_radius":
            radius = state.release_radius_px
            if (state.sizing_release or state.drag == "release_radius") and state.release_center:
                radius = max(
                    4,
                    int(
                        ((cx - state.release_center[0]) ** 2 + (cy - state.release_center[1]) ** 2)
                        ** 0.5
                    ),
                )
            center = state.release_center or (cx, cy)
            cv2.circle(out, center, radius, color, 2)
            cv2.circle(out, center, 7, color, -1)
            cv2.circle(out, center, 9, (255, 255, 255), 1)
        else:
            cv2.circle(out, (cx, cy), 7, color, -1)
            cv2.circle(out, (cx, cy), 9, (255, 255, 255), 1)


def _render(
    state: _UIState,
    *,
    live: bool,
    frame_index: int | None = None,
    frame_total: int | None = None,
) -> np.ndarray:
    out = state.base.copy()
    _apply_ignore_tint(out, state.ignore_mask)

    _draw_poly(out, state.roi_points, _MODE_VERTEX_COLOR[Mode.ROI])
    _draw_poly(out, state.strike_points, _MODE_VERTEX_COLOR[Mode.STRIKE])

    if state.release_center and not (state.sizing_release or state.drag == "release_radius"):
        cx, cy = state.release_center
        color = _MODE_VERTEX_COLOR[Mode.RELEASE]
        cv2.circle(out, (cx, cy), state.release_radius_px, color, 2)
        cv2.circle(out, (cx, cy), 7, color, -1)
        cv2.circle(out, (cx, cy), 9, (255, 255, 255), 1)

    _draw_cursor_preview(state, out)

    if live:
        status = "LIVE — draw on rolling feed"
    elif frame_index is not None and frame_total is not None:
        status = f"Frame {frame_index + 1}/{frame_total}  (,/. or arrows to scrub)"
    elif frame_index is not None:
        status = f"Frame {frame_index + 1}  (,/. or arrows to scrub)"
    else:
        status = f"Frame {state.frame_width}x{state.frame_height}"

    help_lines = [
        f"Mode: {state.mode.name}  |  1=ROI 2=Ignore 3=Strike 4=Release",
        "[ ]=brush  u=clear ignore  c=clear mode  drag points  Enter/S=start  q=quit",
        status,
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


def _open_calibration_window(state: _UIState) -> None:
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_ASPECT_RATIO, 1)
    win_w, win_h = _default_window_size(state.frame_width, state.frame_height)
    cv2.resizeWindow(WIN, win_w, win_h)
    cv2.setMouseCallback(WIN, _on_mouse, state)


def _handle_key(state: _UIState, key: int) -> str | None:
    """Return 'quit', 'start', or None."""
    if key in (ord("q"), 27):
        return "quit"
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
        state.brush_radius = max(_BRUSH_MIN, state.brush_radius - _BRUSH_STEP)
    if key == ord("]") and state.mode == Mode.IGNORE:
        state.brush_radius += _BRUSH_STEP
    if key in (13, ord("s")):
        return "start"
    return None


def _finish_calibration(state: _UIState) -> FieldCalibration | None:
    try:
        return _build_calibration(state)
    except ValueError as exc:
        print(f"Calibration incomplete: {exc}")
        return None


def _run_live_calibration(frame_source: FrameSource) -> FieldCalibration | None:
    """Rolling live feed — tracking starts only after Enter/S."""
    state: _UIState | None = None
    print("Live calibration: feed is rolling — press Enter when regions are ready.")

    while True:
        frame = frame_source.read()
        if frame is None:
            raise SystemExit("Live feed ended during calibration.")
        if state is None:
            state = _UIState(frame)
            _open_calibration_window(state)
        else:
            state.set_frame(frame)

        cv2.imshow(WIN, _render(state, live=True))
        action = _handle_key(state, cv2.waitKey(1) & 0xFF)

        if action == "quit":
            cv2.destroyWindow(WIN)
            return None
        if action == "start":
            cal = _finish_calibration(state)
            if cal is None:
                continue
            cv2.destroyWindow(WIN)
            return cal


def _run_file_calibration(source: FileFrameSource) -> FieldCalibration | None:
    """Frozen frame with ,/. scrubbing to pick a calibration frame."""
    frame_idx = 0
    frame = source.seek(0)
    if frame is None:
        raise SystemExit("Could not read first frame for calibration.")

    total = source.frame_count
    state = _UIState(frame)
    _open_calibration_window(state)

    while True:
        cv2.imshow(
            WIN,
            _render(state, live=False, frame_index=frame_idx, frame_total=total),
        )
        key = cv2.waitKeyEx(30)
        action = _handle_key(state, key & 0xFF)

        if action == "quit":
            cv2.destroyWindow(WIN)
            return None
        if action == "start":
            cal = _finish_calibration(state)
            if cal is None:
                continue
            cv2.destroyWindow(WIN)
            return cal

        if key in _SCRUB_PREV_KEYS and frame_idx > 0:
            frame_idx -= 1
            new_frame = source.seek(frame_idx)
            if new_frame is not None:
                state.set_frame(new_frame, clear_ignore=True)
        elif key in _SCRUB_NEXT_KEYS:
            if total is not None and frame_idx >= total - 1:
                continue
            frame_idx += 1
            new_frame = source.seek(frame_idx)
            if new_frame is None:
                frame_idx -= 1
                continue
            state.set_frame(new_frame, clear_ignore=True)
            if total is None:
                total = source.frame_count


def run_calibration_ui(frame_source: FrameSource, *, live: bool) -> FieldCalibration | None:
    """Run calibration UI. Live = rolling feed; file = frozen frame with scrub."""
    from frame_source import FileFrameSource

    if live:
        return _run_live_calibration(frame_source)
    if isinstance(frame_source, FileFrameSource):
        return _run_file_calibration(frame_source)

    frame = frame_source.read()
    if frame is None:
        raise SystemExit("Could not read frame for calibration.")
    state = _UIState(frame)
    _open_calibration_window(state)
    while True:
        cv2.imshow(WIN, _render(state, live=False))
        action = _handle_key(state, cv2.waitKey(30) & 0xFF)
        if action == "quit":
            cv2.destroyWindow(WIN)
            return None
        if action == "start":
            cal = _finish_calibration(state)
            if cal is None:
                continue
            cv2.destroyWindow(WIN)
            return cal
