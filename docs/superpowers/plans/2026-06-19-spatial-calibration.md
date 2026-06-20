# Spatial Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-tracking calibration phase where the user draws ROI, ignore zones, strike zone, and release zone on the first frame, then run full-frame ball tracking with ignore masking and ROI-based pitch-active gating while always showing the validation dot (Choice B).

**Architecture:** Store calibration as normalized geometry in `calibration.py` plus an ignore-mask PNG sidecar. Pure geometry and mask helpers live in `spatial_filter.py`. OpenCV mouse UI in `calibration_ui.py`. `track_ball.py` runs calibration first, then passes a process-resolution ignore mask into `BallTracker.process()` and manages a minimal `pitch_active` flag in the main loop. Detection math stays global; spatial rules apply after the density peak is found.

**Tech Stack:** Python 3.12, OpenCV (`opencv-python`), NumPy, pytest (new dev dependency for unit tests)

**Design spec:** [2026-06-19-spatial-calibration-design.md](../specs/2026-06-19-spatial-calibration-design.md)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `calibration.py` | Create | `FieldCalibration` dataclass, JSON + PNG load/save |
| `spatial_filter.py` | Create | Polygon/circle tests, mask resize, detection classification |
| `calibration_ui.py` | Create | Interactive first-frame calibration with mouse/keyboard |
| `track_ball.py` | Modify | Calibration phase, ignore mask in `process()`, overlays, `pitch_active` |
| `tests/test_spatial_filter.py` | Create | Geometry unit tests |
| `tests/test_calibration.py` | Create | Serialization round-trip tests |
| `requirements-dev.txt` | Create | `pytest` for test runner |
| `README.md` | Modify | Calibration usage section |

---

### Task 1: Test infrastructure

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add dev requirements**

Create `requirements-dev.txt`:

```
pytest>=8.0
```

- [ ] **Step 2: Create empty test package**

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures for backyard-statcast tests."""
```

- [ ] **Step 3: Install and verify pytest**

Run: `pip install -r requirements-dev.txt && python -m pytest --version`  
Expected: pytest version prints (e.g. `pytest 8.x.x`).

- [ ] **Step 4: Commit**

```bash
git add requirements-dev.txt tests/__init__.py tests/conftest.py
git commit -m "chore: add pytest dev dependency and tests package"
```

---

### Task 2: `FieldCalibration` dataclass and persistence

**Files:**
- Create: `calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write failing serialization test**

Create `tests/test_calibration.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from calibration import FieldCalibration, load_calibration, save_calibration


def test_field_calibration_round_trip(tmp_path: Path):
    ignore = np.zeros((720, 1280), dtype=np.uint8)
    ignore[100:200, 300:400] = 255

    cal = FieldCalibration(
        frame_width=1280,
        frame_height=720,
        roi=[(0.1, 0.2), (0.9, 0.2), (0.9, 0.9), (0.1, 0.9)],
        strike_zone=[(0.4, 0.5), (0.6, 0.5), (0.6, 0.8), (0.4, 0.8)],
        release_center=(0.5, 0.15),
        release_radius=0.08,
        ignore_mask=ignore,
    )

    json_path = tmp_path / "field.json"
    save_calibration(cal, json_path)
    loaded = load_calibration(json_path)

    assert loaded.frame_width == 1280
    assert loaded.frame_height == 720
    assert loaded.roi == cal.roi
    assert loaded.strike_zone == cal.strike_zone
    assert loaded.release_center == cal.release_center
    assert loaded.release_radius == pytest.approx(0.08)
    assert loaded.ignore_mask.shape == (720, 1280)
    assert loaded.ignore_mask[150, 350] == 255


def test_load_missing_ignore_png(tmp_path: Path):
    json_path = tmp_path / "field.json"
    json_path.write_text(
        json.dumps(
            {
                "frame_width": 640,
                "frame_height": 480,
                "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "strike_zone": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
                "release_center": [0.5, 0.1],
                "release_radius": 0.05,
            }
        )
    )
    loaded = load_calibration(json_path)
    assert loaded.ignore_mask.shape == (480, 640)
    assert loaded.ignore_mask.max() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibration.py -v`  
Expected: FAIL — `ModuleNotFoundError: No module named 'calibration'`

- [ ] **Step 3: Implement `calibration.py`**

Create `calibration.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_calibration.py -v`  
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add calibration.py tests/test_calibration.py
git commit -m "feat: add FieldCalibration persistence"
```

---

### Task 3: Spatial filter geometry helpers

**Files:**
- Create: `spatial_filter.py`
- Create: `tests/test_spatial_filter.py`

- [ ] **Step 1: Write failing geometry tests**

Create `tests/test_spatial_filter.py`:

```python
import numpy as np

from calibration import FieldCalibration
from spatial_filter import (
    classify_point,
    is_ignored,
    point_in_circle,
    point_in_polygon,
    resize_ignore_mask,
    scaled_polygons,
)


def _sample_cal() -> FieldCalibration:
    ignore = np.zeros((100, 200), dtype=np.uint8)
    ignore[40:60, 80:120] = 255
    return FieldCalibration(
        frame_width=200,
        frame_height=100,
        roi=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        strike_zone=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
        release_center=(0.5, 0.1),
        release_radius=0.1,
        ignore_mask=ignore,
    )


def test_point_in_polygon():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon((0.5, 0.5), square) is True
    assert point_in_polygon((1.5, 0.5), square) is False


def test_point_in_circle():
    assert point_in_circle((0.5, 0.1), (0.5, 0.1), 0.1, frame_width=200, frame_height=100)
    assert not point_in_circle((0.9, 0.9), (0.5, 0.1), 0.05, frame_width=200, frame_height=100)


def test_is_ignored():
    cal = _sample_cal()
    assert is_ignored((100, 50), cal) is True
    assert is_ignored((10, 10), cal) is False


def test_classify_point():
    cal = _sample_cal()
    cls = classify_point((100, 50), cal)
    assert cls.ignored is True
    assert cls.in_roi is False

    cls2 = classify_point((100, 50), cal, ignore_check=False)
    assert cls2.ignored is False
    assert cls2.in_roi is True


def test_resize_ignore_mask():
    cal = _sample_cal()
    small = resize_ignore_mask(cal.ignore_mask, (100, 50))
    assert small.shape == (50, 100)


def test_scaled_polygons():
    cal = _sample_cal()
    roi_px = scaled_polygons(cal.roi, frame_width=200, frame_height=100)
    assert roi_px[0] == (0, 0)
    assert roi_px[2] == (200, 100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spatial_filter.py -v`  
Expected: FAIL — `ModuleNotFoundError: No module named 'spatial_filter'`

- [ ] **Step 3: Implement `spatial_filter.py`**

Create `spatial_filter.py`:

```python
"""Spatial geometry helpers for field calibration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calibration import FieldCalibration, Point, Polygon

__all__ = [
    "Classification",
    "classify_point",
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
    return mask[y, x] > 0


def resize_ignore_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    if mask.shape[1] == w and mask.shape[0] == h:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def classify_point(
    pixel: tuple[int, int],
    cal: FieldCalibration,
    *,
    ignore_mask: np.ndarray | None = None,
    ignore_check: bool = True,
) -> Classification:
    x, y = pixel
    norm: Point = (x / cal.frame_width, y / cal.frame_height)
    ignored = is_ignored(pixel, cal, ignore_mask=ignore_mask) if ignore_check else False
    return Classification(
        ignored=ignored,
        in_roi=point_in_polygon(norm, cal.roi),
        in_strike_zone=point_in_polygon(norm, cal.strike_zone),
        in_release_zone=point_in_circle(
            norm,
            cal.release_center,
            cal.release_radius,
            frame_width=cal.frame_width,
            frame_height=cal.frame_height,
        ),
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_spatial_filter.py tests/test_calibration.py -v`  
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add spatial_filter.py tests/test_spatial_filter.py
git commit -m "feat: add spatial filter geometry helpers"
```

---

### Task 4: Apply ignore mask inside `BallTracker.process()`

**Files:**
- Modify: `track_ball.py` (BallTracker class, ~lines 32–173)

- [ ] **Step 1: Extend `TrackResult` with classification fields**

In `track_ball.py`, update `TrackResult`:

```python
@dataclass
class TrackResult:
    density_peak: tuple[int, int] | None  # full-frame coords
    density_peak_proc: tuple[int, int] | None = None
    foreground: np.ndarray | None = None
    color_fg: np.ndarray | None = None
    rejected_by_ignore: bool = False
    raw_peak_proc: tuple[int, int] | None = None  # before ignore filter
```

- [ ] **Step 2: Add ignore mask state to `BallTracker.__init__`**

Add parameters and fields:

```python
def __init__(
    self,
    process_scale: float = 0.5,
    ...
    ignore_mask_proc: np.ndarray | None = None,
) -> None:
    ...
    self.ignore_mask_proc = ignore_mask_proc
```

Add method:

```python
def set_ignore_mask_proc(self, mask: np.ndarray | None) -> None:
    self.ignore_mask_proc = mask
```

- [ ] **Step 3: Filter peak by ignore mask in `process()`**

After `peak_small = self._density_peak(color_mask)`, before converting to full coords:

```python
raw_peak_small = peak_small
rejected = False
if peak_small and self.ignore_mask_proc is not None:
    px, py = peak_small
    h, w = self.ignore_mask_proc.shape[:2]
    if 0 <= px < w and 0 <= py < h and self.ignore_mask_proc[py, px] > 0:
        peak_small = None
        rejected = True
```

Update the return:

```python
return TrackResult(
    density_peak=peak_full,
    density_peak_proc=peak_small,
    foreground=foreground,
    color_fg=color_fg,
    rejected_by_ignore=rejected,
    raw_peak_proc=raw_peak_small,
)
```

- [ ] **Step 4: Manual smoke test (no calibration UI yet)**

Temporarily in `main()`, after tracker creation, build a dummy mask and assign it:

```python
# TEMP smoke test — remove after Task 6
h, w = int(1080 * process_scale), int(1920 * process_scale)
dummy = np.zeros((h, w), dtype=np.uint8)
dummy[0:50, :] = 255
tracker.set_ignore_mask_proc(dummy)
```

Run: `python track_ball.py --source /dev/video42` (or a video file)  
Expected: no crash; dot never appears in top 50 rows of process space.

Remove the TEMP block before committing Task 4.

- [ ] **Step 5: Commit**

```bash
git add track_ball.py
git commit -m "feat: apply process-resolution ignore mask in BallTracker"
```

---

### Task 5: Overlay drawing and pitch-active logic in main loop

**Files:**
- Modify: `track_ball.py` (draw helpers + main loop)

- [ ] **Step 1: Add overlay drawing functions**

Add to `track_ball.py`:

```python
from calibration import FieldCalibration
from spatial_filter import scaled_polygons, scale_point_to_pixels


def draw_calibration_overlay(frame: np.ndarray, cal: FieldCalibration) -> None:
    h, w = frame.shape[:2]
    roi = np.array(scaled_polygons(cal.roi, frame_width=w, frame_height=h), np.int32)
    strike = np.array(
        scaled_polygons(cal.strike_zone, frame_width=w, frame_height=h), np.int32
    )
    cv2.polylines(frame, [roi], True, (0, 255, 0), 2)
    cv2.polylines(frame, [strike], True, (255, 128, 0), 2)

    cx, cy = scale_point_to_pixels(
        cal.release_center, frame_width=w, frame_height=h
    )
    radius_px = int(cal.release_radius * min(w, h))
    cv2.circle(frame, (cx, cy), radius_px, (255, 0, 255), 2)

    if cal.ignore_mask is not None and cv2.countNonZero(cal.ignore_mask):
        mask_full = cal.ignore_mask
        if mask_full.shape[:2] != (h, w):
            mask_full = cv2.resize(mask_full, (w, h), interpolation=cv2.INTER_NEAREST)
        tint = frame.copy()
        tint[mask_full > 0] = (40, 40, 40)
        cv2.addWeighted(tint, 0.35, frame, 0.65, 0, frame)


def draw_track_dot(
    img: np.ndarray,
    pt: tuple[int, int] | None,
    *,
    pitch_active: bool,
) -> None:
    if pt is None:
        return
    color = (0, 0, 255) if pitch_active else (0, 255, 255)
    cv2.circle(img, pt, 7, color, -1)
    cv2.circle(img, pt, 9, (255, 255, 255), 2)
```

Replace existing `draw_dot` calls with `draw_track_dot`.

- [ ] **Step 2: Add pitch-active tracker state in `main()`**

After tracker creation (once calibration exists — wire fully in Task 6; for now accept optional cal):

```python
PITCH_IDLE_FRAMES = 30
pitch_active = False
idle_frames = 0
```

Inside the loop, after `result = tracker.process(...)`:

```python
detection = result.density_peak if result else None
if detection:
    idle_frames = 0
    if cal is not None:
        from spatial_filter import classify_point

        cls = classify_point(detection, cal)
        if cls.in_roi:
            pitch_active = True
    draw_track_dot(frame, detection, pitch_active=pitch_active)
else:
    idle_frames += 1
    if idle_frames >= PITCH_IDLE_FRAMES:
        pitch_active = False
        idle_frames = 0
```

Note: once `pitch_active` is True, it **stays** True while the ball is tracked—even outside ROI mid-flight. It clears only after `PITCH_IDLE_FRAMES` consecutive frames with no valid detection (or on `b` reset). Choice B only affects dot color outside ROI (yellow vs red), not whether tracking continues.

Reset on `b` key:

```python
if key == ord("b"):
    tracker.reset()
    pitch_active = False
    idle_frames = 0
```

Update status line:

```python
if pitch_active:
    status += "  PITCH"
else:
    status += "  track"
```

Use `draw_track_dot(frame, detection, pitch_active=pitch_active)`.

- [ ] **Step 3: Commit**

```bash
git add track_ball.py
git commit -m "feat: calibration overlays and pitch-active flag (Choice B)"
```

---

### Task 6: Interactive calibration UI

**Files:**
- Create: `calibration_ui.py`

- [ ] **Step 1: Implement `run_calibration_ui()`**

Create `calibration_ui.py`:

```python
"""Interactive first-frame field calibration."""

from __future__ import annotations

from enum import Enum, auto

import cv2
import numpy as np

from calibration import FieldCalibration, save_calibration

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
```

- [ ] **Step 2: Commit**

```bash
git add calibration_ui.py
git commit -m "feat: interactive first-frame calibration UI"
```

---

### Task 7: Wire calibration phase into `track_ball.py` main

**Files:**
- Modify: `track_ball.py` (CLI + main loop)

- [ ] **Step 1: Add CLI flags**

In `parse_args()`:

```python
p.add_argument(
    "--calibration",
    type=Path,
    default=None,
    help="Load saved calibration JSON (skip interactive UI).",
)
p.add_argument(
    "--save-calibration",
    type=Path,
    default=None,
    help="Save calibration to this JSON path after interactive setup.",
)
p.add_argument(
    "--skip-calibration",
    action="store_true",
    help="Run without spatial calibration (legacy behavior).",
)
```

- [ ] **Step 2: Calibration bootstrap before tracking loop**

At start of `main()`, after opening frame source, before creating windows:

```python
from calibration import load_calibration, save_calibration
from calibration_ui import run_calibration_ui
from spatial_filter import resize_ignore_mask

cal: FieldCalibration | None = None

if args.skip_calibration:
    cal = None
elif args.calibration is not None:
    cal = load_calibration(args.calibration)
else:
    first = frame_source.read()
    if first is None:
        raise SystemExit("Could not read first frame for calibration.")
    cal = run_calibration_ui(first)
    if cal is None:
        frame_source.release()
        raise SystemExit("Calibration cancelled.")
    if args.save_calibration is not None:
        save_calibration(cal, args.save_calibration)
        print(f"Saved calibration to {args.save_calibration}")
```

After `tracker = BallTracker(...)`:

```python
if cal is not None:
    proc_h = int(cal.frame_height * process_scale)
    proc_w = int(cal.frame_width * process_scale)
    ignore_proc = resize_ignore_mask(cal.ignore_mask, (proc_w, proc_h))
    tracker.set_ignore_mask_proc(ignore_proc)
```

In the tracking loop, call `draw_calibration_overlay(frame, cal)` when `cal is not None` (before drawing dot).

Remove any temporary smoke-test mask code from Task 4.

- [ ] **Step 3: Update startup print help**

```python
print("Calibration: 1=ROI 2=Ignore 3=Strike 4=Release  Enter=start")
print("Yellow dot = tracking validation  Red dot = pitch active (inside ROI)")
```

- [ ] **Step 4: Manual verification**

Run interactive calibration on a video file:

```bash
python track_ball.py --source path/to/clip.mp4 --save-calibration configs/field.json
```

Checklist:
1. First frame freezes in calibration window
2. All four region types can be defined
3. Enter starts tracking; overlays visible on live view
4. Painted ignore areas never show a dot
5. Dot appears outside ROI in yellow (Choice B)
6. Dot turns red when moved inside ROI; status shows `PITCH`
7. Reload: `python track_ball.py --source clip.mp4 --calibration configs/field.json`

Run unit tests:

```bash
python -m pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add track_ball.py
git commit -m "feat: wire calibration phase into track_ball main loop"
```

---

### Task 8: README and shell script updates

**Files:**
- Modify: `README.md`
- Modify: `run_tracker.sh` (optional `--save-calibration` hint)

- [ ] **Step 1: Document calibration in README**

Add section:

```markdown
## Field calibration

Before tracking, the first frame freezes so you can define:

1. **ROI** (4 clicks) — pitch-active region
2. **Ignore zones** (paint) — hard-masked false-positive areas
3. **Strike zone** (4 clicks) — overlay only
4. **Release zone** (click + drag radius) — overlay only

Keys during calibration: `1`–`4` switch mode, `[`/`]` brush size, `Enter` start, `q` quit.

```bash
# Interactive calibration, save for reuse
python track_ball.py --source /dev/video42 --save-calibration configs/field.json

# Reuse saved calibration
python track_ball.py --source /dev/video42 --calibration configs/field.json

# Legacy: no spatial filtering
python track_ball.py --source /dev/video42 --skip-calibration
```

Yellow dot = tracker validation (Choice B). Red dot = detection inside ROI (`pitch_active`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add field calibration usage"
```

---

## Spec Coverage Checklist

| Requirement | Task |
|-------------|------|
| 4-point ROI polygon | Task 6, 7 |
| Painted ignore zones (multiple strokes → one mask) | Task 6 |
| 4-point strike zone | Task 6 |
| Circular release zone | Task 6 |
| Full-frame detection unchanged | Task 4 (mask applied post-peak only) |
| Ignore hard reject | Task 4 |
| Choice B: always show validation dot | Task 5 |
| ROI → pitch_active only | Task 5 |
| Strike/release overlay only | Task 5, 6 |
| Normalized coordinate storage | Task 2 |
| Live + file sources | Task 7 (uses existing `FrameSource`) |
| No perspective transform | N/A — not implemented |
| No formal state machine | Task 5 — single boolean + idle counter |

---

## Future Work (out of scope)

- Wire same spatial filter into `detection_strategies.py` / `test_detection_strategies.py`
- ROI candidate prioritization for multi-candidate strategies
- Release zone trajectory origin estimation
- Strike zone pitch classification
- Formal pitch state machine with segmentation

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-19-spatial-calibration.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — implement tasks in this session with checkpoints

Which approach do you want?
