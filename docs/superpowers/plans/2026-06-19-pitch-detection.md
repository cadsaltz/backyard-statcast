# Pitch Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pitch state machine that records timestamped ball trajectories with fast release-based start detection, tolerant gap handling, end detection (impact / plate pass / terminal loss), and a live flight-path tracer that fades out after each pitch returns to IDLE.

**Architecture:** Pure pitch logic in `pitch_geometry.py` and `pitch_state.py` (orthogonal to `TrackState`). Visual-only overlay in `pitch_tracer.py`. `track_ball.py` main loop feeds each frame's `TrackResult` + spatial classification into `PitchStateMachine.update()`, then draws the live tracer during RECORDING and fading tracers during IDLE. Validation runs once in FINALIZING; no analytics during recording.

**Tech Stack:** Python 3.12, OpenCV, NumPy, pytest

**Design spec:** [2026-06-19-pitch-detection-design.md](../specs/2026-06-19-pitch-detection-design.md)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pitch_geometry.py` | Create | Release/strike bounds in process + full-frame pixels |
| `pitch_state.py` | Create | `PitchConfig`, `PitchSample`, `Pitch`, `PitchStateMachine` |
| `pitch_tracer.py` | Create | Live polyline + fading path overlay |
| `track_ball.py` | Modify | Wire pitch machine; replace `pitch_active`; draw tracers |
| `tests/test_pitch_geometry.py` | Create | Geometry helper tests |
| `tests/test_pitch_state.py` | Create | State machine unit tests |
| `tests/test_pitch_tracer.py` | Create | Tracer fade math tests |
| `README.md` | Modify | Pitch tracer + status line docs |

---

### Task 1: Pitch geometry helpers

**Files:**
- Create: `pitch_geometry.py`
- Create: `tests/test_pitch_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

Create `tests/test_pitch_geometry.py`:

```python
import pytest

from calibration import FieldCalibration
from pitch_geometry import FieldGeometry, build_field_geometry


@pytest.fixture
def sample_cal() -> FieldCalibration:
    import numpy as np

    return FieldCalibration(
        frame_width=1920,
        frame_height=1080,
        roi=[(0.3, 0.2), (0.7, 0.2), (0.7, 0.8), (0.3, 0.8)],
        strike_zone=[(0.55, 0.51), (0.58, 0.51), (0.58, 0.57), (0.55, 0.57)],
        release_center=(0.45, 0.33),
        release_radius=0.05,
        ignore_mask=np.zeros((1080, 1920), dtype=np.uint8),
    )


def test_build_field_geometry_process_coords(sample_cal: FieldCalibration):
    geom = build_field_geometry(sample_cal, frame_width=1920, frame_height=1080, process_scale=0.5)
    assert geom.proc_w == 960
    assert geom.proc_h == 540
    assert geom.release_cx_proc < geom.strike_leading_x_proc < geom.strike_trailing_x_proc
    assert geom.release_r_proc > 0


def test_in_release_zone(sample_cal: FieldCalibration):
    geom = build_field_geometry(sample_cal, frame_width=1920, frame_height=1080, process_scale=0.5)
    assert geom.in_release_zone(geom.release_cx_proc, geom.release_cy_proc)
    assert not geom.in_release_zone(0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pitch_geometry.py -v`  
Expected: FAIL with `ModuleNotFoundError: pitch_geometry`

- [ ] **Step 3: Implement `pitch_geometry.py`**

Create `pitch_geometry.py`:

```python
"""Pixel geometry derived from field calibration for pitch detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from calibration import FieldCalibration, Point
from spatial_filter import scale_point_to_pixels, scaled_polygons


@dataclass(frozen=True)
class FieldGeometry:
    proc_w: int
    proc_h: int
    full_w: int
    full_h: int
    release_cx_proc: int
    release_cy_proc: int
    release_r_proc: float
    release_cx_full: int
    release_cy_full: int
    release_r_full: float
    strike_leading_x_proc: int
    strike_trailing_x_proc: int
    strike_leading_x_full: int
    strike_trailing_x_full: int
    pass_margin_proc: int
    approach_margin_proc: int

    def in_release_zone(self, x_proc: int, y_proc: int, *, expand: float = 1.0) -> bool:
        r = self.release_r_proc * expand
        dx = x_proc - self.release_cx_proc
        dy = y_proc - self.release_cy_proc
        return dx * dx + dy * dy <= r * r

    def proc_to_full(self, x_proc: int, y_proc: int) -> tuple[int, int]:
        sx = self.full_w / self.proc_w
        sy = self.full_h / self.proc_h
        return int(x_proc * sx), int(y_proc * sy)


def _strike_x_bounds(strike_pts: list[tuple[int, int]]) -> tuple[int, int]:
    xs = [p[0] for p in strike_pts]
    return min(xs), max(xs)


def build_field_geometry(
    cal: FieldCalibration,
    *,
    frame_width: int,
    frame_height: int,
    process_scale: float,
    pass_margin_frac: float = 0.03,
    approach_margin_frac: float = 0.04,
) -> FieldGeometry:
    proc_w = max(1, int(frame_width * process_scale))
    proc_h = max(1, int(frame_height * process_scale))
    scale = min(frame_width, frame_height)

    rcx, rcy = scale_point_to_pixels(cal.release_center, frame_width=proc_w, frame_height=proc_h)
    r_proc = cal.release_radius * min(proc_w, proc_h)
    rcx_f, rcy_f = scale_point_to_pixels(cal.release_center, frame_width=frame_width, frame_height=frame_height)
    r_full = cal.release_radius * scale

    strike_proc = scaled_polygons(cal.strike_zone, frame_width=proc_w, frame_height=proc_h)
    strike_full = scaled_polygons(cal.strike_zone, frame_width=frame_width, frame_height=frame_height)
    leading_p, trailing_p = _strike_x_bounds(strike_proc)
    leading_f, trailing_f = _strike_x_bounds(strike_full)

    return FieldGeometry(
        proc_w=proc_w,
        proc_h=proc_h,
        full_w=frame_width,
        full_h=frame_height,
        release_cx_proc=rcx,
        release_cy_proc=rcy,
        release_r_proc=r_proc,
        release_cx_full=rcx_f,
        release_cy_full=rcy_f,
        release_r_full=r_full,
        strike_leading_x_proc=leading_p,
        strike_trailing_x_proc=trailing_p,
        strike_leading_x_full=leading_f,
        strike_trailing_x_full=trailing_f,
        pass_margin_proc=int(frame_width * process_scale * pass_margin_frac),
        approach_margin_proc=int(frame_width * process_scale * approach_margin_frac),
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pitch_geometry.py -v`  
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pitch_geometry.py tests/test_pitch_geometry.py docs/superpowers/specs/2026-06-19-pitch-detection-design.md
git commit -m "feat: add pitch geometry helpers from field calibration"
```

---

### Task 2: Pitch data model and config

**Files:**
- Create: `pitch_state.py` (partial — dataclasses + enum only)
- Create: `tests/test_pitch_state.py` (partial)

- [ ] **Step 1: Write failing enum/config test**

Create `tests/test_pitch_state.py`:

```python
from pitch_state import PitchConfig, PitchEndReason, PitchMode, PitchStartReason


def test_pitch_config_defaults():
    cfg = PitchConfig()
    assert cfg.ring_buffer_frames == 10
    assert cfg.tracer_fade_sec == 4.0
    assert cfg.dominance_ratio == 2.0


def test_pitch_modes_exist():
    assert PitchMode.IDLE.value == "idle"
    assert PitchMode.RECORDING.value == "recording"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pitch_state.py::test_pitch_config_defaults -v`  
Expected: FAIL — module not found

- [ ] **Step 3: Add dataclasses to `pitch_state.py`**

Create `pitch_state.py` (top of file):

```python
"""Pitch event state machine — orthogonal to TrackState."""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from pitch_geometry import FieldGeometry
from track_state import TrackMode


class PitchMode(Enum):
    IDLE = "idle"
    RECORDING = "recording"


class PitchStartReason(Enum):
    RELEASE_MOTION = "release_motion"


class PitchEndReason(Enum):
    STRIKE_CONTACT = "strike_contact"
    HIT_OR_DEFLECTION = "hit_or_deflection"
    PLATE_PASSED = "plate_passed"
    TRACK_LOST_TERMINAL = "track_lost_terminal"
    TIMEOUT = "timeout"
    INVALID_REACQUIRE = "invalid_reacquire"
    ABORTED = "aborted"


@dataclass
class PitchConfig:
    ring_buffer_frames: int = 10
    min_dx: float = 5.0
    dominance_ratio: float = 2.0
    min_dy: float = 2.0
    release_expand: float = 1.1
    min_samples_before_impact: int = 4
    impact_angle_deg: float = 45.0
    impact_angle_single_deg: float = 60.0
    impact_decel_ratio: float = 0.5
    terminal_lost_frames: int = 8
    max_gap_frames: int = 3
    max_implausible_frames: int = 5
    max_pitch_frames: int = 22
    max_pitch_sec: float = 1.5
    cooldown_frames: int = 18
    min_detected_samples: int = 6
    min_forward_px: float = 40.0
    reconnect_slack: float = 1.5
    max_step_px: float = 100.0
    max_total_gap_frames: int = 6
    max_reconnects: int = 2
    tracer_fade_sec: float = 4.0


@dataclass
class PitchSample:
    frame_index: int
    timestamp_sec: float
    x_proc: int
    y_proc: int
    x_full: int
    y_full: int
    detected: bool = True
    dx: float = 0.0
    dy: float = 0.0
    speed_px: float = 0.0
    track_mode: TrackMode | None = None
    coasting: bool = False
    density_score: float | None = None
    blob_area: float | None = None
    in_roi: bool = False
    in_release_zone: bool = False
    in_strike_zone: bool = False


@dataclass
class Pitch:
    pitch_id: str
    start_frame: int
    end_frame: int
    started_at_sec: float
    ended_at_sec: float
    start_reason: PitchStartReason
    end_reason: PitchEndReason
    samples: list[PitchSample] = field(default_factory=list)
    backfill_frames: int = 0
    gap_count: int = 0
    reconnect_count: int = 0
    valid: bool = True
    complete: bool = True
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pitch_state.py::test_pitch_config_defaults tests/test_pitch_state.py::test_pitch_modes_exist -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pitch_state.py tests/test_pitch_state.py
git commit -m "feat: add pitch data model and configuration"
```

---

### Task 3: Pitch start detection + ring buffer

**Files:**
- Modify: `pitch_state.py`
- Modify: `tests/test_pitch_state.py`

- [ ] **Step 1: Write failing start-detection test**

Append to `tests/test_pitch_state.py`:

```python
from pitch_geometry import build_field_geometry
from pitch_state import PitchFrameInput, PitchStateMachine


def _geom():
    import numpy as np
    from calibration import FieldCalibration

    cal = FieldCalibration(
        frame_width=1920,
        frame_height=1080,
        roi=[(0.3, 0.2), (0.7, 0.2), (0.7, 0.8), (0.3, 0.8)],
        strike_zone=[(0.55, 0.51), (0.58, 0.51), (0.58, 0.57), (0.55, 0.57)],
        release_center=(0.45, 0.33),
        release_radius=0.05,
        ignore_mask=np.zeros((1080, 1920), dtype=np.uint8),
    )
    return build_field_geometry(cal, frame_width=1920, frame_height=1080, process_scale=0.5)


def test_start_recording_on_release_motion():
    geom = _geom()
    sm = PitchStateMachine(geom)
    cx, cy = geom.release_cx_proc, geom.release_cy_proc

    # Frame 1: inside release, no prior — buffer only
    sm.update(PitchFrameInput(frame_index=1, timestamp_sec=0.0, x_proc=cx - 5, y_proc=cy, detected=True))
    assert sm.mode == PitchMode.IDLE

    # Frame 2: crossed center with strong rightward motion
    sm.update(
        PitchFrameInput(
            frame_index=2,
            timestamp_sec=0.033,
            x_proc=cx + 8,
            y_proc=cy + 1,
            detected=True,
        )
    )
    assert sm.mode == PitchMode.RECORDING
    assert len(sm.active_samples) >= 2
    assert sm.active_samples[0].frame_index == 1  # backfill
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pitch_state.py::test_start_recording_on_release_motion -v`  
Expected: FAIL — `PitchStateMachine` not defined

- [ ] **Step 3: Implement start detection and ring buffer**

Add to `pitch_state.py`:

```python
@dataclass
class PitchFrameInput:
    frame_index: int
    timestamp_sec: float
    x_proc: int | None = None
    y_proc: int | None = None
    detected: bool = False
    track_mode: TrackMode | None = None
    coasting: bool = False
    density_score: float | None = None
    blob_area: float | None = None
    in_roi: bool = False
    in_release_zone: bool = False
    in_strike_zone: bool = False


@dataclass
class PitchUpdateResult:
    mode: PitchMode
    pitch_completed: Pitch | None = None
    pitch_discarded: bool = False


class PitchStateMachine:
    def __init__(self, geom: FieldGeometry, config: PitchConfig | None = None) -> None:
        self.geom = geom
        self.config = config or PitchConfig()
        self.mode = PitchMode.IDLE
        self._ring: deque[PitchSample] = deque(maxlen=self.config.ring_buffer_frames)
        self._active: list[PitchSample] = []
        self._prev: PitchSample | None = None
        self._was_in_release = False
        self._cooldown = 0
        self._recording_started_at: float | None = None
        self._recording_start_frame: int = 0
        self._last_completed: Pitch | None = None

    @property
    def active_samples(self) -> list[PitchSample]:
        return list(self._active)

    def reset(self) -> None:
        self.mode = PitchMode.IDLE
        self._ring.clear()
        self._active.clear()
        self._prev = None
        self._was_in_release = False
        self._cooldown = 0
        self._recording_started_at = None

    def _make_sample(self, inp: PitchFrameInput) -> PitchSample | None:
        if not inp.detected or inp.x_proc is None or inp.y_proc is None:
            return None
        xf, yf = self.geom.proc_to_full(inp.x_proc, inp.y_proc)
        dx = dy = 0.0
        speed = 0.0
        if self._prev is not None and self._prev.detected:
            dx = float(inp.x_proc - self._prev.x_proc)
            dy = float(inp.y_proc - self._prev.y_proc)
            speed = math.hypot(dx, dy)
        return PitchSample(
            frame_index=inp.frame_index,
            timestamp_sec=inp.timestamp_sec,
            x_proc=inp.x_proc,
            y_proc=inp.y_proc,
            x_full=xf,
            y_full=yf,
            detected=True,
            dx=dx,
            dy=dy,
            speed_px=speed,
            track_mode=inp.track_mode,
            coasting=inp.coasting,
            density_score=inp.density_score,
            blob_area=inp.blob_area,
            in_roi=inp.in_roi,
            in_release_zone=inp.in_release_zone,
            in_strike_zone=inp.in_strike_zone,
        )

    def _horizontal_dominance(self, dx: float, dy: float) -> bool:
        cfg = self.config
        return abs(dx) >= cfg.dominance_ratio * max(abs(dy), cfg.min_dy)

    def _should_start(self, sample: PitchSample, in_release: bool, just_exited: bool) -> bool:
        cfg = self.config
        g = self.geom
        if self._cooldown > 0:
            return False
        rightward = sample.dx >= cfg.min_dx
        dominance = self._horizontal_dominance(sample.dx, sample.dy)
        crossed = sample.x_proc >= g.release_cx_proc
        spatial_ok = in_release or just_exited or sample.x_proc < g.strike_leading_x_proc
        return rightward and dominance and crossed and spatial_ok

    def _start_recording(self, sample: PitchSample, start_reason: PitchStartReason) -> None:
        self.mode = PitchMode.RECORDING
        self._active = list(self._ring)
        if not self._active or self._active[-1].frame_index != sample.frame_index:
            self._active.append(sample)
        self._recording_started_at = sample.timestamp_sec
        self._recording_start_frame = sample.frame_index
        self._prev = self._active[-1]
        self._gap_frames = 0
        self._implausible_streak = 0
        self._total_gaps = 0
        self._reconnects = 0
        self._impact_streak = 0
        self._start_reason = start_reason

    def update(self, inp: PitchFrameInput) -> PitchUpdateResult:
        if self._cooldown > 0:
            self._cooldown -= 1

        completed: Pitch | None = None
        discarded = False

        if inp.detected and inp.x_proc is not None and inp.y_proc is not None:
            in_release = self.geom.in_release_zone(
                inp.x_proc, inp.y_proc, expand=self.config.release_expand
            )
            just_exited = self._was_in_release and not in_release and (
                self._prev is not None and self._prev.dx > 0
            )
            sample = self._make_sample(inp)

            if self.mode == PitchMode.IDLE and sample is not None:
                self._ring.append(sample)
                if self._should_start(sample, in_release, just_exited):
                    self._start_recording(sample, PitchStartReason.RELEASE_MOTION)
            elif self.mode == PitchMode.RECORDING and sample is not None:
                self._active.append(sample)
                self._prev = sample
                self._gap_frames = 0

            self._was_in_release = in_release
            if sample is not None:
                self._prev = sample
        else:
            if self.mode == PitchMode.RECORDING:
                self._gap_frames = getattr(self, "_gap_frames", 0) + 1
                self._total_gaps = getattr(self, "_total_gaps", 0) + 1

        return PitchUpdateResult(mode=self.mode, pitch_completed=completed, pitch_discarded=discarded)
```

Note: end detection and finalize wired in Task 4–5. This task only proves start + backfill.

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_pitch_state.py::test_start_recording_on_release_motion -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pitch_state.py tests/test_pitch_state.py
git commit -m "feat: pitch start detection with ring buffer backfill"
```

---

### Task 4: End detection (impact, plate pass, terminal loss)

**Files:**
- Modify: `pitch_state.py`
- Modify: `tests/test_pitch_state.py`

- [ ] **Step 1: Write failing end-detection tests**

Append to `tests/test_pitch_state.py`:

```python
from pitch_state import PitchEndReason


def _recording_sm():
    geom = _geom()
    sm = PitchStateMachine(geom)
    cx, cy = geom.release_cx_proc, geom.release_cy_proc
    sm.update(PitchFrameInput(1, 0.0, cx - 5, cy, detected=True))
    sm.update(PitchFrameInput(2, 0.03, cx + 8, cy + 1, detected=True))
    assert sm.mode == PitchMode.RECORDING
    return sm, geom


def test_end_on_plate_passed():
    sm, geom = _recording_sm()
    x = geom.strike_trailing_x_proc + geom.pass_margin_proc + 5
    for i, fi in enumerate(range(3, 12)):
        result = sm.update(
            PitchFrameInput(fi, 0.05 * i, x + i * 2, geom.release_cy_proc, detected=True)
        )
    assert result.pitch_completed is not None
    assert result.pitch_completed.end_reason == PitchEndReason.PLATE_PASSED
    assert sm.mode == PitchMode.IDLE


def test_gap_tolerance_does_not_end_pitch():
    sm, geom = _recording_sm()
    cx = geom.release_cx_proc + 20
    sm.update(PitchFrameInput(3, 0.1, cx, geom.release_cy_proc, detected=True))
    sm.update(PitchFrameInput(4, 0.13, detected=False))
    sm.update(PitchFrameInput(5, 0.16, detected=False))
    assert sm.mode == PitchMode.RECORDING
    sm.update(PitchFrameInput(6, 0.19, cx + 10, geom.release_cy_proc, detected=True))
    assert sm.mode == PitchMode.RECORDING
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pitch_state.py::test_end_on_plate_passed tests/test_pitch_state.py::test_gap_tolerance_does_not_end_pitch -v`  
Expected: FAIL on plate pass (no end logic yet)

- [ ] **Step 3: Add end detection methods**

Add to `PitchStateMachine` in `pitch_state.py`:

```python
    def _velocity_angle_deg(self, ax: float, ay: float, bx: float, by: float) -> float:
        mag_a = math.hypot(ax, ay)
        mag_b = math.hypot(bx, by)
        if mag_a < 1e-6 or mag_b < 1e-6:
            return 0.0
        dot = ax * bx + ay * by
        cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        return math.degrees(math.acos(cos_theta))

    def _detect_impact(self) -> PitchEndReason | None:
        pts = [s for s in self._active if s.detected]
        if len(pts) < self.config.min_samples_before_impact + 1:
            return None
        a, b, c = pts[-3], pts[-2], pts[-1]
        v_before = (b.x_proc - a.x_proc, b.y_proc - a.y_proc)
        v_after = (c.x_proc - b.x_proc, c.y_proc - b.y_proc)
        angle = self._velocity_angle_deg(v_before[0], v_before[1], v_after[0], v_after[1])
        speed_before = math.hypot(*v_before)
        speed_after = math.hypot(*v_after)
        cfg = self.config
        single_hit = angle >= cfg.impact_angle_single_deg
        streak_hit = angle >= cfg.impact_angle_deg
        decel = speed_before > 1e-3 and speed_after / speed_before < cfg.impact_decel_ratio
        if single_hit or decel:
            self._impact_streak += 1
        elif streak_hit:
            self._impact_streak += 1
        else:
            self._impact_streak = 0
        if self._impact_streak >= 2 or single_hit:
            if c.in_strike_zone:
                return PitchEndReason.STRIKE_CONTACT
            return PitchEndReason.HIT_OR_DEFLECTION
        return None

    def _in_approach_zone(self, x_proc: int) -> bool:
        return x_proc >= self.geom.strike_leading_x_proc - self.geom.approach_margin_proc

    def _check_end_while_recording(self) -> PitchEndReason | None:
        pts = [s for s in self._active if s.detected]
        if not pts:
            return None
        last = pts[-1]
        g = self.geom
        cfg = self.config

        impact = self._detect_impact()
        if impact is not None:
            return impact

        if last.x_proc >= g.strike_trailing_x_proc + g.pass_margin_proc:
            return PitchEndReason.PLATE_PASSED

        if self._in_approach_zone(last.x_proc) and self._gap_frames >= cfg.terminal_lost_frames:
            return PitchEndReason.TRACK_LOST_TERMINAL

        if len(self._active) >= cfg.max_pitch_frames:
            return PitchEndReason.TIMEOUT
        if (
            self._recording_started_at is not None
            and last.timestamp_sec - self._recording_started_at > cfg.max_pitch_sec
        ):
            return PitchEndReason.TIMEOUT

        return None

    def _plausible_reconnect(self, sample: PitchSample, gap_frames: int) -> bool:
        pts = [s for s in self._active if s.detected]
        if not pts:
            return True
        last = pts[-1]
        predicted_x = last.x_proc + last.dx * gap_frames
        predicted_y = last.y_proc + last.dy * gap_frames
        dist = math.hypot(sample.x_proc - predicted_x, sample.y_proc - predicted_y)
        max_allowed = self.config.max_step_px + last.speed_px * gap_frames * self.config.reconnect_slack
        return dist <= max_allowed
```

Update the RECORDING branch in `update()`:

```python
            elif self.mode == PitchMode.RECORDING and sample is not None:
                gap = self._gap_frames
                if gap > 0:
                    if self._plausible_reconnect(sample, gap):
                        self._reconnects += 1
                        self._implausible_streak = 0
                    else:
                        self._implausible_streak += 1
                        if self._implausible_streak >= self.config.max_implausible_frames:
                            completed = self._finalize(PitchEndReason.INVALID_REACQUIRE)
                            discarded = completed is None
                            return PitchUpdateResult(
                                mode=self.mode,
                                pitch_completed=completed,
                                pitch_discarded=discarded,
                            )
                self._active.append(sample)
                self._prev = sample
                self._gap_frames = 0
                end = self._check_end_while_recording()
                if end is not None:
                    completed = self._finalize(end)
                    discarded = completed is None
                    return PitchUpdateResult(
                        mode=self.mode,
                        pitch_completed=completed,
                        pitch_discarded=discarded,
                    )
```

And the no-detection branch:

```python
        else:
            if self.mode == PitchMode.RECORDING:
                self._gap_frames = getattr(self, "_gap_frames", 0) + 1
                self._total_gaps = getattr(self, "_total_gaps", 0) + 1
                pts = [s for s in self._active if s.detected]
                if pts and self._in_approach_zone(pts[-1].x_proc):
                    end = self._check_end_while_recording()
                    if end == PitchEndReason.TRACK_LOST_TERMINAL:
                        completed = self._finalize(end)
                        discarded = completed is None
                        return PitchUpdateResult(
                            mode=self.mode,
                            pitch_completed=completed,
                            pitch_discarded=discarded,
                        )
                if self._gap_frames > self.config.max_gap_frames:
                    pass  # stay recording; validation may reject later
```

Add stub `_finalize` returning None for now (Task 5 implements validation).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pitch_state.py::test_end_on_plate_passed tests/test_pitch_state.py::test_gap_tolerance_does_not_end_pitch -v`  
Expected: plate pass may fail until `_finalize` stub returns a Pitch — adjust stub to return minimal Pitch for test, or implement Task 5 first. **Implement minimal `_finalize` that always accepts PLATE_PASSED** as interim:

```python
    def _finalize(self, end_reason: PitchEndReason) -> Pitch | None:
        pts = [s for s in self._active if s.detected]
        pitch = Pitch(
            pitch_id=str(uuid.uuid4()),
            start_frame=self._recording_start_frame,
            end_frame=pts[-1].frame_index if pts else self._recording_start_frame,
            started_at_sec=self._recording_started_at or 0.0,
            ended_at_sec=pts[-1].timestamp_sec if pts else 0.0,
            start_reason=self._start_reason,
            end_reason=end_reason,
            samples=list(self._active),
            backfill_frames=max(0, len(self._active) - 1),
            gap_count=self._total_gaps,
            reconnect_count=self._reconnects,
        )
        self._last_completed = pitch
        self.mode = PitchMode.IDLE
        self._active.clear()
        self._cooldown = self.config.cooldown_frames
        return pitch
```

- [ ] **Step 5: Run all pitch_state tests**

Run: `python -m pytest tests/test_pitch_state.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pitch_state.py tests/test_pitch_state.py
git commit -m "feat: pitch end detection with gap tolerance"
```

---

### Task 5: Pitch validation in FINALIZING

**Files:**
- Modify: `pitch_state.py`
- Modify: `tests/test_pitch_state.py`

- [ ] **Step 1: Write failing validation tests**

Append to `tests/test_pitch_state.py`:

```python
def test_discard_false_start_with_no_forward_progress():
    sm, geom = _recording_sm()
    # Only samples near release — plate pass never reached; force timeout via max frames
    sm.config.max_pitch_frames = 3
    sm.update(PitchFrameInput(3, 0.1, geom.release_cx_proc + 2, geom.release_cy_proc, detected=True))
    result = sm.update(PitchFrameInput(4, 0.13, geom.release_cx_proc + 3, geom.release_cy_proc, detected=True))
    assert result.pitch_discarded is True
    assert result.pitch_completed is None


def test_track_lost_terminal_saves_incomplete():
    sm, geom = _recording_sm()
    x = geom.strike_leading_x_proc
    for fi in range(3, 10):
        sm.update(PitchFrameInput(fi, 0.05 * fi, x + fi * 3, geom.release_cy_proc, detected=True))
    sm.config.terminal_lost_frames = 2
    sm.update(PitchFrameInput(20, 1.0, detected=False))
    result = sm.update(PitchFrameInput(21, 1.1, detected=False))
    assert result.pitch_completed is not None
    assert result.pitch_completed.end_reason == PitchEndReason.TRACK_LOST_TERMINAL
    assert result.pitch_completed.complete is False
```

- [ ] **Step 2: Run tests to verify failure modes**

Run: `python -m pytest tests/test_pitch_state.py::test_discard_false_start_with_no_forward_progress tests/test_pitch_state.py::test_track_lost_terminal_saves_incomplete -v`  
Expected: FAIL until validation added

- [ ] **Step 3: Replace `_finalize` with validation**

Replace `_finalize` in `pitch_state.py`:

```python
    def _validate(self, end_reason: PitchEndReason) -> tuple[bool, bool]:
        """Return (valid, complete)."""
        cfg = self.config
        pts = [s for s in self._active if s.detected]
        if len(pts) < cfg.min_detected_samples:
            return False, False
        forward = pts[-1].x_proc - pts[0].x_proc
        if forward < cfg.min_forward_px:
            return False, False
        near_release = any(s.in_release_zone for s in pts) or any(
            math.hypot(s.x_proc - self.geom.release_cx_proc, s.y_proc - self.geom.release_cy_proc)
            <= self.geom.release_r_proc * 2
            for s in pts[:3]
        )
        if not near_release:
            return False, False
        if self._total_gaps > cfg.max_total_gap_frames:
            return False, False
        if self._reconnects > cfg.max_reconnects:
            return False, False
        for i in range(1, len(pts)):
            step = math.hypot(pts[i].x_proc - pts[i - 1].x_proc, pts[i].y_proc - pts[i - 1].y_proc)
            if step > cfg.max_step_px:
                return False, False
        complete = end_reason in (
            PitchEndReason.STRIKE_CONTACT,
            PitchEndReason.HIT_OR_DEFLECTION,
            PitchEndReason.PLATE_PASSED,
        )
        if end_reason == PitchEndReason.TRACK_LOST_TERMINAL:
            complete = False
        if end_reason in (PitchEndReason.TIMEOUT, PitchEndReason.INVALID_REACQUIRE):
            return False, False
        return True, complete

    def _finalize(self, end_reason: PitchEndReason) -> Pitch | None:
        valid, complete = self._validate(end_reason)
        pts = [s for s in self._active if s.detected]
        if not valid:
            self.mode = PitchMode.IDLE
            self._active.clear()
            self._cooldown = self.config.cooldown_frames
            return None
        pitch = Pitch(
            pitch_id=str(uuid.uuid4()),
            start_frame=self._recording_start_frame,
            end_frame=pts[-1].frame_index,
            started_at_sec=self._recording_started_at or 0.0,
            ended_at_sec=pts[-1].timestamp_sec,
            start_reason=self._start_reason,
            end_reason=end_reason,
            samples=list(self._active),
            backfill_frames=min(len(self._active), self.config.ring_buffer_frames),
            gap_count=self._total_gaps,
            reconnect_count=self._reconnects,
            valid=True,
            complete=complete,
        )
        self._last_completed = pitch
        self.mode = PitchMode.IDLE
        self._active.clear()
        self._cooldown = self.config.cooldown_frames
        return pitch
```

Update `_finalize` return sites to set `pitch_discarded=True` when `_finalize` returns `None`.

- [ ] **Step 4: Run all pitch_state tests**

Run: `python -m pytest tests/test_pitch_state.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pitch_state.py tests/test_pitch_state.py
git commit -m "feat: pitch validation on finalize with incomplete save support"
```

---

### Task 6: Live tracer + fading overlay

**Files:**
- Create: `pitch_tracer.py`
- Create: `tests/test_pitch_tracer.py`

- [ ] **Step 1: Write failing tracer tests**

Create `tests/test_pitch_tracer.py`:

```python
import numpy as np

from pitch_tracer import PitchTracer, tracer_alpha


def test_tracer_alpha_fades_to_zero():
    assert tracer_alpha(created_at=0.0, now=4.0, fade_sec=4.0) == 0.0
    assert tracer_alpha(created_at=0.0, now=0.0, fade_sec=4.0) == 1.0
    assert 0.0 < tracer_alpha(created_at=0.0, now=2.0, fade_sec=4.0) < 1.0


def test_live_points_cleared_on_finalize():
    tracer = PitchTracer(fade_sec=4.0)
    tracer.add_point(10, 20)
    tracer.add_point(30, 40)
    assert len(tracer.live_points_full) == 2
    tracer.finalize_path(now=1.0, valid=True)
    assert tracer.live_points_full == []
    assert len(tracer.fading_paths) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pitch_tracer.py -v`  
Expected: FAIL — module not found

- [ ] **Step 3: Implement `pitch_tracer.py`**

Create `pitch_tracer.py`:

```python
"""Visual flight-path tracer — live during recording, fading after pitch ends."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def tracer_alpha(*, created_at: float, now: float, fade_sec: float) -> float:
    if fade_sec <= 0:
        return 0.0
    elapsed = now - created_at
    if elapsed >= fade_sec:
        return 0.0
    return max(0.0, 1.0 - elapsed / fade_sec)


@dataclass
class FadingPath:
    points: list[tuple[int, int]]
    created_at: float
    valid: bool = True


@dataclass
class PitchTracer:
    fade_sec: float = 4.0
    live_points_full: list[tuple[int, int]] = field(default_factory=list)
    fading_paths: list[FadingPath] = field(default_factory=list)

    def reset(self) -> None:
        self.live_points_full.clear()
        self.fading_paths.clear()

    def add_point(self, x_full: int, y_full: int) -> None:
        pt = (x_full, y_full)
        if self.live_points_full and self.live_points_full[-1] == pt:
            return
        self.live_points_full.append(pt)

    def sync_from_samples(self, samples: list) -> None:
        """Rebuild live path from pitch samples (full-frame coords)."""
        self.live_points_full.clear()
        for s in samples:
            if s.detected:
                self.add_point(s.x_full, s.y_full)

    def finalize_path(self, *, now: float, valid: bool) -> None:
        if len(self.live_points_full) >= 2:
            self.fading_paths.append(
                FadingPath(points=list(self.live_points_full), created_at=now, valid=valid)
            )
        self.live_points_full.clear()

    def abort_live(self) -> None:
        self.live_points_full.clear()

    def prune_expired(self, now: float) -> None:
        self.fading_paths = [
            p
            for p in self.fading_paths
            if tracer_alpha(created_at=p.created_at, now=now, fade_sec=self.fade_sec) > 0
            and len(p.points) >= 2
        ]

    def draw(self, frame: np.ndarray, now: float) -> None:
        self.prune_expired(now)

        # Fading paths (older first, live on top)
        for path in self.fading_paths:
            alpha = tracer_alpha(created_at=path.created_at, now=now, fade_sec=self.fade_sec)
            color = (255, 200, 0) if path.valid else (160, 160, 160)
            _draw_polyline(frame, path.points, color, alpha=alpha)

        # Live recording path — full brightness cyan
        if len(self.live_points_full) >= 2:
            _draw_polyline(frame, self.live_points_full, (255, 255, 0), alpha=1.0)
        for pt in self.live_points_full:
            cv2.circle(frame, pt, 3, (255, 255, 0), -1, lineType=cv2.LINE_AA)


def _draw_polyline(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    color_bgr: tuple[int, int, int],
    *,
    alpha: float,
) -> None:
    if alpha <= 0 or len(points) < 2:
        return
    overlay = frame.copy()
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    thickness = max(1, int(2 * alpha + 0.5))
    cv2.polylines(overlay, [pts], False, color_bgr, thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pitch_tracer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pitch_tracer.py tests/test_pitch_tracer.py
git commit -m "feat: live pitch tracer with fading overlay"
```

---

### Task 7: Wire pitch machine + tracer into `track_ball.py`

**Files:**
- Modify: `track_ball.py`

- [ ] **Step 1: Add imports and remove old pitch_active constants usage**

At top of `track_ball.py`, add:

```python
from pitch_geometry import build_field_geometry
from pitch_state import PitchFrameInput, PitchMode, PitchStateMachine
from pitch_tracer import PitchTracer
```

Remove or stop using `PITCH_IDLE_FRAMES` for pitch semantics (keep constant only if unused, then delete).

- [ ] **Step 2: Initialize pitch machine after calibration resolves**

Inside `main()`, after `cal` is resolved and frame dimensions are known:

```python
    pitch_machine: PitchStateMachine | None = None
    pitch_tracer: PitchTracer | None = None
    field_geom = None
```

After first frame read (when `frame_w`, `frame_h` known) or lazily on first loop iteration:

```python
        if cal is not None and pitch_machine is None:
            field_geom = build_field_geometry(
                cal,
                frame_width=frame_w,
                frame_height=frame_h,
                process_scale=process_scale,
            )
            pitch_machine = PitchStateMachine(field_geom)
            pitch_tracer = PitchTracer(fade_sec=pitch_machine.config.tracer_fade_sec)
```

- [ ] **Step 3: Replace pitch_active block with pitch update**

Replace the block at lines ~805–822:

```python
        pitch_recording = False
        now = time.perf_counter()

        if pitch_machine is not None and pitch_tracer is not None and result is not None:
            proc_pt = result.density_peak_proc
            full_pt = result.density_peak
            detected = proc_pt is not None and not result.rejected_by_ignore
            cls = None
            if detected and cal is not None and full_pt is not None:
                cls = classify_point(
                    full_pt,
                    cal,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    ignore_check=False,
                )
            inp = PitchFrameInput(
                frame_index=frame_count,
                timestamp_sec=elapsed,
                x_proc=proc_pt[0] if proc_pt else None,
                y_proc=proc_pt[1] if proc_pt else None,
                detected=detected,
                track_mode=TrackMode.TRACK if result.track_mode == "track" else TrackMode.SEARCH,
                coasting=result.coasting,
                in_roi=cls.in_roi if cls else False,
                in_release_zone=cls.in_release_zone if cls else False,
                in_strike_zone=cls.in_strike_zone if cls else False,
            )
            update_result = pitch_machine.update(inp)
            pitch_recording = pitch_machine.mode == PitchMode.RECORDING

            if pitch_recording and proc_pt is not None:
                xf, yf = field_geom.proc_to_full(proc_pt[0], proc_pt[1])
                pitch_tracer.add_point(xf, yf)
            elif pitch_machine.mode == PitchMode.IDLE and pitch_tracer.live_points_full:
                # sync live path from active samples if we just finished this frame
                pitch_tracer.sync_from_samples(pitch_machine.active_samples)

            if update_result.pitch_completed is not None:
                pitch = update_result.pitch_completed
                pitch_tracer.finalize_path(now=now, valid=pitch.valid)
                print(
                    f"Pitch saved id={pitch.pitch_id[:8]} "
                    f"samples={len(pitch.samples)} "
                    f"complete={pitch.complete} "
                    f"end={pitch.end_reason.value}"
                )
            elif update_result.pitch_discarded:
                pitch_tracer.finalize_path(now=now, valid=False)
                print("Pitch discarded")
```

Note: when entering RECORDING mid-frame, also sync tracer from `pitch_machine.active_samples` so backfilled points appear immediately:

```python
            if pitch_recording:
                pitch_tracer.sync_from_samples(pitch_machine.active_samples)
```

- [ ] **Step 4: Update overlay drawing order**

After `draw_calibration_overlay`, before `draw_track_dot`:

```python
        if pitch_tracer is not None:
            pitch_tracer.draw(frame, now)
        draw_track_dot(frame, detection, pitch_active=pitch_recording)
```

Update status line: replace `pitch_active` with `pitch_recording`; add pitch mode label:

```python
        if pitch_recording:
            status += "  PITCH"
        elif pitch_machine is not None and pitch_machine._cooldown > 0:
            status += "  cooldown"
        else:
            status += "  idle"
```

- [ ] **Step 5: Reset on `b` key**

In `b` handler:

```python
        if key == ord("b"):
            tracker.reset()
            if pitch_machine is not None:
                pitch_machine.reset()
            if pitch_tracer is not None:
                pitch_tracer.abort_live()
```

- [ ] **Step 6: Manual smoke test**

Run: `python track_ball.py --source <recorded_clip.mp4> --calibration configs/field.json`  
Expected:
- Cyan polyline grows during pitch flight
- Red dot while RECORDING
- After pitch ends, cyan path fades over ~4 s
- Console prints pitch saved/discarded line

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest tests/ -v`  
Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add track_ball.py
git commit -m "feat: integrate pitch recording, validation, and fading tracer"
```

---

### Task 8: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add pitch detection section**

Append to README after tracking section:

```markdown
## Pitch detection

When calibration is loaded, the tracker runs a pitch state machine alongside Search/Track mode:

- **Recording** starts within 1–2 frames of release-like motion in the release zone (strong rightward vs vertical motion).
- A **cyan flight path** is drawn live during recording (confirms data is being collected).
- When the pitch ends (impact, plate crossing, or track loss near the zone), the path **fades out over ~4 seconds** so the next pitch has a clean view.
- Pitch analytics (velocity, break, fit) are not run live; completed pitches are validated and logged to the console in v1.

Status line: `PITCH` while recording, `idle` otherwise.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: pitch detection and fading tracer"
```

---

## Spec Coverage Checklist

| Spec requirement | Task |
|------------------|------|
| Fast release-based start + ring buffer | Task 3 |
| Impact / plate pass / terminal loss end | Task 4 |
| Gap tolerance + reconnect | Task 4 |
| Validation accept / incomplete / discard | Task 5 |
| Live tracer during RECORDING | Task 6, 7 |
| Fade after IDLE | Task 6, 7 |
| Orthogonal to TrackState | All (no BallTracker changes) |
| Replace pitch_active | Task 7 |
| Unit tests without video | Tasks 1–6 |

## Manual Test Checklist

- [ ] Live GoPro: cyan tracer follows ball; fades after pitch
- [ ] Recorded file replay: same behavior deterministically
- [ ] Windup in release zone does not start recording (vertical motion)
- [ ] High/low miss still ends pitch (plate pass or terminal loss)
- [ ] `b` reset clears in-progress recording and live tracer
- [ ] No calibration → no pitch machine; yellow dot only (Choice B)

---

**Plan complete.** Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — implement task-by-task in this session with checkpoints

Which approach do you want?
