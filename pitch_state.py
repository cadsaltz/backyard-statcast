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
        self._gap_frames = 0
        self._implausible_streak = 0
        self._total_gaps = 0
        self._reconnects = 0
        self._impact_streak = 0
        self._start_reason: PitchStartReason = PitchStartReason.RELEASE_MOTION

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
        self._gap_frames = 0
        self._implausible_streak = 0
        self._total_gaps = 0
        self._reconnects = 0
        self._impact_streak = 0

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

        if len(self._active) > cfg.max_pitch_frames:
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

            self._was_in_release = in_release
            if sample is not None:
                self._prev = sample
        else:
            if self.mode == PitchMode.RECORDING:
                self._gap_frames += 1
                self._total_gaps += 1
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

        return PitchUpdateResult(mode=self.mode, pitch_completed=completed, pitch_discarded=discarded)
