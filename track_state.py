"""Search / Track state machine for frame-to-frame ball continuity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class TrackMode(Enum):
    SEARCH = "search"
    TRACK = "track"


@dataclass
class TrackConfig:
    track_density_radius: int = 6
    track_lost_frames: int = 3
    slow_track_lost_frames: int = 8
    coast_frames: int = 2
    max_jump_px: int = 80
    margin: int = 24
    min_window_pad: int = 32
    max_window_pad: int = 48
    coast_pad_boost: int = 8
    slow_speed_px: float = 5.0
    track_min_density_ratio: float = 0.25
    min_lock_area: float = 12.0
    min_track_area: float = 10.0
    min_lock_density_score: float = 20.0
    roi_exit_frames: int = 8
    stuck_frames: int = 15
    stuck_max_area: float = 8.0
    stuck_max_speed_px: float = 2.0
    track_white_relax: int = 20
    track_sat_relax: int = 25
    track_diff_relax: int = 5
    track_yellow_val_relax: int = 20
    track_yellow_sat_relax: int = 15
    search_lock_streak: int = 2
    search_relock_streak: int = 1
    search_prior_pad: int = 96


@dataclass
class TrackFrameInfo:
    mode: TrackMode
    coasting: bool
    slow_motion: bool
    search_window_proc: tuple[int, int, int, int] | None
    predicted_proc: tuple[int, int] | None
    last_proc: tuple[int, int] | None
    prev_proc: tuple[int, int] | None
    extra_pad: int


def clamp_rect(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x0 = max(0, min(x0, frame_w))
    x1 = max(0, min(x1, frame_w))
    y0 = max(0, min(y0, frame_h))
    y1 = max(0, min(y1, frame_h))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def search_window_rect(
    last: tuple[int, int],
    predicted: tuple[int, int] | None,
    *,
    min_pad: int,
    margin: int,
    slow: bool,
    extra_pad: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Rectangle spanning last → predicted, always covering last with padding."""
    lx, ly = last
    pad = min_pad + extra_pad
    if slow or predicted is None:
        x0, y0 = lx - pad, ly - pad
        x1, y1 = lx + pad, ly + pad
    else:
        px, py = predicted
        pad_pred = pad + margin
        x0 = min(lx, px) - pad
        y0 = min(ly, py) - pad
        x1 = max(lx, px) + pad_pred
        y1 = max(ly, py) + pad_pred

    min_span = min_pad * 2
    if x1 - x0 < min_span:
        cx = (x0 + x1) // 2
        x0, x1 = cx - min_pad, cx + min_pad
    if y1 - y0 < min_span:
        cy = (y0 + y1) // 2
        y0, y1 = cy - min_pad, cy + min_pad

    return clamp_rect(x0, y0, x1, y1, frame_w, frame_h)


def padded_point_rect(
    x: int,
    y: int,
    pad: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    return clamp_rect(x - pad, y - pad, x + pad, y + pad, frame_w, frame_h)


class TrackState:
    def __init__(self, config: TrackConfig | None = None) -> None:
        self.config = config or TrackConfig()
        self._roi_gating = False
        self.reset()

    def reset(self) -> None:
        self.mode = TrackMode.SEARCH
        self._history: list[tuple[int, int]] = []
        self._miss_count = 0
        self._search_streak = 0
        self._search_lock_density: float | None = None
        self._frames_in_track = 0
        self._search_prior: tuple[int, int] | None = None
        self._outside_roi_frames = 0
        self._stuck_frames = 0

    def set_roi_gating(self, enabled: bool) -> None:
        self._roi_gating = enabled

    @property
    def search_prior(self) -> tuple[int, int] | None:
        return self._search_prior

    @property
    def is_coasting(self) -> bool:
        return (
            self.mode == TrackMode.TRACK
            and 0 < self._miss_count <= self.config.coast_frames
        )

    def velocity(self) -> tuple[int, int] | None:
        if len(self._history) < 2:
            return None
        lx, ly = self._history[-1]
        px, py = self._history[-2]
        return lx - px, ly - py

    def speed_px(self) -> float:
        vel = self.velocity()
        if vel is None:
            return 0.0
        return math.hypot(vel[0], vel[1])

    def is_slow_motion(self) -> bool:
        return self.speed_px() < self.config.slow_speed_px

    def predicted(self) -> tuple[int, int] | None:
        if not self._history:
            return None
        last = self._history[-1]
        if self.is_slow_motion():
            return last
        vel = self.velocity()
        if vel is None:
            return last
        return last[0] + vel[0], last[1] + vel[1]

    def last_position(self) -> tuple[int, int] | None:
        return self._history[-1] if self._history else None

    def prev_position(self) -> tuple[int, int] | None:
        if len(self._history) < 2:
            return None
        return self._history[-2]

    def extra_window_pad(self) -> int:
        return self._miss_count * self.config.coast_pad_boost

    def search_window(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        last = self.last_position()
        if last is None:
            return 0, 0, frame_w, frame_h
        cfg = self.config
        return search_window_rect(
            last,
            self.predicted(),
            min_pad=cfg.min_window_pad,
            margin=cfg.margin,
            slow=self.is_slow_motion(),
            extra_pad=self.extra_window_pad(),
            frame_w=frame_w,
            frame_h=frame_h,
        )

    def prior_search_window(
        self, frame_w: int, frame_h: int
    ) -> tuple[int, int, int, int] | None:
        if self._search_prior is None:
            return None
        px, py = self._search_prior
        return padded_point_rect(
            px, py, self.config.search_prior_pad, frame_w, frame_h
        )

    def frame_info(self, frame_w: int, frame_h: int) -> TrackFrameInfo:
        window = None
        if self.mode == TrackMode.TRACK:
            window = self.search_window(frame_w, frame_h)
        elif self._search_prior is not None:
            window = self.prior_search_window(frame_w, frame_h)
        slow = self.is_slow_motion() if self.mode == TrackMode.TRACK else False
        return TrackFrameInfo(
            mode=self.mode,
            coasting=self.is_coasting,
            slow_motion=slow,
            search_window_proc=window,
            predicted_proc=self.predicted() if self.mode == TrackMode.TRACK else None,
            last_proc=self.last_position(),
            prev_proc=self.prev_position(),
            extra_pad=self.extra_window_pad() if self.mode == TrackMode.TRACK else 0,
        )

    def _trim_history(self) -> None:
        if len(self._history) > 2:
            self._history = self._history[-2:]

    def _lock_streak_required(self) -> int:
        if self._search_prior is not None:
            return self.config.search_relock_streak
        return self.config.search_lock_streak

    def qualify_search_lock(
        self,
        *,
        density_score: float,
        blob_area: float,
        in_roi: bool,
        has_motion: bool,
    ) -> bool:
        cfg = self.config
        if self._roi_gating and not in_roi:
            return False
        if not has_motion:
            return False
        if blob_area < cfg.min_lock_area:
            return False
        if density_score < cfg.min_lock_density_score:
            return False
        return True

    def register_search_hit(
        self,
        x: int,
        y: int,
        density_score: float,
        blob_area: float,
        *,
        in_roi: bool,
        has_motion: bool,
    ) -> bool:
        if self.qualify_search_lock(
            density_score=density_score,
            blob_area=blob_area,
            in_roi=in_roi,
            has_motion=has_motion,
        ):
            self.accept_search_peak(x, y, density_score)
            return True
        self.search_miss()
        return False

    def accept_search_peak(self, x: int, y: int, score: float) -> None:
        self._search_streak += 1
        self._history.append((x, y))
        self._trim_history()
        if self._search_streak >= self._lock_streak_required():
            self.mode = TrackMode.TRACK
            self._search_lock_density = score
            self._miss_count = 0
            self._frames_in_track = 0
            self._search_prior = None
            self._outside_roi_frames = 0
            self._stuck_frames = 0

    def search_miss(self) -> None:
        self._search_streak = 0

    def qualify_track_peak(
        self,
        x: int,
        y: int,
        density_score: float,
        blob_area: float,
        *,
        in_roi: bool,
    ) -> bool:
        cfg = self.config
        if self._roi_gating and not in_roi:
            return False
        if blob_area < cfg.min_track_area:
            return False
        last = self.last_position()
        pred = self.predicted()
        if last is not None:
            near_last = math.hypot(x - last[0], y - last[1]) <= cfg.max_jump_px
            near_pred = (
                pred is not None
                and math.hypot(x - pred[0], y - pred[1]) <= cfg.max_jump_px
            )
            if not near_last and not near_pred:
                return False
        if (
            not self.is_slow_motion()
            and self._frames_in_track > 0
            and self._search_lock_density is not None
        ):
            min_score = cfg.track_min_density_ratio * self._search_lock_density
            if density_score < min_score:
                return False
        return True

    def check_track_peak(self, x: int, y: int, score: float) -> bool:
        return self.qualify_track_peak(
            x, y, score, score, in_roi=True
        )

    def accept_track_peak(self, x: int, y: int, score: float) -> None:
        self._history.append((x, y))
        self._trim_history()
        self._miss_count = 0
        self._frames_in_track += 1

    def update_track_health(self, *, in_roi: bool, blob_area: float) -> bool:
        """Return False if Track was forced back to Search."""
        if self.mode != TrackMode.TRACK:
            return True
        cfg = self.config
        if self._roi_gating and not in_roi:
            self._outside_roi_frames += 1
            if self._outside_roi_frames >= cfg.roi_exit_frames:
                self.force_search(clear_prior=True)
                return False
        else:
            self._outside_roi_frames = 0

        if blob_area <= cfg.stuck_max_area and self.speed_px() < cfg.stuck_max_speed_px:
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0
        if self._stuck_frames >= cfg.stuck_frames:
            self.force_search(clear_prior=True)
            return False
        return True

    def _lost_frames_threshold(self) -> int:
        if self.is_slow_motion():
            return self.config.slow_track_lost_frames
        return self.config.track_lost_frames

    def force_search(self, *, clear_prior: bool) -> None:
        if clear_prior:
            self._search_prior = None
            self._history = []
        else:
            prior = self.last_position()
            self._search_prior = prior
            self._history = [prior] if prior is not None else []
        self.mode = TrackMode.SEARCH
        self._miss_count = 0
        self._search_streak = 0
        self._frames_in_track = 0
        self._search_lock_density = None
        self._outside_roi_frames = 0
        self._stuck_frames = 0

    def _fallback_to_search(self) -> None:
        self.force_search(clear_prior=False)

    def track_miss(self) -> tuple[int, int] | None:
        self._miss_count += 1
        self._frames_in_track += 1
        if self._miss_count >= self._lost_frames_threshold():
            self._fallback_to_search()
            return None
        if self._miss_count <= self.config.coast_frames and self._history:
            return self._history[-1]
        return None

    def status_label(self) -> str:
        if self.is_coasting:
            return "COAST"
        if self.mode == TrackMode.TRACK and self.is_slow_motion():
            return "TRACK-SLOW"
        return self.mode.name
