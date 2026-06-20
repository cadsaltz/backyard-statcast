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
