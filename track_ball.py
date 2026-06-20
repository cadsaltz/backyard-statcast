"""Ball tracker: bg subtract + color filter + density peak (red dot).

Performance: process at reduced resolution, display at capture resolution.
Debug panels open in separate windows (toggle with d).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from calibration import (
    DEFAULT_PRESET_PATH,
    FieldCalibration,
    load_calibration,
    save_calibration,
)
from calibration_ui import run_calibration_ui
from frame_source import open_frame_source
from spatial_filter import (
    classify_point,
    ignore_mask_for_process,
    scale_point_to_pixels,
    scaled_polygons,
)
from pitch_geometry import build_field_geometry
from pitch_state import PitchFrameInput, PitchMode, PitchStateMachine
from pitch_tracer import PitchTracer
from track_state import TrackFrameInfo, TrackMode, TrackState

WIN_LIVE = "Ball Tracker"
WIN_FG = "Foreground"
WIN_COLOR = "Color Filter"
WIN_TRACK = "Track Window"

ColorTarget = Literal["white", "yellow"]

RESOLUTION_PRESETS = {
    "1080": {"width": 1920, "height": 1080, "process_scale": 0.5, "live_win": (960, 540)},
    "720": {"width": 1280, "height": 720, "process_scale": 1.0, "live_win": (640, 360)},
}

@dataclass
class PeakCandidate:
    x: int
    y: int
    density_score: float
    blob_area: float


@dataclass
class TrackResult:
    density_peak: tuple[int, int] | None  # full-frame coords
    density_peak_proc: tuple[int, int] | None = None  # process-resolution coords
    foreground: np.ndarray | None = None
    color_fg: np.ndarray | None = None
    rejected_by_ignore: bool = False
    raw_peak_proc: tuple[int, int] | None = None
    track_mode: Literal["search", "track"] = "search"
    coasting: bool = False
    search_window_proc: tuple[int, int, int, int] | None = None
    predicted_proc: tuple[int, int] | None = None
    track_debug_proc: np.ndarray | None = None


class BallTracker:
    def __init__(
        self,
        process_scale: float = 0.5,
        warmup_frames: int = 30,
        diff_threshold: int = 25,
        white_threshold: int = 140,
        max_saturation: int = 100,
        density_radius: int = 12,
        color_target: ColorTarget = "white",
        yellow_hue_low: int = 18,
        yellow_hue_high: int = 38,
        yellow_min_sat: int = 80,
        yellow_min_val: int = 140,
        ignore_mask_proc: np.ndarray | None = None,
    ) -> None:
        self.process_scale = process_scale
        self.warmup_frames = warmup_frames
        self.diff_threshold = diff_threshold
        self.white_threshold = white_threshold
        self.max_saturation = max_saturation
        self.density_radius = density_radius
        self.color_target = color_target
        self.yellow_hue_low = yellow_hue_low
        self.yellow_hue_high = yellow_hue_high
        self.yellow_min_sat = yellow_min_sat
        self.yellow_min_val = yellow_min_val
        self.ignore_mask_proc = ignore_mask_proc
        self._roi_proc: np.ndarray | None = None

        self._bg: np.ndarray | None = None
        self._warmup_stack: list[np.ndarray] = []
        self._noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._density_ksize = density_radius * 2 + 1
        self._proc_size: tuple[int, int] | None = None  # (w, h)
        self._track = TrackState()

    @property
    def is_ready(self) -> bool:
        return self._bg is not None

    @property
    def warmup_progress(self) -> tuple[int, int]:
        n = min(len(self._warmup_stack), self.warmup_frames)
        return n, self.warmup_frames

    def reset(self) -> None:
        self._bg = None
        self._warmup_stack = []
        self._proc_size = None
        self._track.reset()

    def set_ignore_mask_proc(self, mask: np.ndarray | None) -> None:
        self.ignore_mask_proc = mask

    def set_roi_proc(self, polygon: list[tuple[int, int]] | None) -> None:
        if polygon:
            self._roi_proc = np.array(polygon, dtype=np.int32)
            self._track.set_roi_gating(True)
        else:
            self._roi_proc = None
            self._track.set_roi_gating(False)

    def _in_roi_proc(self, px: int, py: int) -> bool:
        if self._roi_proc is None:
            return True
        return (
            cv2.pointPolygonTest(self._roi_proc, (float(px), float(py)), False) >= 0
        )

    def _to_process(self, frame: np.ndarray) -> np.ndarray:
        if self.process_scale == 1.0:
            return frame
        return cv2.resize(
            frame,
            None,
            fx=self.process_scale,
            fy=self.process_scale,
            interpolation=cv2.INTER_AREA,
        )

    def _to_full(self, x: int, y: int) -> tuple[int, int]:
        if self.process_scale == 1.0:
            return x, y
        inv = 1.0 / self.process_scale
        return int(x * inv), int(y * inv)

    def _learn_background(self) -> None:
        stack = np.stack(self._warmup_stack, axis=0)
        self._bg = np.median(stack, axis=0).astype(np.uint8)
        self._warmup_stack.clear()
        h, w = self._bg.shape[:2]
        self._proc_size = (w, h)

    def _density_peak(
        self, color_mask: np.ndarray, radius: int | None = None
    ) -> tuple[int, int, float] | None:
        if not cv2.countNonZero(color_mask):
            return None
        r = self.density_radius if radius is None else radius
        ksize = r * 2 + 1
        density = cv2.boxFilter(
            color_mask, cv2.CV_32F, (ksize, ksize), normalize=False
        )
        min_val, max_val, _, max_loc = cv2.minMaxLoc(density)
        if max_val <= 0:
            return None
        return int(max_loc[0]), int(max_loc[1]), float(max_val)

    def _density_at(
        self, mask: np.ndarray, x: int, y: int, radius: int
    ) -> float:
        h, w = mask.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        ksize = radius * 2 + 1
        density = cv2.boxFilter(
            mask, cv2.CV_32F, (ksize, ksize), normalize=False
        )
        return float(density[y, x])

    def _track_peak(
        self,
        track_mask: np.ndarray,
        radius: int,
        anchor_proc: tuple[int, int] | None,
        window_x0: int,
        window_y0: int,
        *,
        min_contour_area: float = 1.0,
    ) -> PeakCandidate | None:
        if not cv2.countNonZero(track_mask):
            return None

        if anchor_proc is not None:
            ax = anchor_proc[0] - window_x0
            ay = anchor_proc[1] - window_y0
            contours, _ = cv2.findContours(
                track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            best: PeakCandidate | None = None
            best_dist = float("inf")
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_contour_area:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] <= 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                dist = (cx - ax) ** 2 + (cy - ay) ** 2
                if dist < best_dist:
                    best_dist = dist
                    score = self._density_at(track_mask, cx, cy, radius)
                    best = PeakCandidate(cx, cy, score, float(area))
            if best is not None:
                return best

        peak = self._density_peak(track_mask, radius)
        if peak is None:
            return None
        lx, ly, score = peak
        area = float(cv2.countNonZero(track_mask))
        return PeakCandidate(lx, ly, score, area)

    def _find_peak_in_mask(
        self,
        mask: np.ndarray,
        *,
        radius: int,
        anchor_proc: tuple[int, int] | None,
        offset_x: int = 0,
        offset_y: int = 0,
        min_contour_area: float = 1.0,
    ) -> PeakCandidate | None:
        peak = self._track_peak(
            mask,
            radius,
            anchor_proc,
            offset_x,
            offset_y,
            min_contour_area=min_contour_area,
        )
        if peak is None:
            return None
        return PeakCandidate(
            peak.x + offset_x,
            peak.y + offset_y,
            peak.density_score,
            peak.blob_area,
        )

    def _blob_area_at(
        self,
        mask: np.ndarray,
        px: int,
        py: int,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> float:
        lx, ly = px - offset_x, py - offset_y
        h, w = mask.shape[:2]
        if not (0 <= lx < w and 0 <= ly < h):
            return 0.0
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if cv2.pointPolygonTest(contour, (float(lx), float(ly)), False) >= 0:
                return float(cv2.contourArea(contour))
        return 0.0

    def _color_mask(self, hsv: np.ndarray, *, relax: bool = False) -> np.ndarray:
        cfg = self._track.config
        if self.color_target == "yellow":
            min_sat = self.yellow_min_sat
            min_val = self.yellow_min_val
            if relax:
                min_sat = max(0, min_sat - cfg.track_yellow_sat_relax)
                min_val = max(0, min_val - cfg.track_yellow_val_relax)
            return cv2.inRange(
                hsv,
                (self.yellow_hue_low, min_sat, min_val),
                (self.yellow_hue_high, 255, 255),
            )
        white = self.white_threshold
        max_sat = self.max_saturation
        if relax:
            white = max(0, white - cfg.track_white_relax)
            max_sat = min(255, max_sat + cfg.track_sat_relax)
        return cv2.inRange(
            hsv,
            (0, 0, white),
            (180, max_sat, 255),
        )

    def _peak_rejected_by_ignore(self, px: int, py: int) -> bool:
        if self.ignore_mask_proc is None:
            return False
        h, w = self.ignore_mask_proc.shape[:2]
        return 0 <= px < w and 0 <= py < h and self.ignore_mask_proc[py, px] > 0

    def toggle_color_target(self) -> ColorTarget:
        self.color_target = "yellow" if self.color_target == "white" else "white"
        return self.color_target

    def _search_candidate(
        self,
        color_mask: np.ndarray,
        track: TrackState,
        proc_w: int,
        proc_h: int,
    ) -> PeakCandidate | None:
        prior = track.search_prior
        if prior is not None:
            prior_win = track.prior_search_window(proc_w, proc_h)
            if prior_win is not None:
                px0, py0, px1, py1 = prior_win
                prior_crop = color_mask[py0:py1, px0:px1]
                candidate = self._find_peak_in_mask(
                    prior_crop,
                    radius=self.density_radius,
                    anchor_proc=prior,
                    offset_x=px0,
                    offset_y=py0,
                    min_contour_area=track.config.min_lock_area,
                )
                if candidate is not None:
                    return candidate
        peak = self._density_peak(color_mask)
        if peak is None:
            return None
        px, py, score = peak
        blob_area = self._blob_area_at(color_mask, px, py)
        if blob_area < track.config.min_lock_area:
            blob_area = float(score)
        return PeakCandidate(px, py, score, blob_area)

    def process(self, frame: np.ndarray, *, debug: bool = False) -> TrackResult | None:
        small = self._to_process(frame)

        if self._bg is None:
            self._warmup_stack.append(small.copy())
            if len(self._warmup_stack) >= self.warmup_frames:
                self._learn_background()
            return None

        diff = cv2.absdiff(small, self._bg)
        diff_max = np.max(diff, axis=2)
        _, fg_mask = cv2.threshold(
            diff_max, self.diff_threshold, 255, cv2.THRESH_BINARY
        )
        fg_mask = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_OPEN, self._noise_kernel)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        color_mask = self._color_mask(hsv)
        color_mask = cv2.bitwise_and(color_mask, fg_mask)

        proc_w, proc_h = self._proc_size
        track = self._track
        info = track.frame_info(proc_w, proc_h)
        raw_peak_small: tuple[int, int] | None = None
        peak_small: tuple[int, int] | None = None
        rejected = False
        coasting = info.coasting

        if track.mode == TrackMode.SEARCH:
            candidate = self._search_candidate(
                color_mask, track, proc_w, proc_h
            )
            if candidate is not None:
                px, py = candidate.x, candidate.y
                raw_peak_small = (px, py)
                if self._peak_rejected_by_ignore(px, py):
                    rejected = True
                    track.search_miss()
                else:
                    in_roi = self._in_roi_proc(px, py)
                    track.register_search_hit(
                        px,
                        py,
                        candidate.density_score,
                        candidate.blob_area,
                        in_roi=in_roi,
                        has_motion=True,
                    )
                    peak_small = (px, py)
            else:
                track.search_miss()
        else:
            last = track.last_position()
            x0, y0, x1, y1 = info.search_window_proc or (0, 0, proc_w, proc_h)
            hsv_crop = hsv[y0:y1, x0:x1]
            cfg = track.config
            color_relaxed = self._color_mask(hsv_crop, relax=True)
            slow_in_roi = (
                track.is_slow_motion()
                and last is not None
                and self._in_roi_proc(last[0], last[1])
            )
            if slow_in_roi:
                track_mask = color_relaxed
            else:
                diff_crop = diff_max[y0:y1, x0:x1]
                diff_thresh = max(0, self.diff_threshold - cfg.track_diff_relax)
                _, fg_relaxed = cv2.threshold(
                    diff_crop, diff_thresh, 255, cv2.THRESH_BINARY
                )
                track_mask = cv2.bitwise_and(color_relaxed, fg_relaxed.astype(np.uint8))
            candidate = self._find_peak_in_mask(
                track_mask,
                radius=cfg.track_density_radius,
                anchor_proc=last,
                offset_x=x0,
                offset_y=y0,
                min_contour_area=cfg.min_track_area,
            )
            accepted = False
            if candidate is not None:
                px, py = candidate.x, candidate.y
                raw_peak_small = (px, py)
                in_roi = self._in_roi_proc(px, py)
                if track.qualify_track_peak(
                    px,
                    py,
                    candidate.density_score,
                    candidate.blob_area,
                    in_roi=in_roi,
                ):
                    if self._peak_rejected_by_ignore(px, py):
                        rejected = True
                    else:
                        track.accept_track_peak(px, py, candidate.density_score)
                        track.update_track_health(
                            in_roi=in_roi, blob_area=candidate.blob_area
                        )
                        peak_small = (px, py)
                        accepted = True
            if not accepted:
                coast_pos = track.track_miss()
                if coast_pos is not None:
                    peak_small = coast_pos
                    coasting = track.is_coasting
                    in_roi = self._in_roi_proc(coast_pos[0], coast_pos[1])
                    track.update_track_health(
                        in_roi=in_roi, blob_area=cfg.min_track_area
                    )
                info = track.frame_info(proc_w, proc_h)

        peak_full = self._to_full(*peak_small) if peak_small else None

        foreground = None
        color_fg = None
        track_debug_proc = None
        if debug:
            foreground = cv2.bitwise_and(small, small, mask=fg_mask)
            color_fg = cv2.bitwise_and(small, small, mask=color_mask)
            track_debug_proc = draw_track_debug_panel(
                color_mask,
                info,
                raw_peak_small,
            )

        mode_label: Literal["search", "track"] = (
            "search" if track.mode == TrackMode.SEARCH else "track"
        )

        return TrackResult(
            density_peak=peak_full,
            density_peak_proc=peak_small,
            foreground=foreground,
            color_fg=color_fg,
            rejected_by_ignore=rejected,
            raw_peak_proc=raw_peak_small,
            track_mode=mode_label,
            coasting=coasting,
            search_window_proc=info.search_window_proc,
            predicted_proc=info.predicted_proc,
            track_debug_proc=track_debug_proc,
        )


def open_window(name: str, width: int, height: int, x: int, y: int) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.moveWindow(name, x, y)


def parse_args():
    p = argparse.ArgumentParser(description="Fast density-based ball tracker.")
    p.add_argument(
        "--source",
        default="/dev/video42",
        help="Live device or video file path",
    )
    p.add_argument("--device", default=None, help="Deprecated: use --source")
    p.add_argument(
        "--color",
        choices=("white", "yellow"),
        default="white",
        help="Color target for ball detection (toggle at runtime with c).",
    )
    p.add_argument("--threshold", type=int, default=25)
    p.add_argument("--white", type=int, default=140)
    p.add_argument("--saturation", type=int, default=100)
    p.add_argument("--density-radius", type=int, default=12)
    p.add_argument(
        "--resolution",
        choices=sorted(RESOLUTION_PRESETS),
        default="1080",
        help="Capture resolution preset (must match RESOLUTION used in start_gopro.sh).",
    )
    p.add_argument(
        "--process-scale",
        type=float,
        default=None,
        help="Process at this fraction of capture resolution (default depends on --resolution).",
    )
    p.add_argument(
        "--no-debug",
        action="store_true",
        help="Hide debug windows for maximum FPS.",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=f"Load saved preset JSON (default: {DEFAULT_PRESET_PATH} if it exists).",
    )
    p.add_argument(
        "--save-calibration",
        type=Path,
        default=None,
        help="Save preset to this path after interactive calibration.",
    )
    p.add_argument(
        "--recalibrate",
        action="store_true",
        help="Run interactive calibration and overwrite the saved preset.",
    )
    p.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Run without spatial calibration (legacy behavior).",
    )
    return p.parse_args()


def resolve_field_calibration(
    args: argparse.Namespace,
    frame_source,
) -> FieldCalibration | None:
    """Load a saved preset, or run the calibration UI and save for next time."""
    if args.skip_calibration:
        return None

    load_path = args.calibration or DEFAULT_PRESET_PATH
    save_path = args.save_calibration or load_path

    if args.recalibrate or not load_path.is_file():
        cal = run_calibration_ui(frame_source, live=frame_source.is_live)
        if cal is None:
            frame_source.release()
            raise SystemExit("Calibration cancelled.")
        save_calibration(cal, save_path)
        print(f"Saved field preset to {save_path} (+ {save_path.stem}_ignore.png)")
        return cal

    cal = load_calibration(load_path)
    print(f"Loaded field preset from {load_path}  (--recalibrate to redraw)")
    return cal


def color_window_title(target: ColorTarget) -> str:
    label = "White" if target == "white" else "Yellow"
    return f"Color Filter ({label})"


def draw_color_panel_label(img: np.ndarray, target: ColorTarget) -> None:
    label = "WHITE" if target == "white" else "YELLOW"
    cv2.putText(
        img,
        f"Target: {label}  (c=toggle)",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )


def draw_track_debug_panel(
    color_mask: np.ndarray,
    info: TrackFrameInfo,
    raw_peak_proc: tuple[int, int] | None,
) -> np.ndarray:
    panel = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    panel = (panel // 3).astype(np.uint8)

    if info.search_window_proc is not None:
        x0, y0, x1, y1 = info.search_window_proc
        cv2.rectangle(panel, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (255, 255, 0), 1)

    if info.prev_proc is not None and info.last_proc is not None:
        cv2.arrowedLine(
            panel,
            info.prev_proc,
            info.last_proc,
            (255, 255, 255),
            1,
            tipLength=0.3,
        )

    if info.last_proc is not None:
        cv2.circle(panel, info.last_proc, 4, (0, 255, 255), -1)

    if info.predicted_proc is not None:
        cv2.circle(panel, info.predicted_proc, 4, (255, 0, 255), -1)

    if raw_peak_proc is not None:
        cv2.circle(panel, raw_peak_proc, 4, (0, 0, 255), -1)

    if info.coasting:
        label = "COAST"
    elif info.mode == TrackMode.SEARCH and info.last_proc is not None:
        label = "SEARCH+PRIOR"
    elif info.mode == TrackMode.TRACK and info.slow_motion:
        label = f"TRACK-SLOW pad+{info.extra_pad}"
    elif info.mode == TrackMode.TRACK:
        label = f"TRACK pad+{info.extra_pad}"
    else:
        label = info.mode.name
    cv2.putText(
        panel,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )
    return panel


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
        tint[mask_full > 0] = (0, 0, 255)
        cv2.addWeighted(tint, 0.4, frame, 0.6, 0, frame)


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


def main():
    args = parse_args()
    source_str = args.device or args.source
    is_file = Path(source_str).is_file()
    preset = RESOLUTION_PRESETS[args.resolution]

    if is_file:
        process_scale = (
            args.process_scale if args.process_scale is not None else 1.0
        )
        frame_source = open_frame_source(source_str)
    else:
        process_scale = (
            args.process_scale
            if args.process_scale is not None
            else preset["process_scale"]
        )
        frame_source = open_frame_source(
            source_str, preset["width"], preset["height"]
        )

    cal: FieldCalibration | None = resolve_field_calibration(args, frame_source)

    tracker = BallTracker(
        process_scale=process_scale,
        diff_threshold=args.threshold,
        white_threshold=args.white,
        max_saturation=args.saturation,
        density_radius=args.density_radius,
        color_target=args.color,
    )

    show_debug = not args.no_debug
    frame_count = 0
    t0 = time.perf_counter()
    size_warned = False
    pitch_machine: PitchStateMachine | None = None
    pitch_tracer: PitchTracer | None = None
    field_geom = None
    ignore_mask_ready = cal is None
    roi_mask_ready = cal is None

    if is_file:
        live_w, live_h = 960, 540
    else:
        live_w, live_h = preset["live_win"]
    dbg_w, dbg_h = live_w // 2, live_h // 2
    open_window(WIN_LIVE, live_w, live_h, 80, 80)
    if show_debug:
        open_window(WIN_FG, dbg_w, dbg_h, 80, live_h + 100)
        open_window(WIN_COLOR, dbg_w, dbg_h, dbg_w + 100, live_h + 100)
        open_window(WIN_TRACK, dbg_w, dbg_h, dbg_w * 2 + 120, live_h + 100)
        cv2.setWindowTitle(WIN_COLOR, color_window_title(tracker.color_target))

    print("Calibration: 1=ROI 2=Ignore 3=Strike 4=Release  Enter=start")
    print("Yellow dot = tracking validation  Red dot = pitch recording")
    print(
        "Keys: b=relearn  c=white/yellow  d=debug panels  +/-=bg  "
        "9/0=brightness  [/]=sat  ,/.=density  q=quit"
    )
    if is_file:
        print(f"Playing file {source_str}, processing at {process_scale:.0%} of capture.")
    else:
        print(
            f"Resolution {args.resolution}p ({preset['width']}x{preset['height']}), "
            f"processing at {process_scale:.0%} of capture."
        )
        print(
            "Set RESOLUTION=1080 or RESOLUTION=720 for both start_gopro.sh and run_tracker.sh."
        )

    while True:
        frame = frame_source.read()
        if frame is None:
            break

        frame_count += 1
        elapsed = time.perf_counter() - t0
        fps = frame_count / max(elapsed, 1e-6)

        if not size_warned and frame_source.is_live:
            actual_w, actual_h = frame.shape[1], frame.shape[0]
            if (actual_w, actual_h) != (preset["width"], preset["height"]):
                print(
                    f"Warning: frame is {actual_w}x{actual_h}, expected "
                    f"{preset['width']}x{preset['height']}. "
                    f"Restart start_gopro.sh with RESOLUTION={args.resolution}."
                )
            size_warned = True

        frame_h, frame_w = frame.shape[:2]
        if cal is not None and not ignore_mask_ready:
            tracker.set_ignore_mask_proc(
                ignore_mask_for_process(cal, frame_w, frame_h, process_scale)
            )
            ignore_mask_ready = True

        if cal is not None and not roi_mask_ready:
            proc_w = max(1, int(frame_w * process_scale))
            proc_h = max(1, int(frame_h * process_scale))
            tracker.set_roi_proc(
                scaled_polygons(
                    cal.roi,
                    frame_width=proc_w,
                    frame_height=proc_h,
                )
            )
            roi_mask_ready = True

        if cal is not None and pitch_machine is None:
            field_geom = build_field_geometry(
                cal,
                frame_width=frame_w,
                frame_height=frame_h,
                process_scale=process_scale,
            )
            pitch_machine = PitchStateMachine(field_geom)
            pitch_tracer = PitchTracer(fade_sec=pitch_machine.config.tracer_fade_sec)

        result = tracker.process(frame, debug=show_debug)

        detection = result.density_peak if result else None
        pitch_recording = False
        now = elapsed

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

            if pitch_recording:
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

        if is_file:
            status = (
                f"{fps:.0f} fps  {frame.shape[1]}x{frame.shape[0]}"
                f"  proc={tracker.process_scale:.0%}"
            )
        else:
            status = (
                f"{fps:.0f} fps  {args.resolution}p ({frame.shape[1]}x{frame.shape[0]})"
                f"  proc={tracker.process_scale:.0%}"
            )
        status += f"  target={tracker.color_target}"
        if not tracker.is_ready:
            n, total = tracker.warmup_progress
            status += f"  learning bg {n}/{total}"
        elif result:
            if result.coasting:
                status += "  mode=COAST"
            else:
                status += f"  mode={result.track_mode.upper()}"
        if result and result.density_peak:
            x, y = result.density_peak
            status += f"  ball=({x},{y})"
        elif tracker.is_ready:
            status += "  no ball"
        if pitch_recording:
            status += "  PITCH"
        elif pitch_machine is not None and pitch_machine._cooldown > 0:
            status += "  cooldown"
        else:
            status += "  idle"

        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if cal is not None:
            draw_calibration_overlay(frame, cal)
        if pitch_tracer is not None:
            pitch_tracer.draw(frame, now)
        draw_track_dot(frame, detection, pitch_active=pitch_recording)

        cv2.imshow(WIN_LIVE, frame)

        if show_debug and result and result.foreground is not None:
            cv2.imshow(WIN_FG, result.foreground)
            color_view = result.color_fg.copy()
            draw_track_dot(color_view, result.density_peak_proc, pitch_active=pitch_recording)
            draw_color_panel_label(color_view, tracker.color_target)
            cv2.imshow(WIN_COLOR, color_view)
            if result.track_debug_proc is not None:
                cv2.imshow(WIN_TRACK, result.track_debug_proc)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            tracker.reset()
            if pitch_machine is not None:
                pitch_machine.reset()
            if pitch_tracer is not None:
                pitch_tracer.abort_live()
        if key == ord("c"):
            tracker.toggle_color_target()
            cv2.setWindowTitle(WIN_COLOR, color_window_title(tracker.color_target))
        if key == ord("d"):
            show_debug = not show_debug
            if show_debug:
                open_window(WIN_FG, dbg_w, dbg_h, 80, live_h + 100)
                open_window(WIN_COLOR, dbg_w, dbg_h, dbg_w + 100, live_h + 100)
                open_window(WIN_TRACK, dbg_w, dbg_h, dbg_w * 2 + 120, live_h + 100)
                cv2.setWindowTitle(WIN_COLOR, color_window_title(tracker.color_target))
            else:
                cv2.destroyWindow(WIN_FG)
                cv2.destroyWindow(WIN_COLOR)
                cv2.destroyWindow(WIN_TRACK)
        if key in (ord("+"), ord("=")):
            tracker.diff_threshold = min(100, tracker.diff_threshold + 3)
        if key == ord("-"):
            tracker.diff_threshold = max(5, tracker.diff_threshold - 3)
        if key == ord("9"):
            if tracker.color_target == "white":
                tracker.white_threshold = max(50, tracker.white_threshold - 5)
            else:
                tracker.yellow_min_val = max(50, tracker.yellow_min_val - 5)
        if key == ord("0"):
            if tracker.color_target == "white":
                tracker.white_threshold = min(255, tracker.white_threshold + 5)
            else:
                tracker.yellow_min_val = min(255, tracker.yellow_min_val + 5)
        if key == ord("["):
            if tracker.color_target == "white":
                tracker.max_saturation = max(10, tracker.max_saturation - 5)
            else:
                tracker.yellow_min_sat = max(10, tracker.yellow_min_sat - 5)
        if key == ord("]"):
            if tracker.color_target == "white":
                tracker.max_saturation = min(255, tracker.max_saturation + 5)
            else:
                tracker.yellow_min_sat = min(255, tracker.yellow_min_sat + 5)
        if key == ord(","):
            tracker.density_radius = max(3, tracker.density_radius - 2)
        if key == ord("."):
            tracker.density_radius += 2

    frame_source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
