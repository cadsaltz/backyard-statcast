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

from calibration import FieldCalibration, load_calibration, save_calibration
from calibration_ui import run_calibration_ui
from frame_source import open_frame_source
from spatial_filter import (
    classify_point,
    resize_ignore_mask,
    scale_point_to_pixels,
    scaled_polygons,
)

WIN_LIVE = "Ball Tracker"
WIN_FG = "Foreground"
WIN_COLOR = "Color Filter"

ColorTarget = Literal["white", "yellow"]

RESOLUTION_PRESETS = {
    "1080": {"width": 1920, "height": 1080, "process_scale": 0.5, "live_win": (960, 540)},
    "720": {"width": 1280, "height": 720, "process_scale": 1.0, "live_win": (640, 360)},
}

PITCH_IDLE_FRAMES = 30


@dataclass
class TrackResult:
    density_peak: tuple[int, int] | None  # full-frame coords
    density_peak_proc: tuple[int, int] | None = None  # process-resolution coords
    foreground: np.ndarray | None = None
    color_fg: np.ndarray | None = None
    rejected_by_ignore: bool = False
    raw_peak_proc: tuple[int, int] | None = None


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

        self._bg: np.ndarray | None = None
        self._warmup_stack: list[np.ndarray] = []
        self._noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._density_ksize = density_radius * 2 + 1
        self._proc_size: tuple[int, int] | None = None  # (w, h)

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

    def set_ignore_mask_proc(self, mask: np.ndarray | None) -> None:
        self.ignore_mask_proc = mask

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

    def _density_peak(self, color_mask: np.ndarray) -> tuple[int, int] | None:
        if not cv2.countNonZero(color_mask):
            return None
        ksize = self.density_radius * 2 + 1
        # boxFilter counts neighbors — ~14x faster than float32 filter2D at 1080p.
        density = cv2.boxFilter(
            color_mask, cv2.CV_32F, (ksize, ksize), normalize=False
        )
        _, _, _, max_loc = cv2.minMaxLoc(density)
        return int(max_loc[0]), int(max_loc[1])

    def _color_mask(self, hsv: np.ndarray) -> np.ndarray:
        if self.color_target == "yellow":
            return cv2.inRange(
                hsv,
                (self.yellow_hue_low, self.yellow_min_sat, self.yellow_min_val),
                (self.yellow_hue_high, 255, 255),
            )
        return cv2.inRange(
            hsv,
            (0, 0, self.white_threshold),
            (180, self.max_saturation, 255),
        )

    def toggle_color_target(self) -> ColorTarget:
        self.color_target = "yellow" if self.color_target == "white" else "white"
        return self.color_target

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

        peak_small = self._density_peak(color_mask)
        raw_peak_small = peak_small
        rejected = False
        if peak_small and self.ignore_mask_proc is not None:
            px, py = peak_small
            h, w = self.ignore_mask_proc.shape[:2]
            if 0 <= px < w and 0 <= py < h and self.ignore_mask_proc[py, px] > 0:
                peak_small = None
                rejected = True
        peak_full = self._to_full(*peak_small) if peak_small else None

        foreground = None
        color_fg = None
        if debug:
            foreground = cv2.bitwise_and(small, small, mask=fg_mask)
            color_fg = cv2.bitwise_and(small, small, mask=color_mask)

        return TrackResult(
            density_peak=peak_full,
            density_peak_proc=peak_small,
            foreground=foreground,
            color_fg=color_fg,
            rejected_by_ignore=rejected,
            raw_peak_proc=raw_peak_small,
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
    return p.parse_args()


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

    cal: FieldCalibration | None = None
    if args.skip_calibration:
        cal = None
    elif args.calibration is not None:
        cal = load_calibration(args.calibration)
    else:
        first = frame_source.read()
        if first is None:
            frame_source.release()
            raise SystemExit("Could not read first frame for calibration.")
        cal = run_calibration_ui(first)
        if cal is None:
            frame_source.release()
            raise SystemExit("Calibration cancelled.")
        if args.save_calibration is not None:
            save_calibration(cal, args.save_calibration)
            print(f"Saved calibration to {args.save_calibration}")

    tracker = BallTracker(
        process_scale=process_scale,
        diff_threshold=args.threshold,
        white_threshold=args.white,
        max_saturation=args.saturation,
        density_radius=args.density_radius,
        color_target=args.color,
    )

    if cal is not None:
        proc_h = int(cal.frame_height * process_scale)
        proc_w = int(cal.frame_width * process_scale)
        ignore_proc = resize_ignore_mask(cal.ignore_mask, (proc_w, proc_h))
        tracker.set_ignore_mask_proc(ignore_proc)

    show_debug = not args.no_debug
    frame_count = 0
    t0 = time.perf_counter()
    size_warned = False
    pitch_active = False
    idle_frames = 0

    if is_file:
        live_w, live_h = 960, 540
    else:
        live_w, live_h = preset["live_win"]
    dbg_w, dbg_h = live_w // 2, live_h // 2
    open_window(WIN_LIVE, live_w, live_h, 80, 80)
    if show_debug:
        open_window(WIN_FG, dbg_w, dbg_h, 80, live_h + 100)
        open_window(WIN_COLOR, dbg_w, dbg_h, dbg_w + 100, live_h + 100)
        cv2.setWindowTitle(WIN_COLOR, color_window_title(tracker.color_target))

    print("Calibration: 1=ROI 2=Ignore 3=Strike 4=Release  Enter=start")
    print("Yellow dot = tracking validation  Red dot = pitch active (inside ROI)")
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

        result = tracker.process(frame, debug=show_debug)

        detection = result.density_peak if result else None
        if detection:
            idle_frames = 0
            if cal is not None:
                cls = classify_point(detection, cal)
                if cls.in_roi:
                    pitch_active = True
        else:
            idle_frames += 1
            if idle_frames >= PITCH_IDLE_FRAMES:
                pitch_active = False
                idle_frames = 0

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
        elif result and result.density_peak:
            x, y = result.density_peak
            status += f"  ball=({x},{y})"
        else:
            status += "  no ball"
        if pitch_active:
            status += "  PITCH"
        else:
            status += "  track"

        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if cal is not None:
            draw_calibration_overlay(frame, cal)
        draw_track_dot(frame, detection, pitch_active=pitch_active)

        cv2.imshow(WIN_LIVE, frame)

        if show_debug and result and result.foreground is not None:
            cv2.imshow(WIN_FG, result.foreground)
            color_view = result.color_fg.copy()
            draw_track_dot(color_view, result.density_peak_proc, pitch_active=pitch_active)
            draw_color_panel_label(color_view, tracker.color_target)
            cv2.imshow(WIN_COLOR, color_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            tracker.reset()
            pitch_active = False
            idle_frames = 0
        if key == ord("c"):
            tracker.toggle_color_target()
            cv2.setWindowTitle(WIN_COLOR, color_window_title(tracker.color_target))
        if key == ord("d"):
            show_debug = not show_debug
            if show_debug:
                open_window(WIN_FG, dbg_w, dbg_h, 80, live_h + 100)
                open_window(WIN_COLOR, dbg_w, dbg_h, dbg_w + 100, live_h + 100)
                cv2.setWindowTitle(WIN_COLOR, color_window_title(tracker.color_target))
            else:
                cv2.destroyWindow(WIN_FG)
                cv2.destroyWindow(WIN_COLOR)
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
