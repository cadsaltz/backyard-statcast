"""Ball tracker: bg subtract + white filter + density peak (red dot).

Performance: process at reduced resolution, display at capture resolution.
Debug panels open in separate windows (toggle with d).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

WIN_LIVE = "Ball Tracker"
WIN_FG = "Foreground"
WIN_WHITE = "White Filter"

RESOLUTION_PRESETS = {
    "1080": {"width": 1920, "height": 1080, "process_scale": 0.5, "live_win": (960, 540)},
    "720": {"width": 1280, "height": 720, "process_scale": 1.0, "live_win": (640, 360)},
}


@dataclass
class TrackResult:
    density_peak: tuple[int, int] | None  # full-frame coords
    density_peak_proc: tuple[int, int] | None = None  # process-resolution coords
    foreground: np.ndarray | None = None
    white_fg: np.ndarray | None = None


class BallTracker:
    def __init__(
        self,
        process_scale: float = 0.5,
        warmup_frames: int = 30,
        diff_threshold: int = 25,
        white_threshold: int = 140,
        max_saturation: int = 100,
        density_radius: int = 12,
    ) -> None:
        self.process_scale = process_scale
        self.warmup_frames = warmup_frames
        self.diff_threshold = diff_threshold
        self.white_threshold = white_threshold
        self.max_saturation = max_saturation
        self.density_radius = density_radius

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

    def _density_peak(self, white_mask: np.ndarray) -> tuple[int, int] | None:
        if not cv2.countNonZero(white_mask):
            return None
        ksize = self.density_radius * 2 + 1
        # boxFilter counts neighbors — ~14x faster than float32 filter2D at 1080p.
        density = cv2.boxFilter(
            white_mask, cv2.CV_32F, (ksize, ksize), normalize=False
        )
        _, _, _, max_loc = cv2.minMaxLoc(density)
        return int(max_loc[0]), int(max_loc[1])

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
        white_mask = cv2.inRange(
            hsv,
            (0, 0, self.white_threshold),
            (180, self.max_saturation, 255),
        )
        white_mask = cv2.bitwise_and(white_mask, fg_mask)

        peak_small = self._density_peak(white_mask)
        peak_full = self._to_full(*peak_small) if peak_small else None

        foreground = None
        white_fg = None
        if debug:
            foreground = cv2.bitwise_and(small, small, mask=fg_mask)
            white_fg = cv2.bitwise_and(small, small, mask=white_mask)

        return TrackResult(
            density_peak=peak_full,
            density_peak_proc=peak_small,
            foreground=foreground,
            white_fg=white_fg,
        )


def open_window(name: str, width: int, height: int, x: int, y: int) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.moveWindow(name, x, y)


def setup_capture(source: str | int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open: {source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def parse_args():
    p = argparse.ArgumentParser(description="Fast density-based ball tracker.")
    p.add_argument("--device", default="/dev/video42")
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
    return p.parse_args()


def draw_dot(img: np.ndarray, pt: tuple[int, int] | None) -> None:
    if pt is None:
        return
    cv2.circle(img, pt, 7, (0, 0, 255), -1)
    cv2.circle(img, pt, 9, (255, 255, 255), 2)


def main():
    args = parse_args()
    source = int(args.device) if str(args.device).isdigit() else args.device
    preset = RESOLUTION_PRESETS[args.resolution]
    process_scale = (
        args.process_scale if args.process_scale is not None else preset["process_scale"]
    )

    cap = setup_capture(source, preset["width"], preset["height"])
    tracker = BallTracker(
        process_scale=process_scale,
        diff_threshold=args.threshold,
        white_threshold=args.white,
        max_saturation=args.saturation,
        density_radius=args.density_radius,
    )

    show_debug = not args.no_debug
    frame_count = 0
    t0 = time.perf_counter()
    size_warned = False

    live_w, live_h = preset["live_win"]
    dbg_w, dbg_h = live_w // 2, live_h // 2
    open_window(WIN_LIVE, live_w, live_h, 80, 80)
    if show_debug:
        open_window(WIN_FG, dbg_w, dbg_h, 80, live_h + 100)
        open_window(WIN_WHITE, dbg_w, dbg_h, dbg_w + 100, live_h + 100)

    print("Red dot = densest white cluster on foreground.")
    print("Keys: b=relearn  d=debug panels  +/-=bg  9/0=white  [/]=sat  ,/.=density  q=quit")
    print(
        f"Resolution {args.resolution}p ({preset['width']}x{preset['height']}), "
        f"processing at {process_scale:.0%} of capture."
    )
    print("Set RESOLUTION=1080 or RESOLUTION=720 for both start_gopro.sh and run_tracker.sh.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_count += 1
        elapsed = time.perf_counter() - t0
        fps = frame_count / max(elapsed, 1e-6)

        if not size_warned:
            actual_w, actual_h = frame.shape[1], frame.shape[0]
            if (actual_w, actual_h) != (preset["width"], preset["height"]):
                print(
                    f"Warning: frame is {actual_w}x{actual_h}, expected "
                    f"{preset['width']}x{preset['height']}. "
                    f"Restart start_gopro.sh with RESOLUTION={args.resolution}."
                )
            size_warned = True

        result = tracker.process(frame, debug=show_debug)

        status = (
            f"{fps:.0f} fps  {args.resolution}p ({frame.shape[1]}x{frame.shape[0]})"
            f"  proc={tracker.process_scale:.0%}"
        )
        if not tracker.is_ready:
            n, total = tracker.warmup_progress
            status += f"  learning bg {n}/{total}"
        elif result and result.density_peak:
            x, y = result.density_peak
            status += f"  ball=({x},{y})"
        else:
            status += "  no ball"

        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if result:
            draw_dot(frame, result.density_peak)

        cv2.imshow(WIN_LIVE, frame)

        if show_debug and result and result.foreground is not None:
            cv2.imshow(WIN_FG, result.foreground)
            white_view = result.white_fg
            draw_dot(white_view, result.density_peak_proc)
            cv2.imshow(WIN_WHITE, white_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            tracker.reset()
        if key == ord("d"):
            show_debug = not show_debug
            if show_debug:
                open_window(WIN_FG, dbg_w, dbg_h, 80, live_h + 100)
                open_window(WIN_WHITE, dbg_w, dbg_h, dbg_w + 100, live_h + 100)
            else:
                cv2.destroyWindow(WIN_FG)
                cv2.destroyWindow(WIN_WHITE)
        if key in (ord("+"), ord("=")):
            tracker.diff_threshold = min(100, tracker.diff_threshold + 3)
        if key == ord("-"):
            tracker.diff_threshold = max(5, tracker.diff_threshold - 3)
        if key == ord("9"):
            tracker.white_threshold = max(50, tracker.white_threshold - 5)
        if key == ord("0"):
            tracker.white_threshold = min(255, tracker.white_threshold + 5)
        if key == ord("["):
            tracker.max_saturation = max(10, tracker.max_saturation - 5)
        if key == ord("]"):
            tracker.max_saturation = min(255, tracker.max_saturation + 5)
        if key == ord(","):
            tracker.density_radius = max(3, tracker.density_radius - 2)
        if key == ord("."):
            tracker.density_radius += 2

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
