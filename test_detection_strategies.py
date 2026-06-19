"""Compare ball-detection strategies live on the GoPro feed.

Setup (once):
  python download_models.py

Run:
  ./run_strategy_test.sh
  ./run_strategy_test.sh -- --strategy mediapipe

Keys:
  1-8   switch strategy
  7     mediapipe (COCO sports ball)
  8     yolo (COCO sports ball, ONNX)
  +/-   brightness threshold (classical strategies)
  9/0   ML confidence down/up (mediapipe, yolo)
  o     cycle ROI crop (0%, 20%, 30%, 40% — hides pitcher at top)
  ,/.   min area
  [/]   max area
  d     toggle debug panel
  r     reset current strategy
  q     quit
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from detection_strategies import BallDetection, STRATEGY_NAMES, StrategyConfig, create_strategy

ROI_PRESETS = (0.0, 0.20, 0.30, 0.40)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live comparison of ball detection strategies."
    )
    parser.add_argument(
        "--device",
        default="/dev/video42",
        help="Video device path or camera index.",
    )
    parser.add_argument(
        "--strategy",
        default="mediapipe",
        choices=STRATEGY_NAMES,
        help="Starting strategy.",
    )
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-area", type=int, default=2500)
    parser.add_argument("--brightness", type=int, default=175)
    parser.add_argument("--diff-threshold", type=int, default=25)
    parser.add_argument("--ml-confidence", type=float, default=0.15)
    return parser.parse_args()


def draw_detection(frame: np.ndarray, detection: BallDetection) -> None:
    x, y, w, h = detection.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.circle(
        frame,
        (detection.center_x, detection.center_y),
        detection.radius,
        (0, 255, 0),
        2,
    )
    cv2.drawMarker(
        frame,
        (detection.center_x, detection.center_y),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=2,
    )


def draw_roi_line(frame: np.ndarray, roi_top_frac: float) -> None:
    if roi_top_frac <= 0:
        return
    y = int(frame.shape[0] * roi_top_frac)
    cv2.line(frame, (0, y), (frame.shape[1], y), (255, 0, 255), 2)
    cv2.putText(
        frame,
        "ROI top (pitcher excluded below line)",
        (12, max(y - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    *,
    strategy_name: str,
    config: StrategyConfig,
    fps: float,
    detected: bool,
    center: tuple[int, int] | None,
    warming_up: bool,
    warmup_progress: tuple[int, int],
    show_debug: bool,
    debug_info: str,
) -> None:
    lines = [
        f"Strategy: {strategy_name} | {fps:.1f} FPS | debug={'on' if show_debug else 'off'}",
        "1-8 strategy | 7=mediapipe 8=yolo | 9/0 ML conf | o ROI | d debug | r reset | q quit",
        (
            f"min_area={config.min_area} max_area={config.max_area} "
            f"brightness={config.brightness_threshold} ml_conf={config.ml_confidence:.2f} "
            f"roi_top={config.roi_top_frac:.0%}"
        ),
    ]
    if debug_info:
        lines.append(debug_info)
    if warming_up:
        current, total = warmup_progress
        lines.append(f"Warming up: {current}/{total}")
    elif detected and center is not None:
        lines.append(f"Ball: ({center[0]}, {center[1]})")
    else:
        lines.append("Ball: not detected")

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 22


def compose_view(
    frame: np.ndarray,
    debug_mask: np.ndarray | None,
    debug_frame: np.ndarray | None,
    show_debug: bool,
    roi_top_frac: float,
) -> np.ndarray:
    if not show_debug:
        return frame

    if debug_frame is not None:
        panel = debug_frame.copy()
        if roi_top_frac > 0:
            cv2.putText(
                panel,
                "(ROI crop — coords offset on main view)",
                (10, panel.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
    elif debug_mask is not None:
        panel = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
    else:
        return frame

    panel = cv2.resize(panel, (frame.shape[1] // 3, frame.shape[0]))
    cv2.putText(
        panel,
        "debug",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.hstack([frame, panel])


def switch_strategy(name: str, config: StrategyConfig):
    try:
        return create_strategy(name, config)
    except FileNotFoundError as exc:
        print(exc)
        print("Run: python download_models.py")
        return None


def main():
    args = parse_args()
    source = int(args.device) if str(args.device).isdigit() else args.device

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.device}")

    config = StrategyConfig(
        min_area=args.min_area,
        max_area=args.max_area,
        brightness_threshold=args.brightness,
        diff_threshold=args.diff_threshold,
        ml_confidence=args.ml_confidence,
    )

    strategy_index = STRATEGY_NAMES.index(args.strategy)
    strategy = switch_strategy(STRATEGY_NAMES[strategy_index], config)
    if strategy is None:
        raise SystemExit(1)

    show_debug = True
    frame_count = 0
    started_at = time.time()
    window_name = "Detection Strategy Test"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame received from camera.")
            time.sleep(0.1)
            continue

        frame_count += 1
        elapsed = max(time.time() - started_at, 0.001)
        fps = frame_count / elapsed

        result = strategy.detect(frame)
        display = frame.copy()
        draw_roi_line(display, config.roi_top_frac)

        center = None
        if result.detection is not None:
            draw_detection(display, result.detection)
            center = (result.detection.center_x, result.detection.center_y)

        draw_hud(
            display,
            strategy_name=strategy.name,
            config=config,
            fps=fps,
            detected=result.detection is not None,
            center=center,
            warming_up=strategy.is_warming_up,
            warmup_progress=strategy.warmup_progress,
            show_debug=show_debug,
            debug_info=result.debug_info,
        )
        view = compose_view(
            display,
            result.debug_mask,
            result.debug_frame,
            show_debug,
            config.roi_top_frac,
        )
        cv2.imshow(window_name, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            strategy.reset()
        if key == ord("d"):
            show_debug = not show_debug
        if key in (ord("+"), ord("=")):
            config.brightness_threshold = min(255, config.brightness_threshold + 5)
        if key == ord("-"):
            config.brightness_threshold = max(0, config.brightness_threshold - 5)
        if key == ord("9"):
            config.ml_confidence = max(0.01, config.ml_confidence - 0.05)
        if key == ord("0"):
            config.ml_confidence = min(0.95, config.ml_confidence + 0.05)
        if key == ord("o"):
            current = config.roi_top_frac
            try:
                idx = ROI_PRESETS.index(current)
            except ValueError:
                idx = 0
            config.roi_top_frac = ROI_PRESETS[(idx + 1) % len(ROI_PRESETS)]
        if key == ord(","):
            config.min_area = max(1, config.min_area - 5)
        if key == ord("."):
            config.min_area += 5
        if key == ord("["):
            config.max_area = max(config.min_area + 1, config.max_area - 100)
        if key == ord("]"):
            config.max_area += 100
        if ord("1") <= key <= ord("8"):
            idx = key - ord("1")
            if idx < len(STRATEGY_NAMES):
                new_strategy = switch_strategy(STRATEGY_NAMES[idx], config)
                if new_strategy is not None:
                    strategy_index = idx
                    strategy = new_strategy

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
