import argparse
import time

import cv2

from detection_strategies import BallDetection
from ball_tracker import BallTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show GoPro feed with realtime ball detection overlay."
    )
    parser.add_argument(
        "--device",
        default="/dev/video42",
        help="Video device path or camera index. Defaults to /dev/video42.",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Show the raw camera feed without ball detection.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Minimum contour area in pixels for ball candidates.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=2500,
        help="Maximum contour area in pixels for ball candidates.",
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=175,
        help="HSV value threshold (0-255) for bright ball pixels.",
    )
    return parser.parse_args()


def draw_detection(frame, detection: BallDetection) -> None:
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
    cv2.putText(
        frame,
        f"Ball: ({detection.center_x}, {detection.center_y})",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    source = int(args.device) if str(args.device).isdigit() else args.device

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.device}")

    tracker = None
    if not args.no_track:
        tracker = BallTracker(
            min_area=args.min_area,
            max_area=args.max_area,
            brightness_threshold=args.brightness,
        )

    frame_count = 0
    started_at = time.time()
    window_name = "GoPro Live Feed"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame received from camera.")
            time.sleep(0.1)
            continue

        frame_count += 1
        elapsed = max(time.time() - started_at, 0.001)
        fps = frame_count / elapsed

        status = f"GoPro feed | {fps:.1f} FPS | press q to quit"
        if tracker is not None:
            status += " | r reset bg | ./run_strategy_test.sh to compare methods"
            detection = tracker.detect(frame)
            if tracker.is_warming_up:
                current, total = tracker.warmup_progress
                status += f" | warming up ({current}/{total})"
            elif detection is not None:
                draw_detection(frame, detection)
                status += " | ball detected"
            else:
                status += " | no ball"

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r") and tracker is not None:
            tracker.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
