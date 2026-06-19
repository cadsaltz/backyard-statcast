"""Profile ball tracker pipeline stages."""

import time

import cv2
import numpy as np

from track_ball import BallTracker


def bench(name: str, fn, n: int = 80) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {name:36s} {ms:6.2f} ms  (~{1000/ms:.0f} fps if alone)")
    return ms


def main():
    h, w = 1080, 1920
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.circle(frame, (960, 540), 12, (240, 245, 250), -1)

    print(f"Synthetic capture: {w}x{h}\n")
    print("=== BEFORE-style (1080p, debug on, old density) ===")

    class OldTracker(BallTracker):
        def _density_peak(self, white_mask):
            if not cv2.countNonZero(white_mask):
                return None
            ksize = self.density_radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            density = cv2.filter2D(white_mask.astype(np.float32), cv2.CV_32F, kernel)
            _, _, _, max_loc = cv2.minMaxLoc(density)
            return int(max_loc[0]), int(max_loc[1])

    old = OldTracker(process_scale=1.0, warmup_frames=1)
    old._warmup_stack = [frame.copy()]
    old._learn_background()
    t_old = bench("full 1080p + filter2D + debug", lambda: old.process(frame, debug=True), 40)

    print("\n=== AFTER (optimized) ===")
    opt = BallTracker(process_scale=0.5, warmup_frames=1)
    opt._warmup_stack = [cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)]
    opt._learn_background()
    t_opt_proc = bench("process scale=0.5 no debug", lambda: opt.process(frame, debug=False), 60)
    t_opt_dbg = bench("process scale=0.5 with debug", lambda: opt.process(frame, debug=True), 60)

    print("\n=== Summary ===")
    print(f"  Before (1080p full + filter2D + debug): ~{1000/t_old:.0f} fps compute")
    print(f"  After  (0.5 scale, no debug):           ~{1000/t_opt_proc:.0f} fps compute")
    print(f"  After  (0.5 scale, debug windows):      ~{1000/t_opt_dbg:.0f} fps compute")
    print(f"  Estimated speedup (no debug):           {t_old/t_opt_proc:.1f}x")


if __name__ == "__main__":
    main()
