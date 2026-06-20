# Unified Frame Source — Design Spec

**Date:** 2026-06-19  
**Status:** Implemented (2026-06-19)  
**Scope:** Input handling and frame iteration only (no tracking logic changes)

## Problem

The backyard Statcast system tracks a wiffle ball from a live video feed (GoPro via virtual webcam `/dev/video42`). All entry points (`track_ball.py`, `show_gopro.py`, `test_detection_strategies.py`) fuse three responsibilities in a single `main()` loop:

1. Frame capture from a live device
2. Ball tracking on each frame
3. Display, keyboard tuning, and logging

There is no way to run the tracking pipeline on a recorded video file. OpenCV's `VideoCapture` can open both cameras and files, but the codebase never uses file input, and live-specific behavior (retry on failed reads, forced resolution, buffer tuning) is embedded in the main loop.

## Goal

Extend the system so the primary tracker (`track_ball.py`) can run on recorded video as well as live input, with:

- **Identical tracking logic** for both input types — `BallTracker.process()` must not branch on source type
- **Minimal changes** to existing tracking code
- **Clean separation** between frame sourcing and processing
- **Single switch** to choose live camera vs recorded file (one CLI flag or argument change)
- **Simple, incremental** refactor — no state machines, calibration, ROI systems, or over-engineering

## Non-Goals (this spec)

- Changing ball tracking algorithms (`BallTracker`, `detection_strategies.py`, `ml_strategies.py`)
- Playback speed control, frame timestamps, or pitch segmentation
- Refactoring `show_gopro.py` or `test_detection_strategies.py` (may reuse `frame_source.py` later)
- Shell script changes beyond an optional `run_video.sh` helper
- Async/threaded capture queues
- Plugin registries or configuration frameworks

## Recommended Architecture

### Unified frame loop

The right abstraction is a single consumer loop where the only variable is how the next frame is obtained:

```
each iteration:
  frame = source.read()
  if frame is None: break          # EOF or source closed
  result = tracker.process(frame)  # unchanged
  display / log / handle keys      # unchanged
```

Tracking code already follows this contract: `BallTracker.process(frame: np.ndarray)` accepts a BGR frame and returns a `TrackResult`. Detection strategies use the same pattern (`detect(frame)`). The refactor extracts frame sourcing from `main()` without touching these methods.

### Component diagram

```
┌─────────────────────────────────────────────────────────┐
│  main loop (track_ball.py)                              │
│                                                         │
│  while True:                                            │
│    frame = source.read()                                │
│    if frame is None: break                              │
│    result = tracker.process(frame)   ← unchanged        │
│    overlay, imshow, waitKey, keys     ← unchanged       │
└─────────────────────────────────────────────────────────┘
         ▲
         │
┌────────┴────────┐
│   FrameSource   │  frame_source.py (new)
├─────────────────┤
│ LiveFrameSource │  device path, buffer=1, set WxH
│ FileFrameSource │  file path, native resolution, EOF
└─────────────────┘
```

**Invariant:** `BallTracker` and all detection code never import or reference `FrameSource`, file paths, or device paths.

## Alternatives Considered

| Approach | Verdict |
|----------|---------|
| Unified loop + thin `FrameSource` wrapper | **Recommended** — minimal diff, clear seam, easy to extend |
| Two scripts (`track_live.py` / `track_file.py`) | Rejected — duplicates display, keyboard, and tracker setup |
| Async / threaded capture queue | Rejected — unnecessary complexity for two input types |
| Full plugin registry | Rejected — over-engineered for current scope |

OpenCV already abstracts live vs file at `VideoCapture`; we only need a thin wrapper to normalize behavioral differences (EOF vs retry, resolution forcing).

## Components

### `frame_source.py` (new)

A small module (~40–60 lines) providing frame iteration with normalized semantics.

#### `FrameSource` protocol

```python
def read(self) -> np.ndarray | None:
    """Return the next BGR frame, or None when the stream has ended."""

def release(self) -> None:
    """Release underlying capture resources."""
```

A formal ABC is optional; a concrete base class or two standalone classes with a shared factory is sufficient.

#### `LiveFrameSource`

Wraps `cv2.VideoCapture` for camera / v4l2 devices.

| Behavior | Detail |
|----------|--------|
| Open | Device path string or numeric index |
| Buffer | `CAP_PROP_BUFFERSIZE = 1` (drop stale frames on live stream) |
| Resolution | Set `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT` from preset |
| Failed read | Retry internally with brief sleep; only return `None` if capture is no longer opened |
| EOF | N/A — stream is indefinite until user quits |

Move logic from existing `setup_capture()` in `track_ball.py` into this class.

#### `FileFrameSource`

Wraps `cv2.VideoCapture` for recorded files.

| Behavior | Detail |
|----------|--------|
| Open | File path (must exist) |
| Resolution | Use native frame dimensions — do not force WxH |
| Buffer | Default OpenCV buffer (no special tuning) |
| Failed read | Return `None` (end of file) |
| EOF | `read()` returns `None` |

Optionally store `fps` from `CAP_PROP_FPS` for future display pacing; not required in v1.

#### Factory: `open_frame_source()`

```python
def open_frame_source(
    source: str,
    width: int | None = None,
    height: int | None = None,
) -> FrameSource:
```

**Auto-detection:**

- If `Path(source).is_file()` → `FileFrameSource(source)`
- Otherwise → `LiveFrameSource(source, width, height)`

Raises `SystemExit` (or explicit error) if the source cannot be opened.

### `track_ball.py` (modified — main loop only)

**Do not modify:** `BallTracker`, `TrackResult`, `process()`, `reset()`, warmup logic, or detection math.

**Changes:**

1. Replace `cap = setup_capture(...)` with `source = open_frame_source(...)`.
2. Replace the read block:

   ```python
   # Before
   ok, frame = cap.read()
   if not ok:
       time.sleep(0.01)
       continue

   # After
   frame = source.read()
   if frame is None:
       break
   ```

3. Replace `cap.release()` with `source.release()`.
4. Remove or relocate `setup_capture()` to `frame_source.py`.
5. Adjust resolution/size warning logic (see below).

Everything from `tracker.process()` through keyboard handling, debug windows, and overlay drawing remains identical.

### CLI

Replace device-only input with a single source argument:

```
--source SOURCE   Live device path or recorded video file (default: /dev/video42)
```

`--device` may be kept as a deprecated alias for backward compatibility, or removed if all launch paths are updated.

**Resolution preset (`--resolution 1080|720`):**

| Source type | Behavior |
|-------------|----------|
| Live | Sets capture WxH to match GoPro bridge preset (unchanged behavior) |
| File | Ignored for capture; frames use native file resolution |

**Process scale (`--process-scale`):**

| Source type | Behavior |
|-------------|----------|
| Live | Default from resolution preset (0.5 @ 1080p, 1.0 @ 720p) |
| File | Default 1.0, or explicit `--process-scale` override |

### Shell scripts

| Script | Change |
|--------|--------|
| `run_tracker.sh` | No change — still validates v4l2 device and launches live mode |
| `run_video.sh` (optional, new) | `python track_ball.py --source "$VIDEO_FILE"` — no v4l2 check |

## Live vs File Behavioral Differences

All differences are confined to `frame_source.py`. The main loop and tracking code see only `np.ndarray` frames.

| Behavior | Live | File |
|----------|------|------|
| Open target | `/dev/video42`, numeric index | `.mp4`, `.mov`, etc. |
| Force resolution | Yes (GoPro preset) | No (native) |
| `BUFFERSIZE = 1` | Yes | No |
| Failed `read()` | Retry / brief sleep | EOF → `None` |
| End condition | User presses `q` | EOF or `q` |
| Display pacing | `waitKey(1)` | Same in v1 (processes as fast as CPU allows) |

### Size warning

The current main loop warns when actual frame size differs from the resolution preset. For file input:

- Skip the GoPro-specific restart message, or
- Gate the warning on live source only

File frames should not trigger "Restart start_gopro.sh" guidance.

### Background warmup

`BallTracker` learns background from the first N consecutive frames (`warmup_frames = 30`). For recorded video, warmup runs on the first N frames of the clip. This is correct behavior and requires no change. User can press `b` to relearn mid-clip (existing keybinding).

## Data Flow

```
GoPro / webcam                    Recorded file
      │                                │
      ▼                                ▼
LiveFrameSource                  FileFrameSource
      │                                │
      └──────────┬─────────────────────┘
                 │  BGR np.ndarray
                 ▼
           BallTracker.process(frame)
                 │
                 ▼
         overlay + cv2.imshow + keys
```

## Error Handling

| Condition | Behavior |
|-----------|----------|
| Source path does not exist (file mode) | Exit with clear error before loop starts |
| Device cannot be opened | Exit with clear error (existing `setup_capture` behavior) |
| Transient live read failure | Retry inside `LiveFrameSource` (main loop unaffected) |
| File EOF | `read()` returns `None`; main loop breaks cleanly |
| User presses `q` | Break loop (unchanged) |

## Testing / Verification

### Live regression

1. `./start_gopro.sh` then `./run_tracker.sh`
2. Confirm tracking, debug panels, keyboard tuning (`b`, `d`, `+/-`, etc.) behave as before
3. Confirm FPS and resolution match preset

### File mode

1. `python track_ball.py --source recording.mp4`
2. Confirm ball dot appears after warmup
3. Confirm `q` quits cleanly
4. Confirm EOF exits without hanging (no infinite retry loop)

### Tracking isolation

1. Grep `BallTracker` / `process()` — zero references to file, device, `FrameSource`, or source type
2. `profile_tracker.py` (synthetic frames) still passes unchanged

## Implementation Phases

### Phase 1 — `frame_source.py`

- Implement `LiveFrameSource`, `FileFrameSource`, and `open_frame_source()`
- Move `setup_capture()` logic into `LiveFrameSource`

### Phase 2 — `track_ball.py` main loop

- Wire `open_frame_source()` and unified read loop
- Add `--source` CLI argument
- Adjust size warning for file input

### Phase 3 — Optional helpers

- Add `run_video.sh` if desired
- Update README with file mode usage

### Phase 4 — Future (out of scope)

- Reuse `frame_source.py` in `show_gopro.py` and `test_detection_strategies.py`
- `--realtime` flag to throttle file playback to source FPS
- Frame index / timestamp metadata on `read()` return value

## File Change Summary

| File | Action |
|------|--------|
| `frame_source.py` | **Create** — frame sourcing abstraction |
| `track_ball.py` | **Modify** — main loop and CLI only; `BallTracker` untouched |
| `run_video.sh` | **Create** (optional) — convenience launcher for file mode |
| `run_tracker.sh` | **No change** |
| `ball_tracker.py`, `detection_strategies.py`, `ml_strategies.py` | **No change** |

## Success Criteria

- [ ] `python track_ball.py --source /dev/video42` behaves identically to current live mode
- [ ] `python track_ball.py --source pitch_clip.mp4` runs the same tracking pipeline on a file
- [ ] Switching between live and recorded input requires only changing `--source`
- [ ] `BallTracker.process()` has zero input-type branching
- [ ] File playback exits cleanly at EOF
- [ ] No state machines, calibration, or ROI systems introduced
