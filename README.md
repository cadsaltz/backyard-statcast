# Backyard Statcast

Live ball tracking from a GoPro HERO10, USB webcam, or recorded video on Linux.

## Quick start

| Goal | Command |
|------|---------|
| **GoPro live tracking** | Terminal 1: `./start_gopro.sh` → Terminal 2: `./run_tracker.sh` |
| **USB webcam live tracking** | `./run_tracker.sh --input webcam` |
| **Recorded video tracking** | `./run_video.sh clip.mp4` |

---

## User guide: launcher scripts

Three shell scripts start everything. Each script has **its own options** plus **passthrough options** that are forwarded to `track_ball.py` unchanged.

```text
GoPro USB  -->  start_gopro.sh  -->  /dev/video42  --+
                                                       |
USB webcam  -----------------------------------------+-->  run_tracker.sh  -->  track_ball.py
                                                       |
video file  -->  run_video.sh  ------------------------+
```

### Syntax

```bash
./start_gopro.sh [--resolution 1080|720] [--fov wide|narrow|superview|linear]

./run_tracker.sh [--input gopro|webcam] [--resolution 1080|720] [track_ball.py options...]

./run_video.sh <video-file> [track_ball.py options...]
# or: VIDEO_FILE=clip.mp4 ./run_video.sh [track_ball.py options...]
```

---

### `start_gopro.sh`

Starts the GoPro USB webcam bridge. **GoPro workflow only.** Leave running in Terminal 1.

#### Native options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--resolution` | CLI flag | `1080` | GoPro stream resolution: `1080` or `720`. Must match `run_tracker.sh --resolution`. |
| `--resolution=720` | CLI flag | — | Shorthand for `--resolution 720`. |
| `RESOLUTION` | env var | `1080` | Same as `--resolution`. Used only when the flag is omitted. |
| `--fov` | CLI flag | `narrow` | GoPro lens: `wide`, `narrow`, `superview`, `linear`. |
| `--fov=linear` | CLI flag | — | Shorthand for `--fov linear`. |
| `FOV` | env var | `narrow` | Same as `--fov`. Used only when the flag is omitted. |
| `GOPRO_HOST_IP` | env var | auto | Force laptop USB IP (e.g. `172.23.138.52`). Normally auto-detected via `enx*` interface. |

No passthrough options — this script only talks to the GoPro.

#### Examples

```bash
./start_gopro.sh
./start_gopro.sh --resolution 720
./start_gopro.sh --fov linear
./start_gopro.sh --resolution 720 --fov narrow
GOPRO_HOST_IP=172.23.138.52 ./start_gopro.sh
```

#### Success checklist

- GoPro screen shows **webcam**
- Terminal keeps printing `ffmpeg` output
- `/dev/video42` exists and is a capture device

---

### `run_tracker.sh`

Main entry point for **live** ball tracking. Picks the camera, sets resolution, then runs `track_ball.py`.

#### Native options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | CLI flag | `gopro` | Video source: `gopro` (needs bridge) or `webcam` (direct USB camera). |
| `--input=webcam` | CLI flag | — | Shorthand for `--input webcam`. |
| `INPUT_SOURCE` | env var | `gopro` | Same as `--input`. Used only when the flag is omitted. |
| `--resolution` | CLI flag | `1080` (GoPro), `720` (webcam) | Capture preset: `1080` (1920×1080) or `720` (1280×720). For GoPro, must match `start_gopro.sh`. |
| `--resolution=720` | CLI flag | — | Shorthand for `--resolution 720`. |
| `RESOLUTION` | env var | same as above | Same as `--resolution`. Used only when the flag is omitted. |
| `VIDEO_DEVICE` | env var | auto | Override device path. Default: `/dev/video42` (GoPro) or first external USB webcam (skips laptop built-in). |

#### Passthrough options (forwarded to `track_ball.py`)

Any flag not recognized by `run_tracker.sh` is passed through. All of these work:

| Option | Default | Description |
|--------|---------|-------------|
| `--color white\|yellow` | `white` | Ball color target. Toggle live with `c`. |
| `--threshold` | `25` | Motion sensitivity. Higher = less sensitive. Tune live with `-`/`+`. |
| `--white` | `140` | Brightness cutoff (white balls). Tune live with `9`/`0`. |
| `--saturation` | `100` | Max saturation (white balls). Tune live with `[`/`]`. |
| `--density-radius` | `12` | Peak search radius in pixels. Tune live with `,`/`.`. |
| `--process-scale` | auto | Detection scale fraction. Default `0.5` at 1080p, `1.0` at 720p. |
| `--no-debug` | off | Hide debug panels for more FPS. Toggle live with `d`. |
| `--calibration PATH` | `configs/field.json` | Load saved field preset. |
| `--save-calibration PATH` | same as `--calibration` | Save preset path after calibration UI. |
| `--recalibrate` | off | Force calibration UI even if preset exists. |
| `--skip-calibration` | off | Run without ROI / ignore zones. |

> **Note:** Do not pass `--source` or `--resolution` twice — `run_tracker.sh` sets both automatically from `--input`, `VIDEO_DEVICE`, and `--resolution`.

#### Examples

```bash
# GoPro (two terminals, 1080p default)
./start_gopro.sh
./run_tracker.sh

# GoPro at 720p (set on BOTH scripts)
./start_gopro.sh --resolution 720
./run_tracker.sh --resolution 720

# USB webcam (one terminal, 720p default)
./run_tracker.sh --input webcam

# Webcam on a specific device
VIDEO_DEVICE=/dev/video2 ./run_tracker.sh --input webcam

# Tuning and calibration
./run_tracker.sh --color yellow --no-debug
./run_tracker.sh --recalibrate
./run_tracker.sh --skip-calibration
./run_tracker.sh --input webcam --process-scale 0.5
./run_tracker.sh --calibration configs/backyard.json
```

---

### `run_video.sh`

**Recorded video** tracking. No camera or GoPro bridge needed.

#### Native options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `<video-file>` | positional arg | required | Path to `.mp4`, `.mov`, etc. |
| `VIDEO_FILE` | env var | — | Alternative to positional arg: `VIDEO_FILE=clip.mp4 ./run_video.sh` |
| `COLOR` | env var | `white` | Ball color: `white` or `yellow`. Passed as `--color` to `track_ball.py`. |

#### Passthrough options (forwarded to `track_ball.py`)

Same passthrough table as `run_tracker.sh` above, **except**:

- `--resolution` has no effect on files (native video resolution is used)
- `--source` is set automatically from the video file path

| Option | Default | Description |
|--------|---------|-------------|
| `--color white\|yellow` | `white` (or `COLOR` env) | Ball color target. |
| `--threshold` | `25` | Motion sensitivity. |
| `--white` | `140` | Brightness cutoff. |
| `--saturation` | `100` | Max saturation. |
| `--density-radius` | `12` | Peak search radius. |
| `--process-scale` | `1.0` | Detection scale (default full resolution for files). |
| `--no-debug` | off | Hide debug panels. |
| `--calibration PATH` | `configs/field.json` | Load field preset. |
| `--save-calibration PATH` | same as `--calibration` | Save preset path. |
| `--recalibrate` | off | Force calibration UI. |
| `--skip-calibration` | off | Skip spatial filtering. |

#### Examples

```bash
./run_video.sh clip.mp4
./run_video.sh clip.mp4 --color yellow --no-debug
COLOR=yellow ./run_video.sh clip.mp4
./run_video.sh clip.mp4 --recalibrate
./run_video.sh clip.mp4 --skip-calibration --process-scale 0.5
VIDEO_FILE=clip.mp4 ./run_video.sh --calibration configs/backyard.json
```

---

## Resolution and process scaling

| `--resolution` | Capture size | Default `--process-scale` | Internal detection size |
|----------------|-------------|---------------------------|---------------------------|
| `1080` | 1920×1080 | `0.5` | 960×540 |
| `720` | 1280×720 | `1.0` | 1280×720 |

**GoPro:** set `--resolution` on **both** `start_gopro.sh` and `run_tracker.sh`.

**Webcam:** `run_tracker.sh --input webcam` defaults to `720`. Override with `--resolution 720` (or `1080` if your cam supports it).

**Recorded video:** resolution comes from the file. Override detection cost with `--process-scale`.

```bash
# 1080p capture but full-res detection (slower, more precise)
./run_tracker.sh --process-scale 1.0

# 720p capture but half-res detection (faster)
./run_tracker.sh --input webcam --process-scale 0.5
```

---

## Runtime keys

Pressed inside the OpenCV window while tracking.

### Tracking (`track_ball.py`)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `b` | Reset background, pitch state, and live pitch trace |
| `c` | Toggle ball color: white ↔ yellow |
| `d` | Toggle debug panels |
| `-` / `+` | Motion threshold down / up |
| `9` / `0` | Brightness down / up |
| `[` / `]` | Saturation tolerance down / up |
| `,` / `.` | Density radius down / up |

Yellow dot = tracker lock. Red dot = pitch recording active.

### Calibration UI (first run or `--recalibrate`)

| Key | Action |
|-----|--------|
| `1`–`4` | Switch mode: ROI / ignore / strike zone / release zone |
| `[` / `]` | Brush size (ignore mode) |
| `c` | Clear current region |
| `u` | Undo last ignore stroke |
| `Enter` / `s` | Finish and start tracking |
| `q` / `Esc` | Cancel |

Recorded video: `,`/`.` or arrow keys scrub frames.

---

## Field calibration

On first run (or with `--recalibrate`), define four regions:

1. **ROI** — where the ball can appear (hard filter)
2. **Ignore zones** — painted false-positive masks (sky, fence, etc.)
3. **Strike zone** — visual overlay
4. **Release zone** — where pitch recording starts

Saved to `configs/field.json` + `configs/field_ignore.png`. Later runs load automatically.

```bash
./run_tracker.sh                              # loads preset if it exists
./run_tracker.sh --recalibrate                # redraw and overwrite preset
./run_tracker.sh --skip-calibration           # no spatial filtering
./run_video.sh clip.mp4 --calibration configs/backyard.json
```

---

## Pitch detection

When calibration is loaded:

- Recording starts on release-like motion in the release zone
- Cyan flight path drawn live during recording
- Path fades over ~4 seconds after pitch ends
- Analytics logged to console (v1)

Status line: `PITCH` while recording, `idle` otherwise.

---

## Other tools

### `run_strategy_test.sh`

Compare detection algorithms live. Requires GoPro bridge.

```bash
./start_gopro.sh
./run_strategy_test.sh [--strategy mog2|brightness|combined|frame_diff|hough|adaptive|mediapipe|yolo]
```

| Option | Default | Description |
|--------|---------|-------------|
| `VIDEO_DEVICE` | `/dev/video42` | Capture device override |

### `show_gopro.py`

Raw camera viewer, no tracking or calibration.

```bash
python show_gopro.py [--device /dev/video42] [--no-track]
```

### `track_ball.py` (direct)

Call directly for full control. Same flags as the passthrough tables above, plus:

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `/dev/video42` | Device path or video file |
| `--device` | — | Deprecated alias for `--source` |

```bash
python track_ball.py --source /dev/video42 --resolution 1080
python track_ball.py --source clip.mp4 --color yellow
```

---

## Files

| File | Purpose |
|------|---------|
| `start_gopro.sh` | GoPro USB bridge → `/dev/video42` |
| `run_tracker.sh` | Live tracking launcher (GoPro or webcam) |
| `run_video.sh` | Recorded-video tracking launcher |
| `video_source.sh` | Device auto-detection (used by `run_tracker.sh`) |
| `track_ball.py` | Core ball tracker |
| `run_strategy_test.sh` | Detection algorithm comparison |
| `show_gopro.py` | Simple raw feed viewer |
| `gopro_as_webcam_on_linux/` | Third-party GoPro USB helper |

---

## Prerequisites

- GoPro HERO10 + USB data cable (GoPro workflow only)
- Camera set to **GoPro Connect** (GoPro workflow only)
- Python virtualenv in `.venv`
- System packages: `ffmpeg`, `v4l2loopback-dkms`, `v4l2loopback-utils`
- Cloned helper: `gopro_as_webcam_on_linux/`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

- GoPro live feed is ~1080p30 in webcam mode, not high-FPS recording mode.
- USB webcams (e.g. Logitech C270) max at 720p — `--input webcam` defaults to `--resolution 720`.
- If `/dev/video42` is missing, the GoPro bridge is not running.
- Secure Boot: after installing `v4l2loopback-dkms`, enroll the MOK key at boot if prompted.
