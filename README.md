# Backyard Statcast

Live GoPro HERO10 video feed on Linux for a backyard pitch-tracking project.

Ball tracking is not implemented yet. This repo currently only gets a live camera feed into Python/OpenCV.

## Prerequisites

- GoPro HERO10 connected by USB data cable
- Camera set to GoPro Connect mode
- Python virtualenv in `.venv`
- System packages: `ffmpeg`, `v4l2loopback-dkms`, `v4l2loopback-utils`
- Cloned helper repo: `gopro_as_webcam_on_linux/`

Install Python deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Two-terminal workflow

Use two terminals. One starts the camera bridge. The other shows the live feed.

### Terminal 1: `start_gopro.sh`

This script:

1. Starts the GoPro in webcam mode over USB
2. Receives the camera's UDP stream
3. Pipes it through `ffmpeg` into a virtual Linux webcam device, usually `/dev/video42`

Run it and leave it running:

```bash
./start_gopro.sh
```

When it works, the GoPro screen should show webcam mode and the terminal should keep printing `ffmpeg` output.

Environment variables you can override:

- `GOPRO_HOST_IP` — optional fixed laptop USB network IP; by default the script auto-detects the GoPro `enx*` interface
- `RESOLUTION` — `1080` (default) or `720`
- `FOV` — default `narrow`

If the camera does not enter webcam mode, check that it is on, connected with a data cable, and set to **GoPro Connect** in USB settings. Then confirm Linux sees the USB interface:

```bash
ip -4 addr show | grep -A2 enx
```

You should see an `enx...` device with an address like `172.23.x.x`. If that IP changed after a reboot or reconnect, do not hardcode the old value; just rerun `./start_gopro.sh`.

### Terminal 2: `run_tracker.sh`

This script does **not** start the GoPro bridge. It assumes Terminal 1 is already running and `/dev/video42` exists.

It:

1. Checks that `/dev/video42` is a usable capture device
2. Activates `.venv`
3. Runs `track_ball.py`, which tracks the ball and shows the live feed

Run it after Terminal 1 is up:

```bash
./run_tracker.sh
```

Use the same `RESOLUTION` in both terminals:

```bash
# 1080p (default)
./start_gopro.sh
./run_tracker.sh

# 720p
RESOLUTION=720 ./start_gopro.sh
RESOLUTION=720 ./run_tracker.sh
```

Press `q` in the video window to quit the viewer. That does not stop the GoPro bridge in Terminal 1.

Optional overrides:

- `VIDEO_DEVICE` — default `/dev/video42`
- `RESOLUTION` — `1080` or `720`, must match Terminal 1

## Field calibration

Before tracking, the first frame freezes so you can define:

1. **ROI** (4 clicks) — pitch-active region
2. **Ignore zones** (paint) — hard-masked false-positive areas
3. **Strike zone** (4 clicks) — overlay only
4. **Release zone** (click + drag radius) — overlay only

Keys during calibration: `1`–`4` switch mode, `[`/`]` brush size, `Enter` start, `q` quit.

```bash
# Interactive calibration, save for reuse
python track_ball.py --source /dev/video42 --save-calibration configs/field.json

# Reuse saved calibration
python track_ball.py --source /dev/video42 --calibration configs/field.json

# Legacy: no spatial filtering
python track_ball.py --source /dev/video42 --skip-calibration
```

Yellow dot = tracker validation (Choice B). Red dot = detection inside ROI (`pitch_active`).

## How the pieces relate

```text
GoPro HERO10 (USB)
        |
        v
start_gopro.sh
  -> gopro_as_webcam_on_linux/gopro
  -> ffmpeg
  -> /dev/video42
        |
        v
run_tracker.sh
  -> track_ball.py
  -> OpenCV windows on laptop
```

`start_gopro.sh` creates the video device.
`run_tracker.sh` only reads from that device and displays it.

## Files

- `start_gopro.sh` — start GoPro bridge and virtual webcam
- `run_tracker.sh` — ball tracker from `/dev/video42`
- `track_ball.py` — density-based ball tracking
- `show_gopro.py` — simple raw feed viewer (no tracking)
- `gopro_as_webcam_on_linux/` — third-party GoPro USB webcam helper

## Notes

- The live GoPro webcam feed is about 1080p30, not high-FPS recording mode.
- If `/dev/video42` is missing, the bridge is not running or not ready yet.
- If you reboot after installing `v4l2loopback-dkms` with Secure Boot enabled, you may need to enroll the MOK key once at boot.
