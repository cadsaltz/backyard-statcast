#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=video_source.sh
source "$PROJECT_DIR/video_source.sh"

INPUT_SOURCE="${INPUT_SOURCE:-gopro}"
RESOLUTION="${RESOLUTION:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_SOURCE="$2"
      shift 2
      ;;
    --input=*)
      INPUT_SOURCE="${1#*=}"
      shift
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    --resolution=*)
      RESOLUTION="${1#*=}"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$RESOLUTION" ]]; then
  if [[ "$INPUT_SOURCE" == "webcam" ]]; then
    RESOLUTION=720
  else
    RESOLUTION=1080
  fi
fi

if [[ "$RESOLUTION" != "1080" && "$RESOLUTION" != "720" ]]; then
  echo "Resolution must be 1080 or 720 (got: $RESOLUTION)"
  exit 1
fi

if [[ "$INPUT_SOURCE" != "gopro" && "$INPUT_SOURCE" != "webcam" ]]; then
  echo "INPUT_SOURCE must be gopro or webcam (got: $INPUT_SOURCE)"
  exit 1
fi

if ! VIDEO_DEVICE="$(resolve_video_device "$INPUT_SOURCE")"; then
  if [[ "$INPUT_SOURCE" == "webcam" ]]; then
    echo "No external USB webcam found."
    echo "Plug in the webcam and check with: v4l2-ctl --list-devices"
  fi
  exit 1
fi

if ! device_ready "$VIDEO_DEVICE"; then
  echo "$VIDEO_DEVICE is not ready."
  if [[ "$INPUT_SOURCE" == "gopro" ]]; then
    echo "Start the GoPro bridge first with ./start_gopro.sh"
  else
    echo "Check the webcam connection, or set VIDEO_DEVICE to the correct /dev/video* node."
  fi
  exit 1
fi

echo "Using $INPUT_SOURCE input from $VIDEO_DEVICE at ${RESOLUTION}p"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
exec python "$PROJECT_DIR/track_ball.py" --source "$VIDEO_DEVICE" --resolution "$RESOLUTION" "${EXTRA_ARGS[@]}"
