#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video42}"

device_ready() {
  local info
  info="$(v4l2-ctl -d "$VIDEO_DEVICE" --all 2>/dev/null || true)"
  [[ -e "$VIDEO_DEVICE" && "$info" == *"Video Capture"* ]]
}

if ! device_ready; then
  echo "$VIDEO_DEVICE is not ready."
  echo "Start the GoPro bridge first with ./start_gopro.sh"
  exit 1
fi

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
exec python "$PROJECT_DIR/test_detection_strategies.py" --device "$VIDEO_DEVICE" "$@"
