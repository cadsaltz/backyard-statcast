#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video42}"
RESOLUTION="${RESOLUTION:-1080}"

if [[ "$RESOLUTION" != "1080" && "$RESOLUTION" != "720" ]]; then
  echo "RESOLUTION must be 1080 or 720 (got: $RESOLUTION)"
  exit 1
fi

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
exec python "$PROJECT_DIR/track_ball.py" --device "$VIDEO_DEVICE" --resolution "$RESOLUTION" "$@"
