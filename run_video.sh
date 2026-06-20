#!/usr/bin/env bash
# Recorded video tracker — one terminal, no GoPro bridge required.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_FILE="${VIDEO_FILE:-}"
COLOR="${COLOR:-}"

if [[ -z "$VIDEO_FILE" && $# -lt 1 ]]; then
  echo "Usage: VIDEO_FILE=path/to/video.mp4 ./run_video.sh"
  echo "   or: ./run_video.sh path/to/video.mp4 [track_ball.py options...]"
  echo ""
  echo "Examples:"
  echo "  ./run_video.sh clip.mp4"
  echo "  ./run_video.sh clip.mp4 --color yellow --no-debug"
  echo "  COLOR=yellow ./run_video.sh clip.mp4"
  exit 1
fi

if [[ -n "$COLOR" && "$COLOR" != "white" && "$COLOR" != "yellow" ]]; then
  echo "COLOR must be white or yellow (got: $COLOR)"
  exit 1
fi

if [[ $# -ge 1 ]]; then
  VIDEO_FILE="$1"
  shift
fi

if [[ ! -f "$VIDEO_FILE" ]]; then
  echo "Video file not found: $VIDEO_FILE"
  exit 1
fi

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

args=(--source "$VIDEO_FILE")
if [[ -n "$COLOR" ]]; then
  args+=(--color "$COLOR")
fi

exec python "$PROJECT_DIR/track_ball.py" "${args[@]}" "$@"
