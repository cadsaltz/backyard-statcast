#!/usr/bin/env bash

device_ready() {
  local dev="$1"
  local info
  info="$(v4l2-ctl -d "$dev" --all 2>/dev/null || true)"
  [[ -e "$dev" && "$info" == *"Video Capture"* ]]
}

detect_usb_webcam_device() {
  local current_card="" line dev

  while IFS= read -r line; do
    if [[ "$line" =~ ^[^[:space:]/].*:$ ]]; then
      current_card="$line"
      continue
    fi

    dev="$(echo "$line" | sed 's/^[[:space:]]*//' | awk '{print $1}')"
    [[ "$dev" == /dev/video* ]] || continue

    if [[ "$current_card" == *"GoPro"* ]] \
      || [[ "$current_card" == *"loopback"* ]] \
      || [[ "$current_card" == *"Integrat"* ]]; then
      continue
    fi

    if device_ready "$dev"; then
      echo "$dev"
      return 0
    fi
  done < <(v4l2-ctl --list-devices 2>/dev/null)

  return 1
}

resolve_video_device() {
  local input_source="${1:-gopro}"

  case "$input_source" in
    gopro)
      echo "${VIDEO_DEVICE:-/dev/video42}"
      ;;
    webcam)
      if [[ -n "${VIDEO_DEVICE:-}" ]]; then
        echo "$VIDEO_DEVICE"
        return 0
      fi
      detect_usb_webcam_device
      ;;
    *)
      echo "Unknown input source: $input_source (use gopro or webcam)" >&2
      return 1
      ;;
  esac
}
