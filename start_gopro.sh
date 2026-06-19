#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOPRO_DIR="$PROJECT_DIR/gopro_as_webcam_on_linux"
RESOLUTION="${RESOLUTION:-1080}"
FOV="${FOV:-narrow}"

if [[ "$RESOLUTION" != "1080" && "$RESOLUTION" != "720" ]]; then
  echo "RESOLUTION must be 1080 or 720 (got: $RESOLUTION)" >&2
  exit 1
fi

preflight_gopro_usb() {
  local iface

  iface="$(ip -br link 2>/dev/null | awk '/^enx/ {print $1; exit}')"
  if [[ -n "${iface}" ]]; then
    return 0
  fi

  echo "No GoPro USB network interface found (expected an enx* device)." >&2
  echo "" >&2

  if ! lsusb -d 2672: >/dev/null 2>&1; then
    echo "Linux does not see a GoPro on USB at all." >&2
    echo "Before rerunning ./start_gopro.sh:" >&2
    echo "  1. Power on the GoPro." >&2
    echo "  2. Use a USB-C data cable, not a charge-only cable." >&2
    echo "  3. Set Preferences -> Connections -> USB Connection -> GoPro Connect." >&2
    echo "  4. Plug in, wait for the camera to finish booting, then check:" >&2
    echo "       ip -4 addr show | grep -A2 enx" >&2
    echo "       lsusb | grep -i gopro" >&2
  else
    echo "A GoPro is visible on USB, but the enx* network interface is missing." >&2
    echo "Try unplugging and replugging the cable after the camera is on." >&2
    echo "If it still fails, reset connections on the GoPro and reboot the camera." >&2
  fi

  return 1
}

cd "$GOPRO_DIR"

ARGS=(
  webcam
  --auto-start
  --non-interactive
  --resolution "$RESOLUTION"
  --fov "$FOV"
)

# Prefer auto-discovery on the GoPro USB interface (enx*). Override with GOPRO_HOST_IP
# only when you need a fixed address.
if [[ -n "${GOPRO_HOST_IP:-}" ]]; then
  preflight_gopro_usb
  ARGS+=(--ip "$GOPRO_HOST_IP")
else
  preflight_gopro_usb
  ARGS+=(--device-pattern enx)
fi

exec sudo ./gopro "${ARGS[@]}"
