#!/usr/bin/env bash
# Compile and flash the belt ESP32 (classic 38-pin DevKitC, CP2102 USB-C) from this Mac.
#   scripts/flash_belt.sh            belt firmware (BLE; Wi-Fi too if firmware/belt/secrets.h exists)
#   scripts/flash_belt.sh bringup    wiring check: each motor buzzes once, front then clockwise
#   scripts/flash_belt.sh compile    compile only, nothing plugged in
# Optional: PORT=/dev/cu.usbserial-0001 to skip the port search; MONITOR=1 to open the serial
# monitor afterwards. One-time setup is in docs/BUY-plug-and-play.md.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
what="${1:-belt}"
# huge_app: BLE plus Wi-Fi together does not fit the default 1.3 MB app partition
fqbn="esp32:esp32:esp32:PartitionScheme=huge_app"

command -v arduino-cli >/dev/null || { echo "arduino-cli missing: brew install arduino-cli" >&2; exit 1; }
arduino-cli core list | grep -q '^esp32:esp32' || {
  echo "esp32 core missing: arduino-cli core update-index && arduino-cli core install esp32:esp32" >&2
  exit 1
}

case "$what" in
  belt | compile) sketch="$here/firmware/belt" ;;
  bringup) sketch="$here/firmware/belt-bringup" ;;
  *) echo "usage: $0 [belt|bringup|compile]" >&2; exit 2 ;;
esac

if [[ -f "$sketch/secrets.h" ]]; then
  echo "secrets.h found: building with Wi-Fi UDP as well as Bluetooth"
else
  echo "no secrets.h: building Bluetooth only"
fi

build="${TMPDIR:-/tmp}/metasight-$(basename "$sketch")-build"
arduino-cli compile --fqbn "$fqbn" --build-path "$build" "$sketch"
[[ "$what" == compile ]] && exit 0

port="${PORT:-}"
if [[ -z "$port" ]]; then
  shopt -s nullglob
  ports=(/dev/cu.usbserial-* /dev/cu.SLAB_USBtoUART* /dev/cu.wchusbserial*)
  if ((${#ports[@]} == 0)); then
    echo "no ESP32 serial port found. Plug in the board with a data cable (not charge-only)." >&2
    echo "If it still does not show: install the Silicon Labs CP210x driver, then re-run." >&2
    exit 1
  elif ((${#ports[@]} > 1)); then
    echo "several serial ports: ${ports[*]}; pick one with PORT=..." >&2
    exit 1
  fi
  port="${ports[0]}"
fi
echo "uploading to $port"
# If the upload hangs at "Connecting....", hold BOOT on the board until it starts writing.
arduino-cli upload --fqbn "$fqbn" -p "$port" --input-dir "$build"
echo "flashed $what"
[[ "${MONITOR:-0}" == 1 ]] && exec arduino-cli monitor -p "$port" -c baudrate=115200
exit 0
