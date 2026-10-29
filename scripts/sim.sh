#!/bin/sh
# Run the glasses app in the iOS Simulator with a mock Meta Ray-Ban Display:
#   scripts/sim.sh [walk-dir]
# Encodes a saved walk (default: the newest data/walks/*-iphone) to HEVC as the glasses
# camera, builds, installs and launches with --metasight-demo, so frames go to the local
# server (scripts/service.sh) and its cues come back to the glasses' display. The display
# shows in a square on the app's camera screen and in Chrome at http://127.0.0.1:9000/ with
# the "Meta Ray-Ban Display Simulator" extension. Nothing plays sound in the Simulator.
# Walk footage stays under ios/build (git-ignored); it is never committed.
# It runs on its own simulator, "Meta Sight" (an iPhone 17, created on first use), so other
# projects' simulators never get its frames or screenshots. Screenshot it with:
#   xcrun simctl io "$(cat ios/build/sim/udid)" screenshot ~/Pictures/Screenshots/NAME.png
# Env: SIM_DEVICE (default "Meta Sight"), METASIGHT_DEMO_SERVER (default ws://127.0.0.1:8445/ws),
#      NO_CHROME=1 to skip opening the preview tab, SIM_ARGS="--metasight-large-display" for a big in-app display.
set -e
cd "$(dirname "$0")/.."
ROOT=$PWD
DEVICE=${SIM_DEVICE:-Meta Sight}
SERVER=${METASIGHT_DEMO_SERVER:-ws://127.0.0.1:8445/ws}
BUNDLE=com.pranavachar.metasight
OUT=$ROOT/ios/build/sim

WALK=${1:-$(ls -d data/walks/*-iphone 2>/dev/null | tail -1)}
[ -d "$WALK/frames" ] || { echo "no walk with frames: $WALK" >&2; exit 1; }
WALK=$(cd "$WALK" && pwd)
NAME=$(basename "$WALK")
VIDEO=$OUT/$NAME.mp4
mkdir -p "$OUT"

# Frame rate and lens from the walk itself.
eval "$(python3 -c '
import json, sys
walk = sys.argv[1]
rows = [json.loads(l) for l in open(walk + "/frames.jsonl")]
span = (rows[-1]["ts"] - rows[0]["ts"]) / 1000
cam = json.load(open(walk + "/camera.json"))
print("FPS=%.2f HFOV=%s" % (len(rows) / span, cam.get("hfov", "")))
' "$WALK")"

if [ ! -s "$VIDEO" ]; then
  # Heavy-job slot (~/helloworld/.heavy-lock*), released when the encode ends.
  n=1
  while :; do
    slot=$HOME/helloworld/.heavy-lock; [ $n -gt 1 ] && slot=$slot-$n
    if mkdir "$slot" 2>/dev/null; then break; fi
    owner=$(cat "$slot/owner" 2>/dev/null || echo)
    if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then rm -r "$slot"; continue; fi
    n=$((n + 1))
  done
  echo $$ >"$slot/owner"
  trap 'rm -r "$slot"' EXIT
  echo "encoding $NAME at $FPS fps -> $VIDEO"
  nice -n 19 taskpolicy -b ffmpeg -loglevel error -y -framerate "$FPS" -pattern_type glob \
    -i "$WALK/frames/*.jpg" -c:v hevc_videotoolbox -tag:v hvc1 -b:v 4M -pix_fmt yuv420p "$VIDEO"
  rm -r "$slot"
  trap - EXIT
fi

UDID=$(xcrun simctl list devices available -j | python3 -c "
import json, sys
name = sys.argv[1]
devs = [d for rt in json.load(sys.stdin)['devices'].values() for d in rt if d['name'] == name]
print(devs[-1]['udid'] if devs else '')" "$DEVICE")
if [ -z "$UDID" ]; then
  [ -n "$SIM_DEVICE" ] && { echo "no simulator named $DEVICE" >&2; exit 1; }
  UDID=$(xcrun simctl create "$DEVICE" "iPhone 17")
fi
echo "$UDID" >"$OUT/udid"
xcrun simctl boot "$UDID" 2>/dev/null || true
# The Simulator window is optional; the device runs headless when Xcode ships without it.
SIMAPP="$(xcode-select -p)/Applications/Simulator.app"
[ -d "$SIMAPP" ] && open -a "$SIMAPP" --args -CurrentDeviceUDID "$UDID"

echo "building for $DEVICE"
xcodebuild -project ios/CameraAccess.xcodeproj -scheme CameraAccess -configuration Debug \
  -destination "id=$UDID" -derivedDataPath "$OUT/dd" -quiet build
APP=$OUT/dd/Build/Products/Debug-iphonesimulator/CameraAccess.app

xcrun simctl terminate "$UDID" "$BUNDLE" 2>/dev/null || true
xcrun simctl install "$UDID" "$APP"
SIMCTL_CHILD_METASIGHT_DEMO_VIDEO=$VIDEO SIMCTL_CHILD_METASIGHT_DEMO_HFOV=$HFOV \
  SIMCTL_CHILD_METASIGHT_DEMO_SERVER=$SERVER \
  xcrun simctl launch "$UDID" "$BUNDLE" --metasight-demo $SIM_ARGS

if [ -z "$NO_CHROME" ]; then
  sleep 4
  open -a "Google Chrome" http://127.0.0.1:9000/
fi
echo "running: $NAME as the glasses camera -> $SERVER; display preview at http://127.0.0.1:9000/"
