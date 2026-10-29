#!/bin/sh
# Muse Glimmer (open weights, 30B) on this Mac: the answer path with no internet, and the map
# review's model when there is no key. scripts/glimmer.sh [port] | install | uninstall
# Weights: hf download meta-models/Muse-Glimmer-30B-GGUF --local-dir ~/helloworld/models/muse-glimmer \
#   --include "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"   (17 GB; holds ~16 GB of memory while up)
# On by default (providers.offline = "glimmer"): on the 64 GB mini, free memory went from 92% to
# 68% with it loaded next to the service, far above the 10% floor.
set -e
cd "$(dirname "$0")/.."
LABEL=com.pranavachar.muse-glimmer
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
M="${GLIMMER_DIR:-$HOME/helloworld/models/muse-glimmer}/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
case "$1" in
install)
  cat >"$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$PWD/scripts/glimmer.sh</string></array>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/muse-glimmer.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/muse-glimmer.log</string>
</dict></plist>
PL
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  while launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; do sleep 1; done
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  ;;
uninstall)
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  ;;
*)
  exec nice -n 19 taskpolicy -b llama-server -m "$M" --host 127.0.0.1 --port "${1:-8091}" \
    -c 8192 -ngl 99 --jinja -a muse-glimmer
  ;;
esac
