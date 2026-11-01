#!/bin/sh
# Keep the server running on this Mac for the glasses app: scripts/service.sh install | uninstall | status
# Listens on the LAN at ws://<LocalHostName>.local:8445/ws with the glasses profile and the hybrid
# finder (YOLOE locally, SAM 3.1 on crops; local only without a key). Logs go to ~/Library/Logs/meta-sight.log.
set -e
cd "$(dirname "$0")/.."
LABEL=com.pranavachar.meta-sight
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
case "$1" in
install)
  [ -d web/dist ] || (cd web && pnpm install --silent && pnpm build)
  cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>WorkingDirectory</key><string>$PWD</string>
  <key>ProgramArguments</key><array>
    <string>$PWD/.venv/bin/python</string><string>-m</string><string>metasight.server</string>
    <string>--host</string><string>::</string><string>--port</string><string>8445</string>
    <string>--session</string><string>home</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>METASIGHT_CONFIG</key><string>$PWD/metasight.glasses.toml</string>
    <key>METASIGHT_FINDER</key><string>hybrid</string>
    <key>METASIGHT_HOME</key><string>$PWD/data/rooms/bedroom/home</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/meta-sight.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/meta-sight.log</string>
</dict></plist>
EOF
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  while launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; do sleep 1; done
  launchctl enable "gui/$(id -u)/$LABEL"
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "ws://$(scutil --get LocalHostName).local:8445/ws"
  ;;
uninstall)
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  ;;
status)
  launchctl print "gui/$(id -u)/$LABEL" | grep -E "state|pid|last exit" || true
  curl -s -o /dev/null -w "http %{http_code}\n" http://127.0.0.1:8445/
  ;;
*) echo "usage: $0 install | uninstall | status" >&2; exit 2 ;;
esac
