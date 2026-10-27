#!/bin/sh
# Serve the app on the LAN. A phone browser only allows the camera over HTTPS, so pass a
# certificate pair: scripts/run.sh CERT KEY [SESSION]. Without them it serves plain HTTP on 8444
# (fine for a laptop browser on localhost).
set -e
cd "$(dirname "$0")/.."
[ -d web/dist ] || (cd web && pnpm install --silent && pnpm build)
if [ -n "$1" ] && [ -n "$2" ]; then
  exec .venv/bin/python -m metasight.server --host 0.0.0.0 --port 8443 \
    --cert "$1" --key "$2" --session "${3:-house}"
fi
exec .venv/bin/python -m metasight.server --host 127.0.0.1 --port 8444 --session "${1:-house}"
