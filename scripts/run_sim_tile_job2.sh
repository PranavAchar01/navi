#!/bin/sh
# One tile of the reel under a heavy-lock slot: run_sim_tile_job.sh GRID OUT TILE [tile args...]
GRID=$1; OUT=$2; tile=$3; shift 3
cd "$(dirname "$0")/.."
[ -f "$OUT/$tile/.done" ] && exit 0
[ -f "$OUT/$tile.log" ] && exit 0  # started by another runner
while :; do
  free=$(memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
  [ "${free:-0}" -ge 10 ] && break
  sleep 20
done
n=1
while :; do
  slot=$HOME/helloworld/.heavy-lock; [ $n -gt 1 ] && slot=$slot-$n
  if mkdir "$slot" 2>/dev/null; then break; fi
  o=$(cat "$slot/owner" 2>/dev/null || echo)
  if [ -n "$o" ] && ! kill -0 "$o" 2>/dev/null; then rm "$slot/owner" 2>/dev/null; rmdir "$slot" 2>/dev/null; continue; fi
  n=$((n + 1))
done
echo $$ >"$slot/owner"
mkdir -p "$OUT/$tile"
nice -n 19 taskpolicy -b blender -b --factory-startup --python scripts/blender_sim_tile.py -- \
  --grid "$GRID" --tile "$tile" --out "$OUT" "$@" >"$OUT/$tile.log" 2>&1 && touch "$OUT/$tile/.done"
rm "$slot/owner"; rmdir "$slot"
