#!/bin/sh
# Every tile of the parallel-walks reel, N at a time (each under a heavy-lock slot):
#   scripts/run_sim_tiles.sh GRID_DIR OUT_DIR N [blender_sim_tile.py args...]
GRID=$1; OUT=$2; N=$3; shift 3
cd "$(dirname "$0")/.."
python3 -c "
import json, sys
for t in json.load(open(sys.argv[1] + '/timeline.json'))['tiles']:
    print(t['id'])" "$GRID" | xargs -P "$N" -I{} sh scripts/run_sim_tile_job.sh "$GRID" "$OUT" {} "$@"
