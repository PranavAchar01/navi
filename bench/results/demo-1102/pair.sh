#!/bin/bash
# pair.sh NAME1 "SETS1" NAME2 "SETS2": two judge replays of tonight's take side by side, then scores
cd "$(dirname "$0")/../../.."
R=bench/results/demo-1102; W=data/walks/20261102-030055-glasses; H=data/rooms/bedroom/home
run() { # name port sets...
  local name=$1 port=$2; shift 2
  local args=(); for s in "$@"; do args+=(--set "$s"); done
  .venv/bin/python scripts/judge_replay.py $W --port $port --belt-port $((port - 4200)) --home $H \
    --ask "where is my chair" --ask-at ${ASK_AT:-56} --braille off "${args[@]}" \
    --record $R/$name.jsonl > $R/$name.out 2>&1
}
P=${PORT:-8450}; run "$1" $P $2 & run "$3" $((P + 1)) $4 & wait
for n in "$1" "$3"; do grep -E "^answer|answer latency" $R/$n.out; done
.venv/bin/python scripts/score_take.py $R/$1.jsonl $R/$3.jsonl
