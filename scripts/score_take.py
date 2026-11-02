"""Score a take's belt cues: what the four motors would do, from the server's safety messages.

Reads a cue log (the live take's data/demo-takes/*-cues.jsonl, or judge_replay.py --record) and
prints, per motor of the built belt (front-right, back-right, back-left, front-left; a motor
is on when its own diagonal cell or a cardinal cell beside it is on): the on fraction, the
solid fraction, and TRANSIENT buzzes per minute (on, then off again within 0.5 s: the
flicker). REAR is either back motor on. In a room take the walls behind stay put while the
wearer faces forward, so rear buzzes are mostly false; the take has no ground truth, so these
are proxies. Guide: frames that carried a direction, frames where a motor actually played the guide
taps (an obstacle cue hides them), and whether "found" fired.

    .venv/bin/python scripts/score_take.py LOG [LOG ...] [--json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MOTORS = ["front-right", "back-right", "back-left", "front-left"]
# cells 0..7 clockwise from front; each motor's own diagonal and the two cardinals beside it
CELLS = {0: (1, 0, 2), 1: (3, 2, 4), 2: (5, 4, 6), 3: (7, 6, 0)}
CUE = {"slow", "fast", "solid", "guide", "find", "talk"}
TRANSIENT_S = 0.5


def score(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    safety = [r for r in rows if r.get("type") == "safety"]
    if not safety:
        return {"log": path.name, "frames": 0}
    t = [r["wall"] for r in safety]
    minutes = max(1e-6, (t[-1] - t[0]) / 60)
    on = [[False] * 4 for _ in safety]
    solid = [[False] * 4 for _ in safety]
    for k, r in enumerate(safety):
        modes = [c["mode"] for c in r["cues"]]
        for m, cells in CELLS.items():
            on[k][m] = any(modes[c] in CUE for c in cells)
            solid[k][m] = any(modes[c] == "solid" for c in cells)
    out: dict = {"log": path.name, "frames": len(safety), "seconds": round(t[-1] - t[0], 1)}
    for m, name in enumerate(MOTORS):
        col = [row[m] for row in on]
        starts, transients = None, 0
        for k, v in enumerate(col):
            if v and (k == 0 or not col[k - 1]):
                starts = t[k]
            if not v and k and col[k - 1] and starts is not None and t[k] - starts < TRANSIENT_S:
                transients += 1
        out[name] = {
            "on": round(sum(col) / len(col), 3),
            "solid": round(sum(row[m] for row in solid) / len(col), 3),
            "transient_per_min": round(transients / minutes, 1),
        }
    out["rear_on"] = round(sum(row[1] or row[2] for row in on) / len(on), 3)
    out["front_solid"] = round(sum(row[0] and row[3] for row in solid) / len(on), 3)
    guided = [r["guide"] for r in safety if r.get("guide")]
    out["guide_frames"] = len(guided)
    out["guide_pointing"] = sum(g.get("bearing") is not None for g in guided)
    out["guide_in_view"] = sum(bool(g.get("in_view")) for g in guided)
    out["guide_felt"] = sum(any(c["mode"] == "guide" for c in r["cues"]) for r in safety)
    out["found"] = any(r.get("found") for r in safety)
    out["answers"] = [r["text"] for r in rows if r.get("type") == "answer"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    for p in a.logs:
        s = score(p)
        if a.json:
            print(json.dumps(s))
            continue
        print(f"{s['log']}: {s['frames']} frames, {s.get('seconds')} s")
        for name in MOTORS:
            if name in s:
                print(f"  {name:12s} {s[name]}")
        for k in (
            "rear_on",
            "front_solid",
            "guide_frames",
            "guide_pointing",
            "guide_in_view",
            "guide_felt",
        ):
            print(f"  {k}: {s.get(k)}")
        print(f"  found: {s.get('found')}  answers: {s.get('answers')}")


if __name__ == "__main__":
    main()
