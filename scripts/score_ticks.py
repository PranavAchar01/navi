"""Score the hazard beeps the phone would play over recorded takes, and try onset gates.

REAL RECORDED TAKES, REPLAYED; NOT A HUMAN STUDY. The takes have no ground truth, so false
beeps are proxies:

- flicker: beeps inside a warning that lasted under 0.5 s from start to all-clear (nothing in
  a bedroom appears and vanishes that fast; it is the detector or the depth flickering);
- rear: beeps from the three sectors behind the wearer (in a room take the wearer faces where
  they walk, so a hazard behind is mostly the map's walls).

Cost of a gate: the delay from a warning's start to its first beep, over warnings that lasted
at least 1 s (real ones), and the delay to the first Stop beep after the cue first said Stop.

    PYTHONPATH=. .venv/bin/python scripts/score_ticks.py [LOG ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from metasight import spatial_sound as ss

LABEL = "REAL RECORDED TAKES, REPLAYED; NOT A HUMAN STUDY"
FLICKER_S = 0.5
REAL_S = 1.0
TAKES = [
    "data/demo-takes/20261102-030143-cues.jsonl",
    "bench/results/demo-1102/strong.jsonl",
    "bench/results/demo-1102/graded.jsonl",
    "bench/results/demo-1102/base.jsonl",
    "bench/results/demo-1102/room.jsonl",
]
GATES = {
    "none": None,
    "stop 0.05 / fast 0.1 / slow 0.2": {"stop": 0.05, "fast": 0.1, "slow": 0.2},
    "stop 0.1 / fast 0.2 / slow 0.3": {"stop": 0.1, "fast": 0.2, "slow": 0.3},
    "phone display (0.15 / 0.3 / 0.5)": {"stop": 0.15, "fast": 0.3, "slow": 0.5},
    "0.25 / 0.5 / 0.8": {"stop": 0.25, "fast": 0.5, "slow": 0.8},
}


def raw_episodes(msgs: list[dict], t0: float) -> tuple[list[tuple[float, float]], list[float]]:
    """Warning runs on the 40 ms grid as (start, end) seconds from t0, and the times the raw
    warning first turned Stop in each run."""
    safety = [m for m in msgs if m.get("type") == "safety"]
    walls = np.array([m["wall"] - t0 for m in safety])
    runs, stops, cur, was_stop = [], [], None, False
    for t in np.arange(walls[0], walls[-1] + ss.STEP_S, ss.STEP_S):
        k = int(np.searchsorted(walls, t, side="right") - 1)
        w = None if t - walls[k] > ss.STALE_S else ss.take_warning(safety[k])
        if w is not None and cur is None:
            cur = float(t)
        if w is None and cur is not None:
            runs.append((cur, float(t)))
            cur = None
        is_stop = w is not None and w[2] == "stop"
        if is_stop and not was_stop:
            stops.append(float(t))
        was_stop = is_stop
    if cur is not None:
        runs.append((cur, float(walls[-1])))
    return runs, stops


def score(msgs: list[dict], rise: dict[str, float] | None) -> dict:
    t0 = min(m["wall"] for m in msgs)
    ev = list(ss.schedule_take(msgs, beacon=False, rise=rise))
    runs, stops = raw_episodes(msgs, t0)
    safety_t = [m["wall"] - t0 for m in msgs if m.get("type") == "safety"]
    minutes = max(1e-6, (safety_t[-1] - safety_t[0]) / 60)
    beeps = np.array([e.t for e in ev])

    def inside(a: float, b: float) -> np.ndarray:
        return beeps[(beeps >= a - 1e-9) & (beeps <= b + 1e-9)]

    flicker = sum(len(inside(a, b)) for a, b in runs if b - a < FLICKER_S)
    rear = sum(abs(e.where.az_deg) > 112.5 for e in ev)
    lat = []
    for a, b in runs:
        if b - a >= REAL_S:
            got = inside(a, b)
            lat.append(got[0] - a if len(got) else math.inf)
    stop_beeps = np.array([e.t for e in ev if e.kind == "stop"])
    stop_lat = []
    for s in stops:
        later = stop_beeps[stop_beeps >= s - 1e-9]
        stop_lat.append(later[0] - s if len(later) else math.inf)
    finite = [x for x in lat if math.isfinite(x)]
    stop_finite = [x for x in stop_lat if math.isfinite(x) and x < 1.0]
    return {
        "minutes": round(minutes, 2),
        "beeps_per_min": round(len(ev) / minutes, 1),
        "flicker_per_min": round(flicker / minutes, 1),
        "rear_share": round(rear / max(1, len(ev)), 2),
        "real_warnings": len(lat),
        "real_missed": sum(not math.isfinite(x) for x in lat),
        "onset_median_s": round(float(np.median(finite)), 2) if finite else None,
        "stop_onsets": len(stops),
        "stop_beep_median_s": round(float(np.median(stop_finite)), 2) if stop_finite else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="*", default=TAKES)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out: dict = {"label": LABEL}
    print(LABEL)
    for log in a.logs:
        msgs = [json.loads(x) for x in Path(log).read_text().splitlines() if x.strip()]
        print(f"\n{log}")
        out[log] = {}
        for name, rise in GATES.items():
            s = score(msgs, rise)
            out[log][name] = s
            print(
                f"  {name:34s} {s['beeps_per_min']:5.1f}/min"
                f"  flicker {s['flicker_per_min']:4.1f}/min  rear {s['rear_share']:.2f}"
                f"  onset {s['onset_median_s']}s"
                f" ({s['real_missed']}/{s['real_warnings']} missed)"
                f"  stop beep {s['stop_beep_median_s']}s after {s['stop_onsets']} stops"
            )
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
        print(a.json)


if __name__ == "__main__":
    main()
