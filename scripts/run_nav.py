"""Row 3 proof: "take me to my keys" with nav.Navigator's route and Guide pulses, in the simulator.

SIMULATED USER MODEL, NOT A HUMAN STUDY. The goal is the room's far corner (the keys); the
waypoint bearing comes from nav.Navigator on a floor plan drawn from the room, not from the sim's
own path search. The walker's sense of the waypoint drifts (12 degrees per sqrt(second)) unless a
Guide cue is felt. Arms: no Guide, Guide with Stop overriding it (the design). Reports success,
time to goal, collisions, and how often Stop silenced Guide; "violations" counts frames where the
overlay changed a solid motor and must be 0. Writes bench/results/nav-sim.json.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_nav.py
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from metasight import hapsim as h
from metasight import pan

ROOT = Path(__file__).resolve().parent.parent
HELD = list(range(1000, 1100))
SKIN = replace(h.DEFAULT_SKIN, nav_drift_deg=12.0)


def _one(a):
    belt, seed = a
    return h.run(h.make_room(seed), belt, SKIN, seed=0)


def main() -> None:
    out, runs = {}, {}
    with ProcessPoolExecutor(4) as pool:
        for g in ("none", "on"):
            b = h.Belt(mapping=pan.DEFAULT, scheme="sector", guide=g, persist=2, planner=True)
            rs = list(pool.map(_one, [(b, s) for s in HELD], chunksize=2))
            runs[g] = rs
            m = out[g] = h.summarize(rs)
            print(
                f"planner, guide {g:5s} success {m['success']:.2f}  time to goal {m['time_s']:.1f}s"
                f"  collisions {m['collisions']:.2f}  cost {m['cost']:.1f}"
                f"  guide frames {m['guide_frames']}  stop frames {m['stop_frames']}"
                f"  guide silenced {m['stop_overridden']}  violations {m['stop_violations']}",
                flush=True,
            )
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from run_haptics_sweep import paired_ci

    out["on_vs_none"] = paired_ci(runs["on"], runs["none"])
    print("cost diff (guide - none) [95% CI]:", [round(x, 1) for x in out["on_vs_none"]])
    (ROOT / "bench" / "results" / "nav-sim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
