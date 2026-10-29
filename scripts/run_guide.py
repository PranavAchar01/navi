"""Does the Guide cue help a simulated blind walker who does not know the way, and does
"Stop overrides Guide" matter?

SIMULATED USER MODEL, NOT A HUMAN STUDY. The walker's sense of where the next waypoint is drifts
(dead reckoning, 12 degrees per sqrt(second)) unless a Guide cue is felt, which resets it to the
cue's bearing (6 degrees localisation noise). Arms: no guide; guide with Stop overriding it (the
design); guide without the override (a control, where Guide replaces obstacle cues on its motors).
Writes bench/results/guide-sim.json.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_guide.py
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
    with ProcessPoolExecutor(6) as pool:
        for g in ("none", "on", "no_override"):
            b = h.Belt(mapping=pan.DEFAULT, guide=g, persist=2)
            rs = list(pool.map(_one, [(b, s) for s in HELD], chunksize=2))
            runs[g] = rs
            out[g] = h.summarize(rs)
            m = out[g]
            print(
                f"guide {g:12s} success {m['success']:.2f}  collisions {m['collisions']:.2f}"
                f"  time {m['time_s']:.1f}s  contact {m['contact_s']:.1f}s  cost {m['cost']:.1f}"
            )
    sys_path = Path(__file__).parent
    import sys

    sys.path.insert(0, str(sys_path))
    from run_haptics_sweep import paired_ci

    out["on_vs_none"] = paired_ci(runs["on"], runs["none"])
    out["on_vs_no_override"] = paired_ci(runs["on"], runs["no_override"])
    print("cost diff (on - none) [95% CI]:", [round(x, 1) for x in out["on_vs_none"]])
    print("cost diff (on - no_override):", [round(x, 1) for x in out["on_vs_no_override"]])
    (ROOT / "bench" / "results" / "guide-sim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
