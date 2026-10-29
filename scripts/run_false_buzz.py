"""False buzzes per minute in the simulated walk, for the sweep's winner and the old sector scheme.

SIMULATED USER MODEL, NOT A HUMAN STUDY. A false buzz is a motor turning on while nothing real is
within range of that motor. The simulated depth is nearly noiseless, so the rate is a function of
an assumed depth error: each of the 15 depth strips reads a spurious near value (0.4 to 1.8 m)
with probability `phantom_rate` per frame. Measure the real rate on the bench (tape-measured
depth error, docs/audit) and read the matching row. Needs bench/results/haptics-sim.json.

    PYTHONPATH=. .venv/bin/python scripts/run_false_buzz.py
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from metasight import hapsim as h
from metasight import pan

ROOT = Path(__file__).resolve().parent.parent
RATES = [0.0, 0.002, 0.005, 0.01, 0.03]
HELD = list(range(1000, 1060))


def _one(a):
    belt, seed, rate = a
    return h.run(h.make_room(seed), belt, h.DEFAULT_SKIN, seed=0, phantom_rate=rate)


def main() -> None:
    res = json.loads((ROOT / "bench" / "results" / "haptics-sim.json").read_text())
    w = res["winner"]
    arms = {
        "winner": h.Belt(mapping=replace(pan.DEFAULT, **w["mapping"]), **w["belt"]),
        "old sector scheme": h.Belt(scheme="sector"),
    }
    out = {}
    with ProcessPoolExecutor(6) as pool:
        for rate in RATES:
            for name, b in arms.items():
                rs = list(pool.map(_one, [(b, s, rate) for s in HELD], chunksize=2))
                out[f"{name} @ {rate}"] = h.summarize(rs)
                m = out[f"{name} @ {rate}"]
                print(
                    f"{name:18s} phantom {rate:<6} false/min {m['false_per_min']:5.1f}"
                    f"  success {m['success']:.2f}  collisions {m['collisions']:.2f}",
                    flush=True,
                )
    (ROOT / "bench" / "results" / "false-buzz-sim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
