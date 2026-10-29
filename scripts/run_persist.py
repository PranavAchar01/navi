"""Does a persistence filter on the depth strips fix panning's false buzzes, and what does it cost?

SIMULATED USER MODEL, NOT A HUMAN STUDY. For persist = 1 (off), 2, 3 frames and phantom depth
rates, the sweep winner (panned) and the old sector scheme walk the same held-out rooms; prints
false buzzes per minute, success, collisions and cost. Writes bench/results/persist-sim.json.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_persist.py
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from metasight import hapsim as h
from metasight import pan

ROOT = Path(__file__).resolve().parent.parent
HELD = list(range(1000, 1060))
RATES = [0.0, 0.005, 0.01, 0.03]


def _one(a):
    belt, seed, rate = a
    return h.run(h.make_room(seed), belt, h.DEFAULT_SKIN, seed=0, phantom_rate=rate)


def main() -> None:
    out = {}
    with ProcessPoolExecutor(6) as pool:
        for rate in RATES:
            for scheme in ("panned", "sector"):
                for n in (1, 2, 3):
                    b = h.Belt(mapping=pan.DEFAULT, scheme=scheme, persist=n)
                    rs = list(pool.map(_one, [(b, s, rate) for s in HELD], chunksize=2))
                    m = h.summarize(rs)
                    out[f"{scheme} persist={n} phantom={rate}"] = m
                    print(
                        f"phantom {rate:<6} {scheme:7s} persist {n}  false/min "
                        f"{m['false_per_min']:5.1f}  success {m['success']:.2f}  "
                        f"collisions {m['collisions']:.2f}  cost {m['cost']:5.1f}",
                        flush=True,
                    )
    (ROOT / "bench" / "results" / "persist-sim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
