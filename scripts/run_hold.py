"""Does bearing hysteresis (pan.Hold) on the panned cue beat the sector scheme in the sim?

SIMULATED USER MODEL, NOT A HUMAN STUDY. Held-out rooms, persist 2, three phantom depth rates;
prints false buzzes per minute, success, collisions and cost for sector, panned and panned+hold.
Writes bench/results/hold-sim.json.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_hold.py
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from metasight import hapsim as h
from metasight import pan

ROOT = Path(__file__).resolve().parent.parent
HELD = list(range(1000, 1060))
RATES = [0.0, 0.01, 0.03]
SCHEMES = {
    "sector": dict(scheme="sector"),
    "panned": dict(scheme="panned"),
    "panned+hold": dict(scheme="panned", hold=True),
}


def _one(a):
    belt, seed, rate = a
    return h.run(h.make_room(seed), belt, h.DEFAULT_SKIN, seed=0, phantom_rate=rate)


def main() -> None:
    out = {}
    with ProcessPoolExecutor(4) as pool:
        for rate in RATES:
            for name, kw in SCHEMES.items():
                b = h.Belt(mapping=pan.DEFAULT, persist=2, **kw)
                rs = list(pool.map(_one, [(b, s, rate) for s in HELD], chunksize=2))
                m = h.summarize(rs)
                out[f"{name} phantom={rate}"] = m
                print(
                    f"phantom {rate:<5} {name:12s} false/min {m['false_per_min']:5.1f}  "
                    f"success {m['success']:.2f}  collisions {m['collisions']:.2f}  "
                    f"cost {m['cost']:5.1f}",
                    flush=True,
                )
    (ROOT / "bench" / "results" / "hold-sim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
