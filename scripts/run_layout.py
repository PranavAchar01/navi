"""Four motors on the diagonals (the built belt) against the eight-motor design, in the sim.

SIMULATED USER MODEL, NOT A HUMAN STUDY. 100 held-out rooms, persist 2 (the live setting), the
sector scheme (live) and the panned one, each on both layouts. Prints success, collisions, false
buzzes per minute and cost, plus the paired difference diag4 minus eight with a bootstrap 95%
interval. Writes bench/results/layout-sim.json.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_layout.py
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from metasight import hapsim as h
from metasight import pan

ROOT = Path(__file__).resolve().parent.parent
HELD = list(range(1000, 1100))
ARMS = {
    f"{scheme} {layout}": dict(scheme=scheme, layout=layout, fold="max3")
    for scheme in ("sector", "panned")
    for layout in ("eight", "diag4")
}
ARMS["panned diag4-lin"] = dict(scheme="panned", layout="diag4", fold="lin")
ARMS["panned diag4-cp"] = dict(scheme="panned", layout="diag4", fold="cp")
ODD = replace(pan.DEFAULT, cells="odd")
ARMS["panned diag4-odd"] = dict(scheme="panned", layout="diag4", fold="cp", mapping=ODD)
ARMS["panned diag4-odd-cp"] = dict(
    scheme="panned", layout="diag4", fold="cp", mapping=replace(ODD, pan="constant_power")
)


def _one(a):
    belt, seed = a
    return h.run(h.make_room(seed), belt, h.DEFAULT_SKIN, seed=0)


def _walk_cost(r: h.Result) -> float:
    return 100 * (not r.success) + 4 * r.collisions + 0.05 * r.time_s


def _paired(a: list[h.Result], b: list[h.Result]) -> tuple[float, float, float]:
    d = np.array([_walk_cost(y) - _walk_cost(x) for x, y in zip(a, b, strict=True)])
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, d.size).mean() for _ in range(2000)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    out, runs = {}, {}
    with ProcessPoolExecutor(4) as pool:
        for name, kw in ARMS.items():
            b = h.Belt(**{"mapping": pan.DEFAULT, "persist": 2, **kw})
            rs = list(pool.map(_one, [(b, s) for s in HELD], chunksize=2))
            runs[name] = rs
            m = out[name] = h.summarize(rs)
            print(
                f"{name:14s} success {m['success']:.2f}  collisions {m['collisions']:.2f}  "
                f"false/min {m['false_per_min']:4.1f}  cost {m['cost']:5.1f}",
                flush=True,
            )
    pairs = [(f"{s} diag4", f"{s} eight") for s in ("sector", "panned")]
    pairs.append(("panned diag4-lin", "panned diag4"))
    pairs.append(("panned diag4-cp", "panned diag4"))
    pairs += [(f"panned diag4-{x}", "panned diag4-cp") for x in ("odd", "odd-cp")]
    for arm, ref in pairs:
        d, lo, hi = _paired(runs[ref], runs[arm])
        out[f"{arm} minus {ref}"] = {"cost_diff": d, "ci95": [lo, hi]}
        print(f"{arm} minus {ref}: cost {d:+.1f} (95% {lo:+.1f} to {hi:+.1f})")
    (ROOT / "bench" / "results" / "layout-sim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
