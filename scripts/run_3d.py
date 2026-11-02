"""The belt in 3D rooms: MuJoCo-rendered depth through the live safety geometry (hapsim3d).

SIMULATED USER MODEL, NOT A HUMAN STUDY. Same held-out rooms (1000-1099) as the 2D ledger, now
with furniture heights and tables. Arms: the current best (panned, four motors, constant-power
fold), the eight-motor belt and the live sector scheme on four motors, plus the best in 2D for
the perception gap. Paired bootstrap 95% on cost. Writes bench/results/sim3d.json.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_3d.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from metasight import hapsim as h
from metasight import pan

ROOT = Path(__file__).resolve().parent.parent
HELD = list(range(1000, 1100))
BEST = dict(scheme="panned", layout="diag4", fold="cp")
ARMS = {
    "3d panned diag4-cp": (BEST, True),
    "3d panned eight": (dict(scheme="panned", layout="eight"), True),
    "3d sector diag4": (dict(scheme="sector", layout="diag4", fold="max3"), True),
    "2d panned diag4-cp": (BEST, False),
}
PAIRS = [
    ("3d panned diag4-cp", "2d panned diag4-cp"),
    ("3d panned diag4-cp", "3d panned eight"),
    ("3d sector diag4", "3d panned diag4-cp"),
]


def _one(a):
    kw, three_d, seed = a
    room = h.make_room(seed)
    belt = h.Belt(mapping=pan.DEFAULT, persist=2, **kw)
    if not three_d:
        return h.run(room, belt, h.DEFAULT_SKIN, seed=0)
    from metasight.hapsim3d import Sensor3D

    s = Sensor3D(room)
    try:
        return h.run(room, belt, h.DEFAULT_SKIN, seed=0, sensor=s)
    finally:
        s.close()


def _walk_cost(r: h.Result) -> float:
    return 100 * (not r.success) + 4 * r.collisions + 0.05 * r.time_s


def _paired(a, b) -> tuple[float, float, float]:
    d = np.array([_walk_cost(y) - _walk_cost(x) for x, y in zip(a, b, strict=True)])
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, d.size).mean() for _ in range(2000)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    out, runs = {}, {}
    with ProcessPoolExecutor(max(2, (os.cpu_count() or 4) - 2)) as pool:
        for name, (kw, three_d) in ARMS.items():
            rs = list(pool.map(_one, [(kw, three_d, s) for s in HELD]))
            runs[name] = rs
            m = out[name] = h.summarize(rs)
            print(
                f"{name:20s} success {m['success']:.2f}  collisions {m['collisions']:.2f}  "
                f"cost {m['cost']:5.1f}",
                flush=True,
            )
    for arm, ref in PAIRS:
        d, lo, hi = _paired(runs[ref], runs[arm])
        out[f"{arm} minus {ref}"] = {"cost_diff": d, "ci95": [lo, hi]}
        print(f"{arm} minus {ref}: cost {d:+.1f} (95% {lo:+.1f} to {hi:+.1f})")
    (ROOT / "bench" / "results" / "sim3d.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
