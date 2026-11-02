"""Retune the haptic map (pan.Params plus pulse rates) for the built four-motor belt.

SIMULATED USER MODEL, NOT A HUMAN STUDY. pan.DEFAULT was tuned on eight motors
(docs/haptics-sim.md); this repeats the search with layout diag4, persist 2:
  1. one-factor sweeps around the defaults (tune rooms 0-59)
  2. random search over the joint space (tune rooms), plus the best of each factor combined
  3. the best on tune rooms against the defaults on held-out rooms 1000-1099, paired bootstrap
Writes bench/results/tune-diag4.json.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_tune_diag4.py
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from metasight import hapsim as h
from metasight import pan

ROOT = Path(__file__).resolve().parent.parent
TUNE = list(range(60))
HELD = list(range(1000, 1100))
N_RANDOM = 48

SPACE = {
    "off_m": [1.2, 1.5, 2.0, 2.5],
    "fast_m": [0.8, 1.0, 1.3, 1.6],
    "solid_m": [0.5, 0.7, 0.9],
    "floor": [0.2, 0.35, 0.5],
    "gamma": [1.0, 1.5, 2.0],
    "pan": ["constant_power", "linear", "nearest"],
    "k_targets": [1, 2],
}
BELT_SPACE = {"fast_hz": [3.0, 4.0, 6.0]}


def make(mp: dict, bp: dict) -> h.Belt:
    return h.Belt(
        mapping=replace(pan.DEFAULT, **mp), scheme="panned", layout="diag4", persist=2, **bp
    )


def _one(a):
    belt, seed = a
    return h.run(h.make_room(seed), belt, h.DEFAULT_SKIN, seed=0)


def runs(pool, belt, seeds):
    return list(pool.map(_one, [(belt, s) for s in seeds], chunksize=2))


def walk_cost(r) -> float:
    return 100 * (not r.success) + 4 * r.collisions + 0.05 * r.time_s


def paired(a, b):
    d = np.array([walk_cost(y) - walk_cost(x) for x, y in zip(a, b, strict=True)])
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, d.size).mean() for _ in range(2000)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    out: dict = {"one_factor": {}, "random": []}
    best_each: dict = {}
    with ProcessPoolExecutor(max(2, (os.cpu_count() or 4) - 2)) as pool:
        base = h.cost(runs(pool, make({}, {}), TUNE))
        print(f"defaults on tune rooms: cost {base:.1f}", flush=True)
        for name, vals in {**SPACE, **BELT_SPACE}.items():
            row = {}
            for v in vals:
                mp, bp = ({name: v}, {}) if name in SPACE else ({}, {name: v})
                row[str(v)] = h.cost(runs(pool, make(mp, bp), TUNE))
            out["one_factor"][name] = row
            best_each[name] = min(vals, key=lambda v, r=row: r[str(v)])
            print(f"{name:10s} " + "  ".join(f"{k}: {c:.1f}" for k, c in row.items()), flush=True)
        rng = random.Random(0)
        cands = [
            (
                {k: v for k, v in best_each.items() if k in SPACE},
                {k: v for k, v in best_each.items() if k in BELT_SPACE},
            )
        ]
        cands += [
            (
                {k: rng.choice(v) for k, v in SPACE.items()},
                {k: rng.choice(v) for k, v in BELT_SPACE.items()},
            )
            for _ in range(N_RANDOM)
        ]
        scored = []
        for mp, bp in cands:
            if mp.get("fast_m", 0) <= mp.get("solid_m", 0):
                continue  # fast pulse must start before solid
            c = h.cost(runs(pool, make(mp, bp), TUNE))
            scored.append((c, mp, bp))
            out["random"].append({"cost": c, **mp, **bp})
        scored.sort(key=lambda x: x[0])
        top = scored[0]
        print(f"best on tune rooms: cost {top[0]:.1f} {top[1]} {top[2]}", flush=True)
        a = runs(pool, make({}, {}), HELD)
        b = runs(pool, make(top[1], top[2]), HELD)
    d, lo, hi = paired(a, b)
    out["held"] = {
        "defaults": h.summarize(a),
        "tuned": h.summarize(b),
        "tuned_params": {**asdict(replace(pan.DEFAULT, **top[1])), **top[2]},
        "tuned_minus_defaults": {"cost_diff": d, "ci95": [lo, hi]},
    }
    for k in ("defaults", "tuned"):
        m = out["held"][k]
        print(
            f"held {k:8s} success {m['success']:.2f}  collisions {m['collisions']:.2f}  "
            f"cost {m['cost']:5.1f}"
        )
    print(f"tuned minus defaults: {d:+.1f} (95% {lo:+.1f} to {hi:+.1f})")
    (ROOT / "bench" / "results" / "tune-diag4.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
