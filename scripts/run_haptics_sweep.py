"""Tune the belt's haptic map against the simulated blind walker (metasight/hapsim.py).

SIMULATED USER MODEL, NOT A HUMAN STUDY. Stages:
  1. one-factor sweeps around the current defaults (tune rooms)
  2. random search over the combined space (tune rooms)
  3. the winner, the old sector scheme and the defaults on held-out rooms, with paired
     bootstrap intervals, then sensitivity to latency and to a noisier skin
Writes bench/results/haptics-sim.json and the plots in docs/img/.

    PYTHONPATH=. nice -n 19 .venv/bin/python scripts/run_haptics_sweep.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from metasight import hapsim as h
from metasight import pan

ROOT = Path(__file__).resolve().parent.parent
TUNE = list(range(60))
HELD = list(range(1000, 1100))
D = pan.DEFAULT

SPACE = {
    "off_m": [1.5, 2.0, 2.5, 3.0],
    "fast_m": [0.8, 1.0, 1.3, 1.6],
    "solid_m": [0.4, 0.5, 0.7],
    "floor": [0.2, 0.35, 0.5],
    "gamma": [0.7, 1.0, 1.5, 2.0],
    "pan": ["constant_power", "linear", "nearest"],
    "k_targets": [1, 2, 3],
    "sep_deg": [45.0, 60.0, 90.0],
}
BELT_SPACE = {"slow_hz": [1.0, 1.5, 2.5], "fast_hz": [3.0, 4.0, 6.0]}


def make(mp: dict, bp: dict, scheme: str = "panned", **kw) -> h.Belt:
    return h.Belt(mapping=replace(D, **mp), scheme=scheme, **bp, **kw)


def _one(args):
    belt, skin, seed = args
    return h.run(h.make_room(seed), belt, skin, seed=0)


def score(pool, belt, skin, seeds):
    return list(pool.map(_one, [(belt, skin, s) for s in seeds], chunksize=2))


def per_run_cost(rs):
    return np.array([100 * (not r.success) + 4 * r.collisions + 0.05 * r.time_s for r in rs])


def paired_ci(a, b, n=2000, seed=0):
    """Mean of a - b with a 95% bootstrap interval over rooms."""
    d = per_run_cost(a) - per_run_cost(b)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tune, held = (TUNE[:12], HELD[:20]) if a.quick else (TUNE, HELD)
    n_random = 12 if a.quick else 140
    skin = h.DEFAULT_SKIN
    t0 = time.time()
    out: dict = {"label": "simulated user model, not a human study", "tune_rooms": len(tune),
                 "held_rooms": len(held), "oneway": {}}  # fmt: skip
    base_mp, base_bp = {}, {}
    with ProcessPoolExecutor(a.workers) as pool:
        # 1. one-factor sweeps
        for name, vals in {**SPACE, **BELT_SPACE}.items():
            rows = []
            for v in vals:
                mp, bp = dict(base_mp), dict(base_bp)
                (mp if name in SPACE else bp)[name] = v
                rs = score(pool, make(mp, bp), skin, tune)
                rows.append({"value": v, **h.summarize(rs)})
            out["oneway"][name] = rows
            print(name, [(r["value"], round(r["cost"], 1)) for r in rows], flush=True)
        # 2. random search, seeded by the best single value of every factor
        rng = random.Random(7)
        cands = [({}, {})]
        for _ in range(n_random):
            cands.append(
                (
                    {k: rng.choice(v) for k, v in SPACE.items()},
                    {k: rng.choice(v) for k, v in BELT_SPACE.items()},
                )
            )
        best_each = {
            n: min(rows, key=lambda r: r["cost"])["value"] for n, rows in out["oneway"].items()
        }
        cands.append(
            (
                {k: v for k, v in best_each.items() if k in SPACE},
                {k: v for k, v in best_each.items() if k in BELT_SPACE},
            )
        )
        scored = []
        for mp, bp in cands:
            s = h.summarize(score(pool, make(mp, bp), skin, tune))
            scored.append({"mapping": mp, "belt": bp, **s})
        scored.sort(key=lambda r: r["cost"])
        out["search"] = scored[:15]
        out["search_n"] = len(scored)
        print("best tune", scored[0], flush=True)
        # re-score the top 5 on fresh tune rooms so the winner is not just the luckiest
        fresh = list(range(2000, 2000 + len(tune)))
        top = scored[:5]
        for r in top:
            r["fresh_cost"] = h.summarize(score(pool, make(r["mapping"], r["belt"]), skin, fresh))[
                "cost"
            ]
        win = min(top, key=lambda r: (r["cost"] + r["fresh_cost"]) / 2)
        out["winner"] = {"mapping": win["mapping"], "belt": win["belt"]}
        wb = make(win["mapping"], win["belt"])
        # 3. held-out comparison
        arms = {
            "winner": wb,
            "defaults (panned)": make({}, {}),
            "old sector scheme": make({}, {}, scheme="sector"),
        }
        res = {k: score(pool, b, skin, held) for k, b in arms.items()}
        out["held"] = {k: h.summarize(v) for k, v in res.items()}
        out["held_vs_sector"] = paired_ci(res["winner"], res["old sector scheme"])
        out["held_vs_defaults"] = paired_ci(res["winner"], res["defaults (panned)"])
        out["held_defaults_vs_sector"] = paired_ci(
            res["defaults (panned)"], res["old sector scheme"]
        )
        print(json.dumps(out["held"], indent=1), flush=True)
        # sensitivity: the winner and the old scheme as the assumptions get worse
        sens = []
        for label, kw, sk in [
            ("latency 50 ms", {"latency_s": 0.05}, skin),
            ("latency 80 ms (assumed)", {}, skin),
            ("latency 150 ms", {"latency_s": 0.15}, skin),
            ("latency 250 ms", {"latency_s": 0.25}, skin),
            ("spin-up 50 ms", {"spinup_s": 0.05}, skin),
            ("spin-up 100 ms", {"spinup_s": 0.10}, skin),
            ("JND 30%", {}, replace(skin, jnd=0.30)),
            ("localisation noise 12 deg", {}, replace(skin, loc_sd_deg=12.0)),
            ("detection floor 0.2", {}, replace(skin, floor=0.20)),
        ]:
            row = {"case": label}
            for nm, b in (("winner", wb), ("sector", make({}, {}, scheme="sector"))):
                row[nm] = h.summarize(score(pool, replace(b, **kw), sk, held))
            sens.append(row)
            print(
                label, round(row["winner"]["cost"], 1), round(row["sector"]["cost"], 1), flush=True
            )
        out["sensitivity"] = sens
    out["seconds"] = round(time.time() - t0)
    (ROOT / "bench" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "bench" / "results" / "haptics-sim.json").write_text(
        json.dumps(out, indent=1, default=str)
    )
    print("done in", out["seconds"], "s")
    if not a.quick:
        print(asdict(replace(D, **out["winner"]["mapping"])), out["winner"]["belt"])


if __name__ == "__main__":
    main()
