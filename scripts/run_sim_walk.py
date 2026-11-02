"""Score closed-loop walker arms (metasight.simwalk) on held-out seeds, both belt layouts.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Every arm walks every scenario on the same seeds (the
noise streams are paired), with eight motors and with the four built diagonals. An arm is kept
only when its mean walk cost (hapsim.cost: 100 a failure, 4 a collision, 0.05 a second) drops
against the arm it builds on with a paired 95% bootstrap interval below zero, on both layouts
pooled. Writes bench/results/sim-walk.json.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_sim_walk.py [arms...]
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from metasight import hapsim as h
from metasight import pan
from metasight import simwalk as S

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "results" / "sim-walk.json"
SEEDS = range(1000, 1030)
LAYOUTS = ("eight", "diag4")

# name: (base arm, belt fields, walk options, hypothesis)
ARMS: dict[str, tuple[str | None, dict, dict, str]] = {
    "I0": (None, {}, {"map_sides": False}, "glasses as configured: live strips only, no map sides"),
    "I1": ("I0", {}, {"map_sides": True}, "add the map's side and back sectors"),
    "I2": ("I1", {}, {"map_sides": True, "memory": True}, "add field-of-view memory"),
    "I3": ("I1", {"hold": True}, {"map_sides": True}, "bearing hold on the panned cue"),
    "I4": ("I1", {"persist": 2}, {"map_sides": True}, "strips must hold 2 frames"),
    "I5": ("I1", {"planner": True}, {"map_sides": True}, "guide routed by the app's navigator"),
}


def belt_for(arm: str, layout: str) -> h.Belt:
    fields = ARMS[arm][1]
    return h.Belt(layout=layout, hfov=S.HFOV, mapping=pan.DEFAULT, **fields)


def _one(job) -> dict:
    arm, layout, name, seed = job
    sc = S.scenario(name)
    tr = S.walk(sc, belt_for(arm, layout), seed=seed, record=False, **ARMS[arm][2])
    return {"arm": arm, "layout": layout, "scenario": name, "seed": seed, **tr.metrics()}


def paired(a: list[dict], b: list[dict], key: str = "cost", n: int = 4000) -> tuple:
    ka = {(r["layout"], r["scenario"], r["seed"]): r[key] for r in a}
    d = np.array([r[key] - ka[(r["layout"], r["scenario"], r["seed"])] for r in b])
    rng = np.random.default_rng(0)
    boots = rng.choice(d, (n, len(d))).mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summary(rows: list[dict]) -> dict:
    def agg(rs):
        ok = [r for r in rs if r["success"]]
        return {
            "walks": len(rs),
            "success": round(float(np.mean([r["success"] for r in rs])), 3),
            "collisions": round(float(np.mean([r["collisions"] for r in rs])), 2),
            "min_clear_m": round(float(np.mean([r["min_clear_m"] for r in rs])), 3),
            "time_to_goal_s": round(float(np.mean([r["time_s"] for r in ok])), 2) if ok else None,
            "false_per_min": round(float(np.mean([r["false_per_min"] for r in rs])), 2),
            "flicker_per_min": round(float(np.mean([r["flicker_per_min"] for r in rs])), 1),
            "cost": round(float(np.mean([r["cost"] for r in rs])), 2),
        }

    out = {"all": agg(rows)}
    for lay in LAYOUTS:
        out[lay] = agg([r for r in rows if r["layout"] == lay])
    for name in S.SCENARIOS:
        out[name] = agg([r for r in rows if r["scenario"] == name])
    return out


def main() -> None:
    arms = sys.argv[1:] or list(ARMS)
    old = json.loads(OUT.read_text()) if OUT.exists() else {"rows": []}
    rows = [r for r in old["rows"] if r["arm"] not in arms]
    jobs = [(a, lay, n, s) for a in arms for lay in LAYOUTS for n in S.SCENARIOS for s in SEEDS]
    # spawn: a forked MuJoCo GL context hangs on macOS
    with mp.get_context("spawn").Pool(8) as p:
        rows += p.map(_one, jobs, chunksize=4)
    res: dict = {"seeds": [SEEDS.start, SEEDS.stop - 1], "rows": rows, "arms": {}}
    for a, (base, _, _, hyp) in ARMS.items():
        ra = [r for r in rows if r["arm"] == a]
        if not ra:
            continue
        entry = {"hypothesis": hyp, "base": base, "summary": summary(ra)}
        rb = [r for r in rows if r["arm"] == base]
        if base and rb:
            entry["paired_cost_vs_base"] = [round(v, 2) for v in paired(rb, ra)]
            for lay in LAYOUTS:
                entry[f"paired_cost_vs_base_{lay}"] = [
                    round(v, 2)
                    for v in paired(
                        [r for r in rb if r["layout"] == lay], [r for r in ra if r["layout"] == lay]
                    )
                ]
        res["arms"][a] = entry
        s = entry["summary"]["all"]
        print(
            a,
            hyp,
            {k: s[k] for k in ("success", "collisions", "min_clear_m", "time_to_goal_s")},
            {k: s[k] for k in ("false_per_min", "flicker_per_min", "cost")},
            entry.get("paired_cost_vs_base"),
        )
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
