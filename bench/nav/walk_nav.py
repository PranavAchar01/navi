"""Routes and priors on the real home map (data/home, built from his walks).

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python bench/nav/walk_nav.py

For every saved view pose and every placed pin: does nav find a route, how long is it against
the straight line, and how long do set_goal (once per "take me to") and step (every frame) take.
Then priors on each view, with that view's own depth as the camera reading: how long a frame
costs and how often the map explains what the camera reads (obstacles 0.2 to 1.9 m above
the floor, as the safety loop counts them). The views helped build the map, so
that last number checks consistency, not new-obstacle detection. Writes
bench/results/nav/walk_nav.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metasight import home, nav  # noqa: E402
from metasight.geometry import local_floor  # noqa: E402

OUT = ROOT / "bench" / "results" / "nav"


def yaw_of(pose: np.ndarray) -> float:
    f = pose[:3, 2]
    return math.atan2(f[0], f[2])


def view_thirds(v: home.HomeView, hfov: float) -> list[float]:
    """What the safety loop would read per third of the view: the nearest surface 0.2 to 1.9 m
    above the floor (the floor itself never counts), as horizontal range."""
    p = v.pts.astype(np.float64)[v.mask]
    p = p[np.isfinite(p).all(axis=1)]
    hgt = p[:, 1] - local_floor(p)
    p = p[(hgt > nav.MIN_H_M) & (hgt < nav.MAX_H_M)]
    rel = p - v.pose[:3, 3]
    yaw = yaw_of(v.pose)
    f, r = np.array([math.sin(yaw), 0, math.cos(yaw)]), np.array([-math.cos(yaw), 0, math.sin(yaw)])
    d = np.hypot(rel[:, 0], rel[:, 2])
    b = np.degrees(np.arctan2(rel @ r, rel @ f))
    out = []
    for k in range(3):
        lo = -hfov / 2 + k * hfov / 3
        sel = d[(b >= lo) & (b < lo + hfov / 3) & (d > 0.15)]
        out.append(float(np.percentile(sel, 5)) if sel.size >= 5 else math.inf)
    return out


def hfov_of(v: home.HomeView) -> float:
    """Horizontal field of view from the points themselves: the widest bearing either side."""
    p = v.pts.astype(np.float32)[v.mask]
    rel = p - v.pose[:3, 3]
    x, z = rel @ v.pose[:3, 0], rel @ v.pose[:3, 2]
    a = np.degrees(np.arctan2(x[z > 0.05], z[z > 0.05]))
    return float(np.percentile(a, 99) - np.percentile(a, 1)) if a.size else 60.0


def below_of(v: home.HomeView) -> float:
    """How far below level the bottom of this view looks, from its own points."""
    p = v.pts.astype(np.float32)[v.mask]
    rel = p - v.pose[:3, 3]
    a = np.degrees(np.arctan2(-rel[:, 1], np.hypot(rel[:, 0], rel[:, 2])))
    return float(np.percentile(a, 99)) if a.size else 35.0


def main() -> None:
    h = home.load()
    assert h is not None, "no home map in data/home"
    t = time.perf_counter()
    plan = nav.floor_plan(h.occ)
    build_ms = 1000 * (time.perf_counter() - t)
    pins = [p for p in h.pins if p.get("placed", True)]
    navi = nav.Navigator(plan)
    goal_ms, step_ms, rows = [], [], []
    for p in pins:
        t = time.perf_counter()
        ok = navi.set_goal(p["pos"], p["label"])
        goal_ms.append(1000 * (time.perf_counter() - t))
        for k, v in enumerate(h.views):
            pos, yaw = v.pose[:3, 3], yaw_of(v.pose)
            t = time.perf_counter()
            g = navi.step(pos, yaw) if ok else None
            step_ms.append(1000 * (time.perf_counter() - t))
            straight = math.hypot(p["pos"][0] - pos[0], p["pos"][2] - pos[2])
            rows.append(
                {
                    "pin": p["label"],
                    "view": k,
                    "goal_on_floor": ok,
                    "route": g is not None and math.isfinite(g.route_m),
                    "route_m": g.route_m if g is not None else None,
                    "straight_m": straight,
                }
            )
    pri_ms, known, new = [], 0, 0
    for v in h.views:
        hfov = hfov_of(v)
        cam = view_thirds(v, hfov)
        below = below_of(v)
        for _ in range(20):  # time it like a run of frames
            pri = nav.priors(
                plan, v.pose[:3, 3], yaw_of(v.pose), cam, hfov, speed_mps=0.8, below_deg=below
            )
            pri_ms.append(pri.ms)
        known += sum(k is True for k in pri.known)
        new += sum(k is False for k in pri.known)
    routed = [r for r in rows if r["route"]]
    detour = [r["route_m"] / r["straight_m"] for r in routed if r["straight_m"] > 0.5]
    out = {
        "map": {
            "cells": len(h.occ),
            "grid": list(plan.blocked.shape),
            "blocked_cells": int(plan.blocked.sum()),
            "free_reachable_share": float(np.isfinite(navi.togo).mean())
            if navi.togo is not None
            else None,
            "build_ms": build_ms,
        },
        "pins": len(pins),
        "views": len(h.views),
        "pairs": len(rows),
        "goal_on_floor": float(np.mean([r["goal_on_floor"] for r in rows])),
        "routed": len(routed) / len(rows),
        "detour_median": float(np.median(detour)) if detour else None,
        "set_goal_ms_median": float(np.median(goal_ms)),
        "set_goal_ms_max": float(np.max(goal_ms)),
        "step_ms_median": float(np.median(step_ms)),
        "step_ms_p95": float(np.percentile(step_ms, 95)),
        "priors_ms_median": float(np.median(pri_ms)),
        "priors_ms_p95": float(np.percentile(pri_ms, 95)),
        "camera_sectors_known": known,
        "camera_sectors_new": new,
        "unrouted": sorted({r["pin"] for r in rows if not r["route"]}),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "walk_nav.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
