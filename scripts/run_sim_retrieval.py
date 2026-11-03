"""Find my things, in the simulated bedroom: the wearer asks for the laptop and the belt guides
them from wherever they stand to the desk it sits on.

SIMULATED USER MODEL, NOT A HUMAN STUDY. The laptop is where the scan put it (laptop_06, on the
black desk). Each seed starts the walker at its own free spot on the floor with its own heading,
so the laptop can be ahead, beside or behind; the guide target is the floor in front of the desk.
Two arms (run_sim_walk.ARMS): I0, the glasses as configured, and I6, the current walker. A walk
counts as reached when it gets within 0.45 m of that spot without touching anything (the same
strict rule as the chair montage); its arrival heading error to the laptop is kept too. On a
reach the scene adds a short scripted turn to face the laptop, the way the chair walks add the
sit-down. Writes bench/results/sim-retrieval.json and one scene file per walk for
scripts/blender_sim_tile.py (data/demo-takes/sim/retrieval/scenes/{arm}-{layout}-{seed}.json).

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_sim_retrieval.py
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_sim_scene as X  # noqa: E402
import run_sim_walk as E  # noqa: E402

from metasight import fovrooms  # noqa: E402
from metasight import hapsim as h  # noqa: E402
from metasight import simwalk as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "results" / "sim-retrieval.json"
SCENES = ROOT / "data" / "demo-takes" / "sim" / "retrieval" / "scenes"
ARMS = ("I0", "I6")
SEEDS = range(1000, 1030)
ITEM = "laptop_06"
FACE_S = 0.9  # the scripted turn to face the item after a reach


def item_xyz() -> tuple[float, float, float]:
    """The laptop's centre in the simwalk floor frame (export_sim_scene.flush_parts' mapping)."""
    data = fovrooms.load_objects()
    o = next(o for o in data["objects"] if o["id"] == ITEM)
    px, py, pz = o["pos"]
    ox, oy = S.OFFSET
    return px + ox, -pz + oy, py - float(data["floor_y"])


def scenario(seed: int) -> S.Scenario:
    base = S.scenario("find_chair")
    ix, iy, _ = item_xyz()
    desk = base.world.obj("desk_01").obb
    goal = (desk[0] + desk[3] + 0.4, iy)  # the floor in front of the desk, level with the laptop
    rects, _ = base.world.rects()
    rng = np.random.default_rng(seed)
    w, hh = base.world.size
    while True:
        x, y = rng.uniform(0.3, w - 0.3), rng.uniform(0.3, hh - 0.3)
        if h._clearance(rects, x, y) > 0.12 and math.dist((x, y), goal) > 1.4:
            break
    return replace(
        base,
        name="find_laptop",
        title="Find my laptop",
        start=(x, y, rng.uniform(-math.pi, math.pi)),
        goal=goal,
        target="desk_01",
        sit=False,
    )


def face_turn(doc: dict, item: tuple[float, float, float]) -> None:
    """Append the turn toward the item, before the hold frames export_sim_scene added."""
    fr = doc["frames"]
    hold = int(X.HOLD_S * X.FPS)
    walk, tail = fr[:-hold], fr[-hold:]
    last = walk[-1]
    want = math.atan2(item[1] - last["y"], item[0] - last["x"])
    turn = (want - last["heading"] + math.pi) % (2 * math.pi) - math.pi
    m = int(FACE_S * X.FPS)
    for k in range(m):
        a = (k + 1) / m
        b = a * a * (3 - 2 * a)
        walk.append(
            dict(
                last,
                t=round(last["t"] + a * FACE_S, 4),
                heading=round(last["heading"] + turn * b, 4),
                felt=[0.0] * 8,
                walking=False,
            )
        )
    end = walk[-1]
    doc["frames"] = walk + [
        dict(end, t=round(end["t"] + (k + 1) / X.FPS, 4)) for k in range(len(tail))
    ]
    doc["face_turn"] = {"from_frame": len(fr) - hold, "frames": m}


def one(job) -> dict:
    arm, layout, seed = job
    sc = scenario(seed)
    tr = S.walk(sc, E.belt_for(arm, layout), seed=seed, **E.ARMS[arm][2])
    m = tr.metrics()
    item = item_xyz()
    _, x, y, hd = tr.steps[-1][:4]
    err = (math.atan2(item[1] - y, item[0] - x) - hd + math.pi) % (2 * math.pi) - math.pi
    doc = X.scene_file(sc, tr, layout, seed)
    doc["arm"] = arm
    doc["item"] = {"id": ITEM, "label": "laptop", "xyz": [round(v, 4) for v in item]}
    reached = bool(m["success"] and m["collisions"] == 0)
    if reached:
        face_turn(doc, item)
    SCENES.mkdir(parents=True, exist_ok=True)
    (SCENES / f"{arm}-{layout}-{seed}.json").write_text(json.dumps(doc))
    return {
        "arm": arm,
        "layout": layout,
        "seed": seed,
        "start": [round(v, 3) for v in sc.start],
        "arrived": bool(m["success"]),
        "reached": reached,
        "collisions": m["collisions"],
        "time_s": m["time_s"],
        "arrival_heading_err_deg": round(abs(math.degrees(err)), 1),
        "frames": len(doc["frames"]),
    }


def main() -> None:
    jobs = [(a, lay, s) for a in ARMS for lay in E.LAYOUTS for s in SEEDS]
    with mp.get_context("spawn").Pool(8) as p:  # a forked MuJoCo GL context hangs on macOS
        rows = p.map(one, jobs, chunksize=2)
    summary = {}
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a]
        summary[a] = {
            "walks": len(rs),
            "reached_no_contact": sum(r["reached"] for r in rs),
            "arrived_any": sum(r["arrived"] for r in rs),
            "contacts": sum(r["collisions"] for r in rs),
            "median_time_s": float(np.median([r["time_s"] for r in rs if r["reached"]] or [0])),
        }
        print(a, summary[a])
    OUT.write_text(
        json.dumps(
            {
                "item": ITEM,
                "seeds": [SEEDS.start, SEEDS.stop - 1],
                "summary": summary,
                "rows": rows,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
