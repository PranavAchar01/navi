"""Scene files for the six-worlds reel: one walk per world, belt layout and arm, replayed from
the walk ledger so every room in the reel ends the way its recorded episode did.

SIMULATED USER MODEL, NOT A HUMAN STUDY. The six worlds are the ones the walker was tuned and
scored in (`simwalk.SCENARIOS`): the reconstructed bedroom twice (chair tucked at the desk, chair
pulled into the aisle) and the four field-of-view catalogue rooms (desk edge, shelf at head
height, low box, chair on thin legs). Seed 1000, the first ledger seed, on both belt layouts,
for I0 and I6. Each replay is checked against `bench/results/sim-walk.json` (contacts, reached)
before it is written to data/demo-takes/sim/worlds/scenes/{arm}-{scenario}-{layout}-{seed}.json
for scripts/blender_sim_tile.py.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_sim_worlds.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_sim_scene as X  # noqa: E402
import run_sim_walk as E  # noqa: E402

from metasight import simwalk as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "data" / "demo-takes" / "sim" / "worlds" / "scenes"
ARMS = ("I0", "I6")
SEED = 1000


def ledger() -> dict:
    rows = json.loads((ROOT / "bench" / "results" / "sim-walk.json").read_text())["rows"]
    return {(r["arm"], r["scenario"], r["layout"], r["seed"]): r for r in rows}


def one(job) -> str:
    arm, name, layout = job
    sc = S.scenario(name)
    tr = S.walk(sc, E.belt_for(arm, layout), seed=SEED, **E.ARMS[arm][2])
    doc = X.scene_file(sc, tr, layout, SEED)
    doc["arm"] = arm
    row = ledger()[(arm, name, layout, SEED)]
    clean = bool(row["success"] and row["collisions"] == 0)
    assert doc["metrics"]["collisions"] == row["collisions"], (job, doc["metrics"], row)
    assert doc["metrics"]["success"] == clean, (job, doc["metrics"], row)
    SCENES.mkdir(parents=True, exist_ok=True)
    (SCENES / f"{arm}-{name}-{layout}-{SEED}.json").write_text(json.dumps(doc))
    return f"{arm} {name} {layout} contacts {row['collisions']} clean {clean}"


def main() -> None:
    jobs = [(a, n, lay) for a in ARMS for n in S.SCENARIOS for lay in E.LAYOUTS]
    with mp.get_context("spawn").Pool(8) as p:  # a forked MuJoCo GL context hangs on macOS
        for line in p.map(one, jobs):
            print(line)


if __name__ == "__main__":
    main()
