"""Field-of-view failure scenes: live cue vs live plus hazard memory (fovsim, fovmem).

SIMULATED USER MODEL, NOT A HUMAN STUDY. Every scene in fovsim.SCENES, `--seeds` noise and path
jitter seeds each. Per scene and arm: the share of danger frame-sectors missed (a hazard under
1 m that the cue reports more than 0.3 m too far, or not at all), the same for the walking path,
and false cues a minute. Paired over seeds, bootstrap 95% on the miss difference against the live
cue. Writes bench/results/fov-sim.json.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_fov.py [--seeds 20]
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from metasight import fovrooms
from metasight import fovsim as F

ROOT = Path(__file__).resolve().parent.parent
# The recommended memory: L14's mask and slope, plus L16's floor checks on the pitch and L24's
# looser backing for cramped rooms.
L14 = dict(low_slope=0.02, mask=True, people=True)
REC = dict(**L14, trust="level", min_span=0.8, loose=True)
ARMS = {
    "memory, L14": F.Arms(**L14),
    "memory": F.Arms(**REC),
    # L27 cue gate (fovmem.gate_memory_cues): s still mute, a approach gate, r rear mute
    "gate s": F.Arms(**REC, cue_gate="s"),
    "gate a": F.Arms(**REC, cue_gate="a"),
    "gate r": F.Arms(**REC, cue_gate="r"),
    "gate sar": F.Arms(**REC, cue_gate="sar"),
}

# Tuning hook: FOV_ARMS_EXTRA='{"name": {"cue_gate": "a", "approach_m": 0.1}}' adds arms (an
# environment variable, so the worker processes see it too).
for _name, _kw in json.loads(os.environ.get("FOV_ARMS_EXTRA", "{}")).items():
    ARMS[_name] = F.Arms(**REC, **_kw)


def _scene(name: str, objects: str | None = None) -> F.Scene:
    """A synthetic scene, or one of the reconstructed bedroom's (built again in each worker)."""
    if name in F.SCENES:
        return F.SCENES[name]
    return fovrooms.bedroom_scenes(objects or fovrooms.BEDROOM_OBJECTS)[name]


def _one(a):
    name, seed, hfov, eye_cfg, objects = a
    sc = _scene(name, objects)
    scores = F.run(sc, list(ARMS.values()), seed=seed, hfov=hfov, eye_h_cfg=eye_cfg)
    return name, seed, [asdict(s) for s in scores]


def _ci(diff: np.ndarray, rng) -> list[float]:
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    return [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=0, help="20 for the held-out set")
    ap.add_argument("--hfov", type=float, default=F.HFOV)
    ap.add_argument("--eye-cfg", type=float, default=F.EYE_H, help="eye height the code believes")
    ap.add_argument("--scenes", nargs="*", default=list(F.SCENES))
    ap.add_argument(
        "--bedroom",
        nargs="?",
        const=str(fovrooms.BEDROOM_OBJECTS),
        default=None,
        metavar="OBJECTS_JSON",
        help="score the reconstructed bedroom scenes (fovrooms) (`--scenes` then picks among them)",
    )
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "fov-sim.json"))
    a = ap.parse_args()
    if a.bedroom:
        every = list(fovrooms.bedroom_scenes(a.bedroom))
        a.scenes = [n for n in a.scenes if n in every] or every
    jobs = [
        (n, s, a.hfov, a.eye_cfg, a.bedroom)
        for n in a.scenes
        for s in range(a.first_seed, a.first_seed + a.seeds)
    ]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(_one, jobs))
    names = ["live", *ARMS]
    rng = np.random.default_rng(0)
    out = {"hfov": a.hfov, "eye_cfg": a.eye_cfg, "seeds": a.seeds, "scenes": {}}
    total = {k: F.Score() for k in names}
    for sc in a.scenes:
        rows = [r for r in res if r[0] == sc]
        agg = {}
        per_seed = {k: [] for k in names}
        for _, _, scores in rows:
            for k, s in zip(names, scores, strict=True):
                per_seed[k].append(F.Score(**s))
        for k in names:
            tot = F.Score()
            for s in per_seed[k]:
                tot.add(s)
            total[k].add(tot)
            agg[k] = tot.rates()

        def miss(s: F.Score) -> float:
            return s.missed / s.danger if s.danger else 0.0

        base = np.array([miss(s) for s in per_seed["live"]])
        for k in ARMS:
            arm = np.array([miss(s) for s in per_seed[k]])
            agg[k]["miss_diff_ci"] = _ci(arm - base, rng)
        out["scenes"][sc] = {"note": _scene(sc, a.bedroom).note, "arms": agg}
    out["total"] = {k: v.rates() for k, v in total.items()}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for sc, v in out["scenes"].items():
        print(sc)
        for k, r in v["arms"].items():
            print(
                f"  {k:24s} miss {r['miss']}  front {r['front_miss']}  path {r['path_miss']}  "
                f"false/min {r['false_per_min']} (far {r['false_far_per_min']})  "
                f"drop miss {r['drop_miss']} false/min {r['drop_false_per_min']}  "
                f"diff {r.get('miss_diff_ci', '')}"
            )
    print("total")
    for k, r in out["total"].items():
        print(f"  {k:24s} {r}")


if __name__ == "__main__":
    main()
