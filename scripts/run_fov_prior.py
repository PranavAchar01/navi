"""Prior seed for the field-of-view memory (L29): the bedroom with a perturbed prior map.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Worlds per seed (`metasight.fovprior`): the room as
recorded with an exact prior (the ceiling, circular: the scenes are built from the same
objects.json), pose noise, two hazards missing, the frame 0.4 / 0.8 m off; the chairs moved or
gone; one hazard gone; and all of that together ("real"). Each world is scored for the live cue,
the recommended memory (REC) and REC seeded with the prior, for each `--variants` entry
(keyword arguments of `HazardMemory.seed`). Miss, path miss, false cues a
minute and far false cues a minute. Writes bench/results/fov-prior.json (or --out).

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/run_fov_prior.py --first-seed 20
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_fov as R  # noqa: E402

from metasight import fovprior, fovrooms, recon  # noqa: E402
from metasight import fovsim as F  # noqa: E402

SCENES = [
    "bedroom_desk_lookup",
    "bedroom_turn_back",
    "bedroom_bed_corner",
    "bedroom_doorway",
    "bedroom_low_table",
]
WORLDS = ["same", "chair", "ghost", "real"]


def _one(a):
    seed, scene, wi, variants, objects, only = a
    raw = fovrooms.load_objects(objects)
    prior = recon.load_prior(objects)
    w = fovprior.worlds(raw, prior, seed)[wi]
    scenes = fovprior.truth_scenes(w.truth, raw)
    arms, priors, names = [F.Arms(**R.REC)], {}, []
    for pname, doc in w.priors.items():
        if only and pname not in only:
            continue
        for vname, kw in variants.items():
            key = f"{pname}@{vname}"
            priors[key] = {**doc, "kw": kw}
            arms.append(F.Arms(**R.REC, prior=key))
            names.append(key)
    sc = F.run(scenes[scene], arms, seed=seed, priors=priors)
    return w.name, scene, ["live", "REC", *names], [asdict(s) for s in sc]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--worlds", nargs="*", default=WORLDS)
    ap.add_argument("--priors", nargs="*", help="only these prior names (default all)")
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument(
        "--variants",
        default="{}",
        help='JSON {"name": HazardMemory.seed kwargs}, e.g. {"n3": {"need_fixed": 3}}',
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--objects", default=str(fovrooms.BEDROOM_OBJECTS))
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "fov-prior.json"))
    a = ap.parse_args()
    variants = json.loads(a.variants) or {"seed": {}}
    jobs = [
        (s, sc, WORLDS.index(w), variants, a.objects, a.priors)
        for w in a.worlds
        for sc in a.scenes
        for s in range(a.first_seed, a.first_seed + a.seeds)
    ]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_one, jobs))
    tot: dict[str, dict[str, F.Score]] = {}
    per_scene: dict[str, dict[str, dict[str, F.Score]]] = {}
    for world, scene, names, scores in res:
        for n, s in zip(names, scores, strict=True):
            tot.setdefault(world, {}).setdefault(n, F.Score()).add(F.Score(**s))
            per_scene.setdefault(world, {}).setdefault(scene, {}).setdefault(n, F.Score()).add(
                F.Score(**s)
            )
    out = {
        "seeds": [a.first_seed, a.seeds],
        "total": {w: {n: s.rates() for n, s in d.items()} for w, d in tot.items()},
        "by_scene": {
            w: {sc: {n: s.rates() for n, s in d.items()} for sc, d in v.items()}
            for w, v in per_scene.items()
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    for w, d in out["total"].items():
        print(w)
        for n, r in d.items():
            print(
                f"  {n:22s} miss {r['miss']} path {r['path_miss']} front {r['front_miss']} "
                f"false/min {r['false_per_min']} far {r['false_far_per_min']}"
            )


if __name__ == "__main__":
    main()
