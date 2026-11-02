"""Perturbed prior maps for scoring the memory's prior seed (L29).

SIMULATED USER MODEL, NOT A HUMAN STUDY. The bedroom scenes (`fovrooms`) are built from the same
`objects.json` a prior would be seeded with, so seeding it unchanged is circular: an upper bound.
`worlds` returns, for one seed, the pairs a walker meets in practice: a room as it is now (the
truth the sim renders) and the prior map as it was recorded.

- `same`: the room is as recorded; the prior is exact (the ceiling), or has pose noise (centres
  0.1 to 0.3 m, yaw up to 10 degrees), or lacks two hazards, or sits 0.4 / 0.8 m off in the
  frame (the offset seen on the real take).
- `chair`: each chair has moved 0.6 to 1.2 m or gone since the recording.
- `ghost`: one recorded hazard is no longer in the room.
- `real`: pose noise, the chairs moved, two hazards missing from the prior, one ghost, together.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from .fovmem import PRIOR_SKIP

MOVABLE_CLASSES = ("chair",)


@dataclass
class World:
    name: str
    truth: dict  # raw objects.json the sim renders
    priors: dict[str, dict] = field(default_factory=dict)  # name -> load_prior-style doc


def _hazards(raw: dict) -> list[str]:
    return [
        o["id"] for o in raw["objects"] if o.get("hazard", True) and o["class"] not in PRIOR_SKIP
    ]


def _drop(raw: dict, ids) -> dict:
    out = copy.deepcopy(raw)
    out["objects"] = [o for o in out["objects"] if o["id"] not in ids]
    return out


def _prior_drop(prior: dict, ids) -> dict:
    return {**prior, "objects": [o for o in prior["objects"] if o["id"] not in ids]}


def jitter(prior: dict, rng: np.random.Generator, lo: float = 0.1, hi: float = 0.3) -> dict:
    """Pose noise: every centre moved lo..hi m in a random direction, yaw +-10 degrees."""
    out = copy.deepcopy(prior)
    for o in out["objects"]:
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(lo, hi)
        o["center"] = np.array(o["center"], float) + [r * math.cos(a), 0.0, r * math.sin(a)]
        o["yaw_rad"] = float(o["yaw_rad"]) + math.radians(rng.uniform(-10, 10))
    return out


def shift(prior: dict, dx: float, dz: float) -> dict:
    """The whole prior in a frame `dx`, `dz` metres off (map x and z)."""
    out = copy.deepcopy(prior)
    for o in out["objects"]:
        o["center"] = np.array(o["center"], float) + [dx, 0.0, dz]
    return out


def move_chairs(raw: dict, rng: np.random.Generator) -> tuple[dict, list[str]]:
    """The room with each chair moved 0.6 to 1.2 m or removed (half each)."""
    out = copy.deepcopy(raw)
    keep, gone = [], []
    for o in out["objects"]:
        if o["class"] in MOVABLE_CLASSES:
            if rng.random() < 0.5:
                gone.append(o["id"])
                continue
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.6, 1.2)
            c = o["obb"]["center"]
            o["obb"]["center"] = [c[0] + r * math.cos(a), c[1], c[2] + r * math.sin(a)]
        keep.append(o)
    out["objects"] = keep
    return out, gone


def worlds(raw: dict, prior: dict, seed: int) -> list[World]:
    """The four worlds of one seed: raw is the parsed objects.json, prior its `load_prior`."""
    rng = np.random.default_rng([seed, 29])
    ids = [i for i in _hazards(raw) if not i.startswith("chair")]
    same = World(
        "same",
        raw,
        {
            "ceiling": prior,
            "pose": jitter(prior, rng),
            "missing2": _prior_drop(prior, set(rng.choice(ids, 2, replace=False))),
            "shift0.4": shift(prior, 0.28, 0.28),
            "shift0.8": shift(prior, 0.57, 0.57),
        },
    )
    moved, _ = move_chairs(raw, rng)
    chair = World("chair", moved, {"prior": prior})
    ghost_id = str(rng.choice(ids))
    ghost = World("ghost", _drop(raw, {ghost_id}), {"prior": prior})
    rest = [i for i in ids if i != ghost_id]
    miss = set(rng.choice(rest, 2, replace=False))
    truth, _ = move_chairs(_drop(raw, {ghost_id}), rng)
    real = World("real", truth, {"prior": _prior_drop(jitter(prior, rng), miss)})
    return [same, chair, ghost, real]


def truth_scenes(truth: dict, base: dict) -> dict:
    """The five bedroom paths through `truth`, with the walls of `base` (the room's walls do not
    move when a piece of furniture is taken out)."""
    from dataclasses import replace

    from . import fovrooms

    objs, _ = fovrooms.scene_boxes(truth, open_doors=("door_03",), walls=False)
    all_b, _ = fovrooms.scene_boxes(base, open_doors=("door_03",))
    n = len(fovrooms.scene_boxes(base, walls=False)[0])
    return {
        k: replace(s, boxes=objs + all_b[n:]) for k, s in fovrooms.bedroom_scenes(truth).items()
    }
