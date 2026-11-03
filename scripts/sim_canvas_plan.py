"""Timeline of the training-canvas reel, shared by the Blender pass (scripts/blender_sim_canvas.py)
and the overlay pass (scripts/build_sim_canvas.py) so both agree on which walk is where when.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Every walk is a recorded episode of the walk ledger
(scenes from scripts/run_sim_worlds.py, each checked against bench/results/sim-walk.json). A
window plays a walk from scene frame `start` at `speed` sim frames per output frame, and the
walker is on screen from `show` to `hide` (exclusive).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "data" / "demo-takes" / "sim" / "worlds" / "scenes"
FPS = 30
FRAMES = 480  # 16 s
LAYOUT, SEED = "eight", 1000
HERO = "find_chair"
# 3 x 2 on the ground, row 0 at the back (+y)
GRID = (("avoid_chair", "find_chair", "desk_lookup"), ("shelf_head", "low_box", "chair_legs"))
PITCH = (6.6, 8.0)  # metres between room centres
TAG = {
    "find_chair": ("BEDROOM / CHAIR TUCKED", "find the chair and sit"),
    "avoid_chair": ("BEDROOM / CHAIR IN THE AISLE", "get past the pulled-out chair"),
    "desk_lookup": ("DESK EDGE", "looks down at the desk, then up"),
    "shelf_head": ("SHELF AT HEAD HEIGHT", "a board at 1.6 m"),
    "low_box": ("LOW BOX", "a shoebox on the floor"),
    "chair_legs": ("CHAIR ON THIN LEGS", "legs the camera barely sees"),
}
# phases (scene frames, 1-based)
OVERVIEW_I0 = 1
UPDATES = (122, 150)  # the five policy updates between I0 and I6
OVERVIEW_I6 = 150
FLY = (166, 256)  # camera flies down into the bedroom
CLOSE_I0 = 258
CLOSE_I6 = 380
SPEED_WIDE, SPEED_CLOSE = 3.0, 2.5


@dataclass
class Window:
    key: str  # unique name
    world: str
    arm: str
    start: int
    speed: float
    show: int
    hide: int  # exclusive
    layout: str = LAYOUT
    seed: int = SEED

    def scene(self) -> dict:
        name = f"{self.arm}-{self.world}-{self.layout}-{self.seed}.json"
        return json.loads((SCENES / name).read_text())

    def n_out(self, n_sim: int) -> int:
        return math.ceil((n_sim - 1) / self.speed) + 1

    def sim_at(self, f: int, n_sim: int) -> int:
        return int(min(max(round((f - self.start) * self.speed), 0), n_sim - 1))

    def out_of(self, k: int) -> float:
        return self.start + k / self.speed


def windows() -> list[Window]:
    out = []
    for row in GRID:
        for w in row:
            out.append(Window(f"wide-I0-{w}", w, "I0", OVERVIEW_I0, SPEED_WIDE, 1, OVERVIEW_I6))
            hide = CLOSE_I0 if w == HERO else FRAMES + 1
            out.append(Window(f"wide-I6-{w}", w, "I6", OVERVIEW_I6, SPEED_WIDE, OVERVIEW_I6, hide))
    out.append(Window("close-I0", HERO, "I0", CLOSE_I0, SPEED_CLOSE, CLOSE_I0, CLOSE_I6))
    out.append(Window("close-I6", HERO, "I6", CLOSE_I6, SPEED_CLOSE, CLOSE_I6, FRAMES + 1))
    return out


# the lattice: every walk of the ledger gets its own tile, 6 worlds x 2 belt layouts x 30 tuning
# seeds = 360 tiles in 24 columns x 15 rows. Neighbouring tiles (diagonals too) are different
# worlds; the bedroom tile at the centre is the one the camera flies into (eight motors, seed 1000).
WORLDS = [w for row in GRID for w in row]
LATTICE = (24, 15)
HERO_CELL = (12, 7)
LAYOUTS = ("eight", "diag4")
SEEDS = tuple(range(1000, 1030))
ROUND_I0 = (1, 148)  # every I0 walk finishes inside these output frames
ROUND_I6 = (150, 250)  # and every I6 walk inside these


def world_at(c: int, r: int) -> str:
    return WORLDS[(c + 3 * r + 4) % 6]


def cell_xy(c: int, r: int) -> tuple[float, float]:
    """Centre of a lattice cell on the ground; the hero cell sits at the origin."""
    return (c - HERO_CELL[0]) * PITCH[0], (HERO_CELL[1] - r) * PITCH[1]


def tiles() -> list[tuple[int, int, str, str, int]]:
    """(column, row, world, layout, seed) for all 360 tiles. Within a world, walks go out from
    the centre in seed order, so the hero cell gets the bedroom's eight-motor seed-1000 walk."""
    walks = [(lay, sd) for sd in SEEDS for lay in LAYOUTS]
    out = []
    for w in WORLDS:
        cells = [
            (c, r) for c in range(LATTICE[0]) for r in range(LATTICE[1]) if world_at(c, r) == w
        ]
        cells.sort(key=lambda cr: (math.dist(cr, HERO_CELL), cr))
        assert len(cells) == len(walks), (w, len(cells))
        out += [(c, r, w, lay, sd) for (c, r), (lay, sd) in zip(cells, walks, strict=True)]
    hero = next(t for t in out if t[:2] == HERO_CELL)
    assert hero[2:] == (HERO, LAYOUT, SEED), hero
    return out


def tile_window(world: str, arm: str, layout: str, seed: int, n_sim: int) -> Window:
    """A tile's walk, sped up only as far as it must be to finish inside its round."""
    lo, hi = ROUND_I0 if arm == "I0" else ROUND_I6
    speed = max(SPEED_WIDE, (n_sim - 1) / (hi - lo))
    show, hide = (1, OVERVIEW_I6) if arm == "I0" else (OVERVIEW_I6, FRAMES + 1)
    key = f"{world}-{layout}-{seed}-{arm}"
    return Window(key, world, arm, lo, speed, show, hide, layout, seed)


def offsets(bounds: dict) -> dict:
    """World -> canvas shift per room so each room's centre sits on its grid cell."""
    out = {}
    for r, row in enumerate(GRID):
        for c, w in enumerate(row):
            if w not in bounds:
                continue
            x0, x1, y0, y1 = bounds[w]
            gx, gy = (c - 1) * PITCH[0], (0.5 - r) * PITCH[1]
            out[w] = (gx - (x0 + x1) / 2, gy - (y0 + y1) / 2)
    return out
