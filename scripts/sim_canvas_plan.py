"""Timeline of the training-canvas reel, shared by the Blender pass (scripts/blender_sim_canvas.py)
and the overlay pass (scripts/build_sim_canvas.py) so both agree on which walk is where when.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Every walk is a recorded episode of the walk ledger
(seed 1000, eight motors; scenes from scripts/run_sim_worlds.py). A window plays a walk from
scene frame `start` at `speed` sim frames per output frame, and the walker is on screen from
`show` to `hide` (exclusive).
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

    def scene(self) -> dict:
        return json.loads((SCENES / f"{self.arm}-{self.world}-{LAYOUT}-{SEED}.json").read_text())

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
