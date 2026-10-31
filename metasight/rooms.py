"""Rooms of the house, as floor outlines in the room mesh's frame (floor y = 0, metres), and the
footage ledger that counts how much of each room the walks have seen.

data/rooms/rooms.json: {"rooms": [{"name": "bedroom", "outline": [[x, z], ...]}]}
A camera is in a room when its x, z is inside the outline.

Which frames are worth keeping: a frame is needed while its view bucket (0.25 m floor cell,
30 degree heading) holds fewer than KEEP_PER_BUCKET sharp frames. Standing still at the desk
fills one bucket and stops counting; walking the room fills many.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT

ROOMS_DIR = ROOT / "data" / "rooms"
CELL_M = 0.25
HEADING_DEG = 30
KEEP_PER_BUCKET = 3
TARGET_HOURS = 10.0  # his benchmark: ten hours of footage of every room
MAX_GAP_S = 0.5  # a longer gap between frames is not footage


@dataclass
class Room:
    name: str
    outline: np.ndarray  # (n, 2) x, z

    def contains(self, pos) -> bool:
        pt = (float(pos[0]), float(pos[2]))
        return cv2.pointPolygonTest(self.outline.astype(np.float32), pt, False) >= 0


def load(path: Path | None = None) -> list[Room]:
    p = path or ROOMS_DIR / "rooms.json"
    if not p.is_file():
        return []
    return [
        Room(r["name"], np.array(r["outline"], float)) for r in json.loads(p.read_text())["rooms"]
    ]


def room_at(rooms: list[Room], pos) -> str | None:
    return next((r.name for r in rooms if r.contains(pos)), None)


def bucket(pose: np.ndarray) -> tuple[int, int, int]:
    """Floor cell and heading of a camera (OpenCV axes: z is where it looks)."""
    x, _, z = pose[:3, 3]
    f = pose[:3, 2]
    yaw = math.degrees(math.atan2(f[0], f[2])) % 360
    return int(math.floor(x / CELL_M)), int(math.floor(z / CELL_M)), int(yaw // HEADING_DEG)


@dataclass
class Ledger:
    """Footage per room: seconds seen, frames, and which view buckets are full."""

    seconds: dict[str, float] = field(default_factory=dict)
    frames: dict[str, int] = field(default_factory=dict)
    buckets: dict[str, dict[str, int]] = field(default_factory=dict)
    walks: list[str] = field(default_factory=list)

    def add(self, room: str, dt: float, pose: np.ndarray, sharp: bool) -> bool:
        """Count one frame; True when it is needed (its view bucket is not full yet)."""
        self.seconds[room] = self.seconds.get(room, 0.0) + min(dt, MAX_GAP_S)
        self.frames[room] = self.frames.get(room, 0) + 1
        b = self.buckets.setdefault(room, {})
        key = ",".join(map(str, bucket(pose)))
        if not sharp or b.get(key, 0) >= KEEP_PER_BUCKET:
            return False
        b[key] = b.get(key, 0) + 1
        return True

    def report(self) -> dict[str, dict]:
        return {
            r: {
                "hours": round(s / 3600, 3),
                "of_target": round(s / 3600 / TARGET_HOURS, 4),
                "frames": self.frames[r],
                "view_buckets": len(self.buckets.get(r, {})),
                "full_buckets": sum(v >= KEEP_PER_BUCKET for v in self.buckets.get(r, {}).values()),
            }
            for r, s in sorted(self.seconds.items())
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Ledger:
        return cls(**json.loads(path.read_text())) if path.is_file() else cls()
