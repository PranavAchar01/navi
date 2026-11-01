"""The room's saved objects as a second opinion on the detector, once a frame has a room pose.

Two moves, each on normalised boxes (x0, y0, x1, y1 in 0..1):
- confirm: a weak box (CONFIRM_CONF up to the live threshold) is kept when a saved object of its
  label projects inside it;
- reject: a furniture box below REJECT_CONF whose centre ray meets the room mesh more than REJECT_M
  from every saved object of its label is dropped (only for labels the room has saved).

Leave-one-out on walk 20261031-004732-glasses (scripts/eval_room_prior.py, 41 frames, 26 placed):
precision 0.92 to 0.96 and recall 0.46 to 0.55 against YOLOE-26L boxes. Filling in saved objects
the detector missed raised recall to 0.71 but dropped precision to 0.49: the map calls the
whiteboard a tv, and laptops, phones and mice move. So the map vouches for boxes, never adds them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pins import FURNITURE

MIN_VIEWS = 20  # saved objects fewer frames agreed on are too unsure to vouch for anything
LIVE_CONF = 0.35  # the detector's live threshold
CONFIRM_CONF = 0.2
REJECT_CONF = 0.6
REJECT_M = 1.0
NEAR_M, FAR_M = 0.3, 5.0
HIDDEN_M = 0.3  # the mesh in front of an object by more than this hides it
GROW = 0.1  # boxes grow by this fraction per side before a projection is tested against them


@dataclass
class Box:
    """Any detection with these three fields will do (safety.Obstacle has them)."""

    label: str
    score: float
    box: tuple[float, float, float, float]


def _inside(uv: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    gx, gy = GROW * (x1 - x0), GROW * (y1 - y0)
    return x0 - gx <= uv[0] <= x1 + gx and y0 - gy <= uv[1] <= y1 + gy


class RoomPrior:
    def __init__(self, objects: list[dict], mesh_path: Path) -> None:
        import open3d as o3d

        keep = [o for o in objects if o["views"] >= MIN_VIEWS]
        self.labels = [o["label"] for o in keep]
        self.pos = np.array([o["pos"] for o in keep], np.float64).reshape(-1, 3)
        mesh = o3d.t.io.read_triangle_mesh(str(mesh_path))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh)

    def _cast(self, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Distance along each unit ray to the mesh (inf where it escapes)."""
        import open3d as o3d

        rays = np.c_[np.repeat(origin[None], len(dirs), 0), dirs].astype(np.float32)
        return self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()

    def visible(self, pose: np.ndarray, k: np.ndarray) -> list[tuple[int, np.ndarray, float]]:
        """Saved objects this camera sees: (index, normalised pixel, distance m). `pose` is camera
        to room (OpenCV axes), `k` the pinhole matrix in normalised image units."""
        if not len(self.pos):
            return []
        c = (self.pos - pose[:3, 3]) @ pose[:3, :3]
        z = c[:, 2]
        uv = (c[:, :2] / np.maximum(z, 1e-6)[:, None]) * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
        dist = np.linalg.norm(c, axis=1)
        cand = np.flatnonzero(
            (z > 0) & (dist >= NEAR_M) & (dist <= FAR_M) & np.all((uv >= 0) & (uv <= 1), axis=1)
        )
        if not len(cand):
            return []
        dirs = (self.pos[cand] - pose[:3, 3]) / dist[cand, None]
        hit = self._cast(pose[:3, 3], dirs)
        return [
            (int(i), uv[i], float(dist[i]))
            for i, t in zip(cand, hit, strict=True)
            if t >= dist[i] - HIDDEN_M
        ]

    def apply(self, boxes: list, pose: np.ndarray, k: np.ndarray) -> list:
        """The boxes the prior keeps. `boxes` come from the detector at CONFIRM_CONF."""
        seen = self.visible(pose, k)
        known = set(self.labels)
        out: list[Box] = []
        for b in boxes:
            near = [uv for i, uv, _ in seen if self.labels[i] == b.label]
            if b.score < LIVE_CONF:
                if any(_inside(uv, b.box) for uv in near):
                    out.append(b)
                continue
            if b.label in FURNITURE and b.label in known and b.score < REJECT_CONF:
                x0, y0, x1, y1 = b.box
                ray = np.array(
                    [((x0 + x1) / 2 - k[0, 2]) / k[0, 0], ((y0 + y1) / 2 - k[1, 2]) / k[1, 1], 1.0]
                )
                ray = pose[:3, :3] @ (ray / np.linalg.norm(ray))
                t = float(self._cast(pose[:3, 3], ray[None])[0])
                if np.isfinite(t):
                    p = pose[:3, 3] + t * ray
                    mine = self.pos[[lab == b.label for lab in self.labels]]
                    if np.linalg.norm(mine - p, axis=1).min() > REJECT_M:
                        continue
            out.append(b)
        return out
