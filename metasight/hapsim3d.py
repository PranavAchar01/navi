"""The simulated walker with rendered depth: MuJoCo draws each room in 3D and the live safety
code (`safety.analyse`, floor fit and all) turns the depth into the strips the belt plans from.

SIMULATED USER MODEL, NOT A HUMAN STUDY. The 2D walker (`hapsim.sense`) casts rays against the
furniture footprints, so a table is a solid block. Here the same rooms get heights: boxes 0.4 to
1.1 m tall and tables (a top at about 0.75 m on four thin legs), walls 2.5 m, a floor. The camera
sits at the live eye height, pitched down like the glasses, and the depth goes through the same
`analyse` -> `grid_strips` path as `server.py`. The side and back sectors still come from the
footprints, as the live belt takes them from the map. The walker itself stays the 2D kinematic
one (footprint collisions), so only perception changes.

    sensor = Sensor3D(room); hapsim.run(room, belt, sensor=sensor)
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import hapsim as h
from .geometry import grid_strips
from .safety import GRID, analyse

EYE_H = 1.37  # config.Settings.eye_height_m
PITCH_DEG = 15.0  # glasses look a little down
WALL_H = 2.5


def room_xml(room: h.Room) -> str:
    """MJCF for a hapsim room. Furniture gets a height and a kind from the room's own seed."""
    rng = np.random.default_rng(room.seed + 31_337)
    geoms = ['<geom type="plane" size="20 20 0.1" pos="4.5 2.75 0"/>']
    for i, (x0, y0, x1, y1) in enumerate(room.rects):
        cx, cy, sx, sy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        if i < room.n_walls:
            geoms.append(_box(cx, cy, WALL_H / 2, sx, sy, WALL_H / 2))
        elif rng.random() < 0.4:  # table: a top and four legs
            top = rng.uniform(0.70, 0.78)
            geoms.append(_box(cx, cy, top - 0.02, sx, sy, 0.02))
            for lx in (x0 + 0.03, x1 - 0.03):
                for ly in (y0 + 0.03, y1 - 0.03):
                    geoms.append(_box(lx, ly, (top - 0.04) / 2, 0.02, 0.02, (top - 0.04) / 2))
        else:
            hgt = rng.uniform(0.4, 1.1)
            geoms.append(_box(cx, cy, hgt / 2, sx, sy, hgt / 2))
    fovy = 2 * math.degrees(math.atan(math.tan(math.radians(h.HFOV) / 2) * GRID[1] / GRID[0]))
    return (
        f'<mujoco><visual><global offwidth="{GRID[0]}" offheight="{GRID[1]}"/>'
        '<map znear="0.02" zfar="20"/></visual><worldbody>'
        + "".join(geoms)
        + f'<camera name="eye" pos="0 0 {EYE_H}" fovy="{fovy:.4f}"/></worldbody></mujoco>'
    )


def _box(cx, cy, cz, sx, sy, sz) -> str:
    return f'<geom type="box" pos="{cx:.4f} {cy:.4f} {cz:.4f}" size="{sx:.4f} {sy:.4f} {sz:.4f}"/>'


def cam_quat(heading: float, pitch_deg: float = PITCH_DEG) -> np.ndarray:
    """MuJoCo cameras look down their -z with +y up; heading is counter-clockwise from +x."""
    p = math.radians(pitch_deg)
    fwd = np.array([math.cos(heading) * math.cos(p), math.sin(heading) * math.cos(p), -math.sin(p)])
    right = np.array([math.sin(heading), -math.cos(heading), 0.0])
    z = -fwd
    y = np.cross(z, right)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([right, y, z]).ravel())
    return q


class Sensor3D:
    """`hapsim.run(..., sensor=Sensor3D(room))`: depth rendered at 25 fps, the live geometry on it.
    `noise` is the per-frame scale error of monocular depth (log sd) plus 1% pixel noise."""

    def __init__(self, room: h.Room, noise: float = 0.04):
        self.m = mujoco.MjModel.from_xml_string(room_xml(room))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, GRID[1], GRID[0])
        self.r.enable_depth_rendering()
        self.noise = noise
        self.prev = None

    def depth(self, x: float, y: float, heading: float) -> np.ndarray:
        self.m.cam_pos[0] = (x, y, EYE_H)
        self.m.cam_quat[0] = cam_quat(heading)
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera="eye")
        z = self.r.render().astype(np.float32)
        z[z > 15.0] = 0.0  # nothing there (far plane): no data, as on the phone
        return z

    def __call__(self, room: h.Room, x, y, heading, rng) -> tuple[list[float], list[float]]:
        z = self.depth(x, y, heading)
        z *= rng.lognormal(0, self.noise) * rng.normal(1, 0.01, z.shape).astype(np.float32)
        scene, self.prev = analyse(z, 1.0, 1.0, h.HFOV, EYE_H, self.prev)
        strips = [s if s < h.MAX_RANGE else math.inf for s in grid_strips(scene.grid, h.STRIPS)]
        _, sectors = h.sense(room, x, y, heading, rng)
        return strips, sectors

    def close(self) -> None:
        self.r.close()
