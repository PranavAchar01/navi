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

    def __init__(self, room: h.Room, noise: float = 0.04, xml: str | None = None, strip_q=(0.0, 1)):
        self.m = mujoco.MjModel.from_xml_string(xml or room_xml(room))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, GRID[1], GRID[0])
        self.r.enable_depth_rendering()
        self.noise = noise
        self.strip_q = strip_q  # (percentile, min pixels) for geometry.grid_strips
        self.prev = None

    def depth(self, x: float, y: float, heading: float) -> np.ndarray:
        self.m.cam_pos[0] = (x, y, EYE_H)
        self.m.cam_quat[0] = cam_quat(heading)
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera="eye")
        z = self.r.render().astype(np.float32)
        z[~(z <= 15.0)] = 0.0  # nothing there (far plane, NaN): no data, as on the phone
        return z

    def __call__(self, room: h.Room, x, y, heading, rng) -> tuple[list[float], list[float]]:
        z = self.depth(x, y, heading)
        z *= rng.lognormal(0, self.noise) * rng.normal(1, 0.01, z.shape).astype(np.float32)
        scene, self.prev = analyse(z, 1.0, 1.0, h.HFOV, EYE_H, self.prev)
        strips = [
            s if s < h.MAX_RANGE else math.inf
            for s in grid_strips(scene.grid, h.STRIPS, *self.strip_q)
        ]
        _, sectors = h.sense(room, x, y, heading, rng)
        return strips, sectors

    def close(self) -> None:
        self.r.close()


# ------------------------------------------------------------------ the scanned bedroom
# The scan is in the home frame (y up, floor at y = 0). The sim plane is x right, y = -z, so the
# mesh goes into MuJoCo as (x, -z, y) and both get the same offset to keep coordinates positive.
CELL = 0.1


def scan_rooms(ply: str, n_pairs: int = 20, seed: int = 0) -> tuple[list[h.Room], str]:
    """Rooms (one per start and goal pair) and the MJCF for a scanned mesh. Floor plan from the
    mesh: a 10 cm cell with surface at 1.9 to 2.6 m is structure (walls, tall wardrobes: the walker
    knows them, like the walls of a generated room), one with surface from 0.3 to 1.8 m only is
    furniture (the belt has to find it), one with neither and no floor seen is outside the scan.
    Clutter under 0.3 m is drawn, so the camera sees it, but the walker does not collide with it."""
    import trimesh

    m = trimesh.load(ply, process=False)
    v = np.asarray(m.vertices)
    sim = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
    off = -sim[:, :2].min(0) + 0.3
    sim[:, :2] += off
    p, _ = trimesh.sample.sample_surface(
        trimesh.Trimesh(sim, m.faces, process=False), 400_000, seed=0
    )
    size = tuple(float(s) for s in sim[:, :2].max(0) + 0.3)
    nx, ny = int(size[0] / CELL), int(size[1] / CELL)

    def count(sel):
        c = np.zeros((nx, ny), int)
        q = p[sel]
        np.add.at(
            c,
            (
                np.clip((q[:, 0] / CELL).astype(int), 0, nx - 1),
                np.clip((q[:, 1] / CELL).astype(int), 0, ny - 1),
            ),
            1,
        )
        return c

    tall = count((p[:, 2] > 1.9) & (p[:, 2] < 2.6)) > 4  # below the ceiling (about 3 m)
    low = (count((p[:, 2] > 0.3) & (p[:, 2] < 1.8)) > 8) & ~tall
    floor = count(p[:, 2] < 0.08) > 3
    walls = tall | (~floor & ~low)
    rects = _runs(walls) + _runs(low)
    n_walls = len(_runs(walls))
    rects_a = np.array(rects, float)
    free = np.argwhere(floor & ~walls & ~low)
    rng = np.random.default_rng(seed)
    rooms: list[h.Room] = []
    for _ in range(5000):
        if len(rooms) >= n_pairs:
            break
        a, b = free[rng.integers(len(free), size=2)]
        s, g = (a + 0.5) * CELL, (b + 0.5) * CELL
        if math.dist(s, g) < 1.5:
            continue
        hd = float(rng.uniform(-math.pi, math.pi))
        room = h.Room(rects_a, n_walls, (s[0], s[1], hd), (g[0], g[1]), size, 50_000 + len(rooms))
        if (
            _clear_of(rects_a, s)
            and _clear_of(rects_a, g)
            and h.route(room, True, 0.27)
            and h.route(room, False)
        ):
            rooms.append(room)
    return rooms, _scan_xml(sim, m.faces)


def _runs(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Cells merged into rectangles along x, so the ray cast has hundreds, not thousands."""
    out = []
    for j in range(mask.shape[1]):
        i = 0
        while i < mask.shape[0]:
            if mask[i, j]:
                k = i
                while k < mask.shape[0] and mask[k, j]:
                    k += 1
                out.append((i * CELL, j * CELL, k * CELL, (j + 1) * CELL))
                i = k
            else:
                i += 1
    return out


def _clear_of(rects: np.ndarray, pt, r: float = h.BODY_R + 0.05) -> bool:
    x, y = pt
    return not np.any(
        (rects[:, 0] - r < x)
        & (x < rects[:, 2] + r)
        & (rects[:, 1] - r < y)
        & (y < rects[:, 3] + r)
    )


def _scan_xml(verts: np.ndarray, faces: np.ndarray) -> str:
    vs = " ".join(f"{x:.4f}" for x in verts.ravel())
    fs = " ".join(str(int(i)) for i in faces.ravel())
    fovy = 2 * math.degrees(math.atan(math.tan(math.radians(h.HFOV) / 2) * GRID[1] / GRID[0]))
    return (
        f'<mujoco><visual><global offwidth="{GRID[0]}" offheight="{GRID[1]}"/>'
        '<map znear="0.02" zfar="20"/></visual>'
        f'<asset><mesh name="scan" vertex="{vs}" face="{fs}" inertia="shell"/></asset><worldbody>'
        '<geom type="plane" size="30 30 0.1" pos="0 0 -0.01"/>'
        '<geom type="mesh" mesh="scan" contype="0" conaffinity="0"/>'
        f'<camera name="eye" pos="0 0 {EYE_H}" fovy="{fovy:.4f}"/></worldbody></mujoco>'
    )
