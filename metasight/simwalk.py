"""Closed-loop walker in the reconstructed bedroom, seen through the glasses camera.

SIMULATED USER MODEL, NOT A HUMAN STUDY. The body and the steering are `hapsim`'s blindfolded
walker (it knows the walls and the big furniture of its own room, the rest reaches it only as belt
cues through the skin model). What changes from `hapsim3d` is the eye: depth is rendered by MuJoCo
from the glasses (eye height and field of view from `metasight.glasses.toml`, portrait, pitched
down), goes through the live floor geometry (`safety.analyse` -> `geometry.grid_strips`) and the
real belt planner (`pan.scene_targets` + `pan.plan_pan`, eight motors or the four built
diagonals), and the walker reacts to what it feels.

Worlds are built from map-recon's objects (`fovrooms.scene_boxes`, recon-v2 v2.0), shifted so every
coordinate is positive (hapsim's grids start at 0). A chair model (seat 0.45 m on four thin legs,
back to 0.9 m) stands in for the scan's chair, which the scan only caught as its back. The field of
view catalogue (desk edge, head-height shelf, low box, chair legs) uses `fovsim.SCENES` boxes in a
4 by 3.2 m room with the same walker.

    world, sc = scenario("avoid_chair"); trace = walk(sc, belt, seed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import fovrooms, fovsim
from . import hapsim as h
from .fovmem import HazardMemory, level_floor, loose_floor
from .geometry import grid_strips
from .safety import GRID, analyse

EYE_H = 1.45  # metasight.glasses.toml
HFOV = 44.8
PITCH_DEG = 12.0
OFFSET = (1.6, 1.5)  # bedroom scene frame -> positive coordinates
KNOWN = ("bed", "desk", "nightstand")  # the wearer's own room: these never move
# A detection far smaller than its class can be is mislabelled (desk_18 is 0.35 x 0.15 m):
# it stays an obstacle, named only "object".
MIN_SIDE_M = {"bed": 1.2, "desk": 0.6, "nightstand": 0.3}
COLOURS = {
    "wall": "0.86 0.84 0.80 1",
    "bed": "0.45 0.55 0.70 1",
    "desk": "0.66 0.47 0.30 1",
    "nightstand": "0.60 0.42 0.28 1",
    "chair": "0.93 0.55 0.20 1",
    "floor clutter": "0.85 0.75 0.30 1",
    "box": "0.85 0.75 0.30 1",
    "object": "0.62 0.60 0.56 1",
    "shelf": "0.55 0.40 0.30 1",
}
Box = fovsim.Box


@dataclass
class Obj:
    """One labelled object: its boxes (drawn, felt) and an oriented outline for the overlays
    (centre x, y, z, half sizes along its own axes, yaw)."""

    name: str
    label: str
    boxes: list[Box]
    obb: tuple[float, float, float, float, float, float, float]
    known: bool = False

    @property
    def dims(self) -> str:
        _, _, _, hx, hy, hz, _ = self.obb
        a, b = sorted((2 * hx, 2 * hy), reverse=True)
        return f"{a:.2f} x {b:.2f} x {2 * hz:.2f} m"


@dataclass
class World:
    objs: list[Obj]
    walls: list[Box]
    size: tuple[float, float]

    def obj(self, name: str) -> Obj:
        return next(o for o in self.objs if o.name == name)

    def rects(self) -> tuple[np.ndarray, int]:
        """Footprints for the 2D body: walls and known furniture first (the walker's route knows
        them), then everything it can only feel. Flat things under 6 cm are stepped on."""
        known, unknown = [], []
        for b in self.walls:
            known.append(_foot(b))
        for o in self.objs:
            for b in o.boxes:
                if b[2] + b[5] < 0.06 or b[2] - b[5] > 1.75:
                    continue
                (known if o.known else unknown).append(_foot(b))
        return np.array(known + unknown, float), len(known)


def _foot(b: Box) -> tuple[float, float, float, float]:
    return (b[0] - b[3], b[1] - b[4], b[0] + b[3], b[1] + b[4])


def chair_boxes(cx: float, cy: float, facing_deg: float = 0.0) -> list[Box]:
    """Seat 0.44 m square at 0.45 m on four 3 cm legs, back on the side away from `facing_deg`
    (0 = the seat's front faces +x; only quarter turns)."""
    q = round(facing_deg / 90) % 4
    out = [(cx, cy, 0.45, 0.22, 0.22, 0.02)]
    bx, by = [(-0.2, 0.0), (0.0, -0.2), (0.2, 0.0), (0.0, 0.2)][q]
    hx, hy = (0.02, 0.22) if q % 2 == 0 else (0.22, 0.02)
    out.append((cx + bx, cy + by, 0.69, hx, hy, 0.22))
    for lx in (-0.19, 0.19):
        for ly in (-0.19, 0.19):
            out.append((cx + lx, cy + ly, 0.215, 0.015, 0.015, 0.215))
    return out


def _shift(b: Box, ox: float, oy: float) -> Box:
    return (b[0] + ox, b[1] + oy, *b[2:])


def _obb_of(o: dict, floor_y: float, ox: float, oy: float):
    bb = o["obb"]
    (px, py, pz), (sx, sh, sz) = bb["center"], bb["size"]
    bottom = max(0.0, py - sh / 2 - floor_y)
    top = py + sh / 2 - floor_y
    return (
        px + ox,
        -pz + oy,
        (bottom + top) / 2,
        sx / 2,
        sz / 2,
        (top - bottom) / 2,
        -float(bb["yaw_rad"]),
    )


def bedroom(
    chair: tuple[float, float, float] | None,
    box: tuple[float, float] | None = None,
    aisle_clutter: bool = True,
):
    """The reconstructed bedroom. `chair` (x, y, facing deg) in the scene frame replaces the
    scan's chairs with the chair model; `box` moves the floor clutter in the aisle there (a
    shoebox, 0.30 x 0.22 x 0.12 m); `aisle_clutter=False` takes it out (picked up)."""
    data = fovrooms.load_objects()
    boxes, by_id = fovrooms.scene_boxes(data, open_doors=("door_03",))
    ox, oy = OFFSET
    floor_y = float(data["floor_y"])
    objs: list[Obj] = []
    for o in data["objects"]:
        if o["id"] not in by_id:
            continue
        cls = o["class"]
        if cls == "chair" and chair is not None:
            continue
        side = max(o["size_m"]["width"], o["size_m"]["depth"])
        if side < MIN_SIDE_M.get(cls, 0.0):
            cls = "object"
        if o["id"] == "floor_clutter_07" and (box is not None or not aisle_clutter):
            continue
        objs.append(
            Obj(
                o["id"],
                cls,
                [_shift(b, ox, oy) for b in by_id[o["id"]]],
                _obb_of(o, floor_y, ox, oy),
                cls in KNOWN,
            )
        )
    if chair is not None:
        cx, cy, f = chair[0] + ox, chair[1] + oy, chair[2]
        objs.append(
            Obj("chair", "chair", chair_boxes(cx, cy, f), (cx, cy, 0.45, 0.22, 0.22, 0.45, 0.0))
        )
    if box is not None:
        bx, by = box[0] + ox, box[1] + oy
        b = (bx, by, 0.06, 0.15, 0.11, 0.06)
        objs.append(Obj("box", "box", [b], (bx, by, 0.06, 0.15, 0.11, 0.06, 0.0)))
    n_obj = sum(len(v) for v in by_id.values())
    walls = [_shift(b, ox, oy) for b in boxes[n_obj:]]
    x1 = max(b[0] + b[3] for b in walls) + 0.3
    y1 = max(b[1] + b[4] for b in walls) + 0.3
    return World(objs, walls, (x1, y1))


def catalogue_room(name: str) -> World:
    """A field-of-view catalogue scene (`fovsim.SCENES`) in a 4 x 3.2 m room, shifted by (1, 1.6)
    so the wearer starts at (1, 1.6) facing +x."""
    sc = fovsim.SCENES[name]
    ox, oy = 1.0, 1.6
    labels = {
        "desk_lookup": "desk",
        "shelf_head": "shelf",
        "low_box": "box",
        "chair_legs": "chair",
    }
    objs: list[Obj] = []
    walls: list[Box] = []
    furn: list[Box] = []
    for b in sc.boxes:
        if max(b[3], b[4]) >= 1.5 and b[5] >= 1.0:
            walls.append(_shift(b, ox, oy))
        else:
            furn.append(_shift(b, ox, oy))
    lab = labels[name]
    if name == "chair_legs":
        chair, desk = furn[:6], furn[6:]
        objs.append(Obj("chair", "chair", chair, _bound(chair)))
        objs.append(Obj("desk", "desk", desk, _bound(desk)))
    elif name == "desk_lookup":
        objs.append(Obj("desk", "desk", furn[:5], _bound(furn[:5])))
        objs.append(Obj("monitor", "monitor", furn[5:], _bound(furn[5:])))
    else:
        objs.append(Obj(lab, lab, furn, _bound(furn)))
    x0, x1, y0, y1 = 0.2, 4.2, 0.0, 3.2
    hgt = 1.25
    for b in (
        (x0 - 0.05, (y0 + y1) / 2, hgt, 0.05, (y1 - y0) / 2, hgt),
        (x1 + 0.05, (y0 + y1) / 2, hgt, 0.05, (y1 - y0) / 2, hgt),
        ((x0 + x1) / 2, y0 - 0.05, hgt, (x1 - x0) / 2, 0.05, hgt),
        ((x0 + x1) / 2, y1 + 0.05, hgt, (x1 - x0) / 2, 0.05, hgt),
    ):
        walls.append(b)
    return World(objs, walls, (x1 + 0.4, y1 + 0.4))


def _bound(bs: list[Box]):
    a = np.array(bs)
    lo = (a[:, :3] - a[:, 3:]).min(0)
    hi = (a[:, :3] + a[:, 3:]).max(0)
    c, s = (lo + hi) / 2, (hi - lo) / 2
    return (*c, *s, 0.0)


@dataclass
class Scenario:
    name: str
    title: str
    world: World
    start: tuple[float, float, float]  # x, y, heading rad (counter-clockwise from +x)
    goal: tuple[float, float]
    target: str | None = None  # object that speaks (spatial sound beacon)
    hazard: str | None = None  # object that ticks as a hazard
    guide: bool = False
    sit: bool = False
    pitch: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, PITCH_DEG)])
    max_s: float = 60.0

    def room(self, seed: int = 0) -> h.Room:
        rects, n_walls = self.world.rects()
        return h.Room(rects, n_walls, self.start, self.goal, self.world.size, seed)

    def pitch_at(self, t: float) -> float:
        k = np.array(self.pitch, float)
        return float(np.interp(t, k[:, 0], k[:, 1]))


def scenario(name: str) -> Scenario:
    ox, oy = OFFSET
    if name == "avoid_chair":
        # the seat half under the desk edge leaves a 0.69 m gap to the bed (the body is 0.5);
        # the walker starts 1.2 m short of it, walks up the aisle and out past the desks
        w = bedroom(chair=(-0.02, 0.6, 180.0), aisle_clutter=False)
        return Scenario(
            name,
            "Avoid the chair: pulled out into the aisle",
            w,
            (0.15 + ox, -0.78 + oy, math.pi / 2),
            (0.4 + ox, 1.55 + oy),
            hazard="chair",
        )
    if name == "find_chair":
        w = bedroom(chair=(0.08, 0.05, 0.0), aisle_clutter=False)
        return Scenario(
            name,
            "Find the chair and sit",
            w,
            (1.9 + ox, 1.5 + oy, math.pi),
            (0.6 + ox, 0.05 + oy),
            target="chair",
            guide=True,
            sit=True,
        )
    cat = {
        "desk_lookup": ((1.0, 1.6, 0.0), (2.7, 2.85), [(0, 45.0), (1.5, 45.0), (2.3, 8.0)]),
        "shelf_head": ((1.0, 1.6, 0.0), (3.3, 2.8), [(0, 5.0), (1.2, 10.0), (2.4, 50.0)]),
        "low_box": ((1.0, 1.6, 0.0), (3.8, 1.6), [(0, 15.0)]),
        "chair_legs": ((1.0, 1.6, 0.0), (3.0, 2.9), [(0, 30.0), (2.0, 30.0), (3.0, 0.0)]),
    }
    if name in cat:
        start, goal, pitch = cat[name]
        w = catalogue_room(name)
        titles = {
            "desk_lookup": "Desk edge: looks up from the desk",
            "shelf_head": "Shelf at head height",
            "low_box": "Low box on the floor",
            "chair_legs": "Chair on thin legs",
        }
        return Scenario(name, titles[name], w, start, goal, hazard=w.objs[0].name, pitch=pitch)
    raise KeyError(name)


SCENARIOS = ("avoid_chair", "find_chair", "desk_lookup", "shelf_head", "low_box", "chair_legs")


# ------------------------------------------------------------------ MuJoCo
def eye_option() -> mujoco.MjvOption:
    """What the glasses see: walls (group 3, hidden by MuJoCo's default) and furniture, never
    the walker's own body (group 2)."""
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 0
    opt.geomgroup[3] = 1
    return opt


def _look_quat(pos, target) -> np.ndarray:
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([right, up, -fwd]).ravel())
    return q


def world_xml(world: World, body: bool, layout: str = "eight", w: int = 1280, hh: int = 720):
    """MJCF: walls in geom group 3 (hidden from the room camera), furniture coloured by class,
    and with `body` the walker (mocap bodies, group 2, hidden from the eye) and a room camera."""
    fovy = 2 * math.degrees(math.atan(math.tan(math.radians(HFOV) / 2) * GRID[0] / GRID[1]))
    g = ['<geom type="plane" size="20 20 0.1" material="floor"/>']
    for b in world.walls:
        g.append(_geom(b, COLOURS["wall"], 3))
    for i, o in enumerate(world.objs):
        for k, b in enumerate(o.boxes):
            g.append(_geom(b, COLOURS.get(o.label, "0.7 0.7 0.7 1"), 0, f"obj{i}_{k}"))
    bodies = ""
    cams = f'<camera name="eye" pos="0 0 {EYE_H}" fovy="{fovy:.4f}"/>'
    if body:
        belt = []
        degs = h.MOTOR_DEG[layout]
        for k, d in enumerate(degs):
            a = -math.radians(d)  # clockwise from ahead (+x of the body)
            belt.append(
                f'<geom name="m{k}" type="sphere" size="0.045" group="2" '
                f'pos="{0.2 * math.cos(a):.3f} {0.2 * math.sin(a):.3f} 0.98" '
                'rgba="0.15 0.15 0.18 1"/>'
            )
        bodies = (
            '<body name="walker" mocap="true" pos="0 0 0">'
            '<geom name="torso" type="capsule" fromto="0 0 0.98 0 0 1.32" size="0.17" group="2" '
            'rgba="0.30 0.42 0.62 1"/>'
            '<geom name="belt" type="cylinder" pos="0 0 0.98" size="0.185 0.03" group="2" '
            'rgba="0.12 0.12 0.14 1"/>'
            + "".join(belt)
            + '<geom name="head" type="sphere" pos="0 0 1.56" size="0.11" group="2" '
            'rgba="0.86 0.70 0.58 1"/>'
            '<geom name="blindfold" type="cylinder" pos="0 0 1.575" size="0.113 0.025" '
            'group="2" rgba="0.05 0.05 0.05 1"/>'
            '<geom name="glasses" type="box" pos="0.105 0 1.55" size="0.012 0.075 0.012" '
            'group="2" rgba="0.1 0.1 0.1 1"/>'
            '<geom name="legL" type="capsule" fromto="0 0.09 0.08 0 0.09 0.88" size="0.07" '
            'group="2" rgba="0.22 0.22 0.26 1"/>'
            '<geom name="legR" type="capsule" fromto="0 -0.09 0.08 0 -0.09 0.88" size="0.07" '
            'group="2" rgba="0.22 0.22 0.26 1"/>'
            '<geom name="thighL" type="capsule" fromto="0 0.09 0.5 0.42 0.09 0.5" size="0.07" '
            'group="2" rgba="0.22 0.22 0.26 0"/>'
            '<geom name="thighR" type="capsule" fromto="0 -0.09 0.5 0.42 -0.09 0.5" size="0.07" '
            'group="2" rgba="0.22 0.22 0.26 0"/>'
            '<geom name="shinL" type="capsule" fromto="0.42 0.09 0.5 0.42 0.09 0.06" size="0.06" '
            'group="2" rgba="0.22 0.22 0.26 0"/>'
            '<geom name="shinR" type="capsule" fromto="0.42 -0.09 0.5 0.42 -0.09 0.06" '
            'size="0.06" group="2" rgba="0.22 0.22 0.26 0"/>'
            "</body>"
        )
        sx, sy = world.size
        pos = (-0.4, -0.9, 3.6)
        cams += (
            f'<camera name="room" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'quat="{" ".join(f"{v:.5f}" for v in _look_quat(pos, (sx * 0.5, sy * 0.48, 0.2)))}"'
            ' fovy="58"/>'
        )
    return (
        f'<mujoco><visual><global offwidth="{w}" offheight="{hh}"/>'
        '<quality shadowsize="4096"/><map znear="0.02" zfar="30"/>'
        '<headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual><asset>'
        '<texture name="chk" type="2d" builtin="checker" rgb1="0.80 0.79 0.76" '
        'rgb2="0.72 0.71 0.68" width="64" height="64"/>'
        '<texture type="skybox" builtin="gradient" rgb1="0.30 0.32 0.36" rgb2="0.08 0.08 0.10" '
        'width="64" height="64"/>'
        '<material name="floor" texture="chk" texrepeat="40 40"/></asset><worldbody>'
        f'<light pos="{world.size[0] / 2} {world.size[1] / 2} 4" dir="0 0 -1" '
        'diffuse="0.5 0.5 0.5" castshadow="true"/>'
        + "".join(g)
        + bodies
        + cams
        + "</worldbody></mujoco>"
    )


def _geom(b: Box, rgba: str, group: int, name: str = "") -> str:
    nm = f' name="{name}"' if name else ""
    return (
        f'<geom{nm} type="box" group="{group}" pos="{b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" '
        f'size="{b[3]:.4f} {b[4]:.4f} {b[5]:.4f}" rgba="{rgba}"/>'
    )


class GlassesSensor:
    """`hapsim.run(..., sensor=GlassesSensor(sc))`: the glasses' depth at 25 fps, portrait, through
    the live floor geometry; the side and back sectors from the room's footprints (the map), off
    when `map_sides` is False (as `metasight.glasses.toml` runs). With `memory`, cells that leave
    the view are held (`fovmem.HazardMemory`) and fill the sectors the camera no longer sees."""

    def __init__(
        self,
        sc: Scenario,
        noise: float = 0.04,
        map_sides: bool = True,
        memory: bool = False,
    ):
        self.sc = sc
        self.m = mujoco.MjModel.from_xml_string(world_xml(sc.world, body=False))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, GRID[0], GRID[1])
        self.r.enable_depth_rendering()
        self.opt = eye_option()
        self.noise = noise
        self.map_sides = map_sides
        self.mem = HazardMemory(HFOV, EYE_H, low_slope=0.02) if memory else None
        self.prev = None
        self.t = 0.0
        self.last: dict = {}

    def __call__(self, room: h.Room, x, y, heading, rng) -> tuple[list[float], list[float]]:
        pitch = self.sc.pitch_at(self.t)
        self.m.cam_pos[0] = (x, y, EYE_H)
        self.m.cam_quat[0] = fovsim.cam_quat(heading, pitch)
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera="eye", scene_option=self.opt)
        z = self.r.render().astype(np.float32)
        z[~(z <= 15.0)] = 0.0
        z *= rng.lognormal(0, self.noise) * rng.normal(1, 0.01, z.shape).astype(np.float32)
        scene, self.prev = analyse(z, 1.0, 1.0, HFOV, EYE_H, self.prev)
        strips = [s if s < h.MAX_RANGE else math.inf for s in grid_strips(scene.grid, h.STRIPS)]
        if self.map_sides:
            _, sectors = h.sense(room, x, y, heading, rng)
        else:
            sectors = [math.inf] * 8
        if self.mem is not None:
            fc = self.mem.update(self.t, z, pitch, heading, (y, x), self._mask(z, scene, pitch))
            sectors = [min(a, b) for a, b in zip(sectors, fc.hidden, strict=True)]
        self.last = {"strips": strips, "sectors": sectors, "pitch": pitch}
        self.t += h.SENSE_DT
        return strips, sectors

    def _mask(self, z: np.ndarray, scene, pitch: float) -> np.ndarray:
        """The memory's recommended settings (`run_fov.REC`): only pixels the floor geometry calls
        obstacles may become cells, and none when no floor backs the pitch in use."""
        mask = scene.grid < 255 if scene.floor else None
        if level_floor(z, HFOV, pitch, EYE_H, min_span=0.8) is None and not (
            z.any() and loose_floor(z, HFOV, pitch, EYE_H, pitch)
        ):
            mask = np.zeros(z.shape, bool)
        return mask

    def close(self) -> None:
        self.r.close()


FALSE_TOL_M = 0.5  # a cue is false when nothing real is within off_m + this of its cell


@dataclass
class Trace:
    """What a walk did: per physics step (t, x, y, heading, felt motors, the live command,
    collisions so far, min clearance so far, false onsets so far) when recorded, the result, and
    the cue counts scored against the 3D boxes: false onsets (a cell turns on with nothing within
    reach of it or its neighbours) and flicker (cell mode changes)."""

    steps: list[tuple] = field(default_factory=list)
    result: h.Result | None = None
    false_onsets: int = 0
    flicker: int = 0

    def metrics(self) -> dict:
        r = self.result
        mins = r.time_s / 60
        return {
            "success": bool(r.success),
            "collisions": r.collisions,
            "min_clear_m": round(max(0.0, r.min_clear_m), 3),
            "time_s": round(r.time_s, 2),
            "false_per_min": round(self.false_onsets / mins, 2),
            "flicker_per_min": round(self.flicker / mins, 1),
            "cost": round(100 * (not r.success) + 4 * r.collisions + 0.05 * r.time_s, 3),
        }


def _truth_points(world: World) -> np.ndarray:
    boxes = list(world.walls) + [b for o in world.objs for b in o.boxes]
    pts = np.vstack([fovsim._surface(b, 0.05) for b in boxes])
    return pts[(pts[:, 2] > 0.05) & (pts[:, 2] < EYE_H + 0.15)]


def _true_cells(pts: np.ndarray, x: float, y: float, heading: float) -> np.ndarray:
    """Nearest real surface per belt cell (clockwise from ahead), horizontal metres."""
    d = pts[:, :2] - [x, y]
    fw = np.array([math.cos(heading), math.sin(heading)])
    rt = np.array([math.sin(heading), -math.cos(heading)])
    cf, cr = d @ fw, d @ rt
    dist = np.hypot(cf, cr) - h.BODY_R
    cell = ((np.degrees(np.arctan2(cr, cf)) + 22.5) % 360 // 45).astype(int)
    out = np.full(8, np.inf)
    np.minimum.at(out, cell, dist)
    return out


def walk(
    sc: Scenario,
    belt: h.Belt,
    seed: int = 0,
    map_sides: bool = True,
    memory: bool = False,
    record: bool = True,
) -> Trace:
    room = sc.room(seed)
    sensor = GlassesSensor(sc, map_sides=map_sides, memory=memory)
    tr = Trace()
    rects = room.rects
    clear = [9.0]
    pts = _truth_points(sc.world)
    reach = belt.mapping.off_m + FALSE_TOL_M
    prev: list = [None]

    def on_step(t, x, y, heading, live, felt, coll):
        if prev[0] is not None and live is not prev[0]:
            changed = [k for k in range(8) if live[k][0] != prev[0][k][0]]
            tr.flicker += len(changed)
            onsets = [k for k in changed if live[k][1] > 0 and prev[0][k][1] == 0]
            if onsets:
                near = _true_cells(pts, x, y, heading)
                for k in onsets:
                    if (
                        live[k][0] != h.MODES["guide"]
                        and min(near[(k + j) % 8] for j in (-1, 0, 1)) > reach
                    ):
                        tr.false_onsets += 1
        prev[0] = live
        if not record:
            return
        c = h._clearance(rects, x, y)
        clear[0] = min(clear[0], c)
        tr.steps.append(
            (t, x, y, heading, np.array(felt), list(live), coll, clear[0], tr.false_onsets)
        )

    if sc.guide:
        belt = h.replace(belt, guide="on")
    try:
        tr.result = h.run(room, belt, seed=seed, max_s=sc.max_s, sensor=sensor, on_step=on_step)
    finally:
        sensor.close()
    return tr
