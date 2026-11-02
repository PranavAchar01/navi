"""Field-of-view failure scenes for the glasses camera, rendered in MuJoCo and scored per frame.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Each scene is a few boxes (a desk, a shelf, a chair on
thin legs, a doorway...) and a scripted head path: where the wearer stands, which way the body
and the head point, and how far the head looks down. Depth is rendered from the glasses' eye
(1.6 m, 44.8 by 72 degrees portrait unless set) with monocular-like noise and goes through the
live floor geometry (`safety.analyse`), which is what the belt hears today. `fovmem.HazardMemory`
is the arm under test. Ground truth is the boxes themselves, sampled every 2 cm.

Score per frame and belt sector (the head's frame, as the server sends it): a DANGER is a true
hazard nearer than 1 m; it is MISSED when the cue reports nothing or something more than 0.3 m
further. A FALSE cue reports under 1 m where the truth is at least 0.5 m further. The same for the
walking path (0.9 m wide, straight ahead of the head). Scene specs are plain data (`SCENES`) so
the closed-loop walker (sim-montage) can load the same boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from .fovmem import (
    CORRIDOR_HALF_M,
    DropMemory,
    HazardMemory,
    PitchFuse,
    band_name,
    gate_memory_cues,
    level_floor,
    loose_floor,
)
from .safety import GRID, analyse

EYE_H = 1.6
STILL_SPEED_MPS = 0.15  # the walker is still under this speed (no step)
HFOV = 44.8
FPS = 13.0
DANGER_M = 1.0
MISS_TOL_M = 0.3
FALSE_TOL_M = 0.5
REACH_M = 2.0
CAMERA_SECTORS = (7, 0, 1)

Box = tuple[float, float, float, float, float, float]  # centre x, y, z and half sizes, metres


@dataclass
class Scene:
    """`boxes` are drawn and are hazards; `unseen` are hazards the depth cannot see (glass);
    `movers` are (box, start time, end time, dx, dy): a box that slides (a person stepping in).
    `keys` are (t, x, y, body heading deg, head yaw offset deg, pitch deg below level), linear
    in between. MuJoCo axes: x ahead at the start, y to the left, z up; heading counter-clockwise
    from +x."""

    name: str
    note: str
    boxes: list[Box]
    keys: list[tuple[float, float, float, float, float, float]]
    unseen: list[Box] = field(default_factory=list)
    movers: list[tuple[Box, float, float, float, float]] = field(default_factory=list)
    blur_dps: float = math.inf  # head turns faster than this smear the frame to nothing
    # Floors that end: `slabs` are (x from, x to, top height) raised floors (walkable, drawn, not
    # hazards), `edges` the x of each step down ahead, `ground_z` the lowest floor. The walker
    # stands on the slab whose top is 0.
    slabs: list[tuple[float, float, float]] = field(default_factory=list)
    edges: list[float] = field(default_factory=list)
    ground_z: float = 0.0

    @property
    def duration(self) -> float:
        return self.keys[-1][0]

    def at(self, t: float) -> tuple[float, float, float, float, float]:
        k = np.array(self.keys, float)
        return tuple(float(np.interp(t, k[:, 0], k[:, i])) for i in range(1, 6))


def _desk(x0: float, depth: float = 0.7, half_w: float = 0.8, top: float = 0.75) -> list[Box]:
    cx = x0 + depth / 2
    out = [(cx, 0.0, top - 0.02, depth / 2, half_w, 0.02)]
    for lx in (x0 + 0.04, x0 + depth - 0.04):
        for ly in (-half_w + 0.04, half_w - 0.04):
            out.append((lx, ly, (top - 0.04) / 2, 0.02, 0.02, (top - 0.04) / 2))
    return out


def _wall_x(x: float, half_w: float = 3.0, h: float = 2.5) -> Box:
    return (x + 0.05, 0.0, h / 2, 0.05, half_w, h / 2)


def _chair(cx: float, cy: float) -> list[Box]:
    out = [(cx, cy, 0.45, 0.22, 0.22, 0.02), (cx + 0.2, cy, 0.7, 0.02, 0.22, 0.23)]
    for lx in (cx - 0.19, cx + 0.19):
        for ly in (cy - 0.19, cy + 0.19):
            out.append((lx, ly, 0.215, 0.015, 0.015, 0.215))
    return out


SCENES: dict[str, Scene] = {
    s.name: s
    for s in [
        Scene(
            "desk_lookup",
            "His example: the desk edge is seen while looking down, then he looks up at the "
            "monitor and steps in; the edge is under the frame at hip height.",
            _desk(1.3) + [(1.85, 0.0, 0.95, 0.03, 0.3, 0.2), _wall_x(2.1)],
            [
                (0, 0, 0, 0, 0, 45),
                (1.5, 0.5, 0, 0, 0, 45),
                (2.3, 0.75, 0, 0, 0, 8),
                (5.0, 0.95, 0, 0, 0, 5),
            ],
        ),
        Scene(
            "shelf_head",
            "A shelf at head height seen from across the room; walking up to it the wearer "
            "looks at the floor, and the shelf is above the top of the frame.",
            [(2.3, 0.0, 1.62, 0.2, 0.5, 0.025), _wall_x(2.5)],
            [
                (0, 0, 0, 0, 0, 5),
                (1.2, 0.4, 0, 0, 0, 10),
                (2.4, 0.9, 0, 0, 0, 50),
                (4.5, 1.65, 0, 0, 0, 50),
            ],
        ),
        Scene(
            "low_box",
            "A shoebox on the floor ahead, seen from 2 m; walking up with the head level it "
            "drops under the bottom edge at about 1.2 m.",
            [(1.7, 0.0, 0.06, 0.15, 0.15, 0.06)],
            [(0, 0, 0, 0, 0, 15), (3.5, 1.3, 0, 0, 0, 15), (5.0, 1.35, 0, 0, 0, 12)],
        ),
        Scene(
            "chair_legs",
            "A chair on thin legs, seat at 0.45 m: approached looking down, then the head comes "
            "up to look for the table beyond it.",
            _chair(1.5, 0.0) + _desk(2.4),
            [
                (0, 0, 0, 0, 0, 30),
                (2.0, 0.8, 0, 0, 0, 30),
                (3.0, 1.0, 0, 0, 0, 0),
                (4.5, 1.1, 0, 0, 0, 0),
            ],
        ),
        Scene(
            "doorway_turn",
            "Walking through a 0.8 m doorway while looking 70 degrees to the right (someone "
            "called): the jambs left the view a step before the shoulders pass them.",
            [(1.6, 0.5, 1.1, 0.05, 0.1, 1.1), (1.6, -0.5, 1.1, 0.05, 0.1, 1.1)],
            [
                (0, 0, 0, 0, 0, 15),
                (1.5, 0.6, 0, 0, 0, 15),
                (2.1, 0.85, 0, 0, -70, 15),
                (4.5, 1.9, 0, 0, -70, 15),
            ],
        ),
        Scene(
            "turn_back",
            "Standing at a table, turning the head away 90 degrees to talk, then turning the "
            "body back toward the table and stepping in: the edge is at the side, then ahead.",
            _desk(0.9, depth=0.8, half_w=0.6),
            [
                (0, 0.3, 0, 0, 0, 35),
                (1.0, 0.3, 0, 0, 0, 35),
                (1.6, 0.3, 0, 0, 90, 15),
                (3.0, 0.3, 0, 0, 90, 15),
                (3.6, 0.3, 0, 0, 30, 15),
                (5.0, 0.55, 0, 0, 30, 15),
            ],
        ),
        Scene(
            "fast_turn_blur",
            "A fast head sweep (about 200 deg/s) across a table at hip height: the frames in "
            "the sweep are smeared and read as nothing.",
            _desk(0.8, depth=0.6, half_w=0.5),
            [
                (0, 0.2, 0, 0, 0, 30),
                (1.0, 0.2, 0, 0, 0, 30),
                (1.4, 0.2, 0, 0, 80, 25),
                (1.8, 0.2, 0, 0, 0, 25),
                (2.2, 0.2, 0, 0, -80, 25),
                (2.6, 0.2, 0, 0, 0, 25),
                (4.0, 0.3, 0, 0, 0, 30),
            ],
            blur_dps=120.0,
        ),
        Scene(
            "side_entry",
            "Someone steps into the path from the side, inside 1 m: they were never in the "
            "view before, so memory cannot help (a control).",
            [],
            [(0, 0, 0, 0, 0, 20), (4.0, 0.6, 0, 0, 0, 20)],
            movers=[((1.3, 1.6, 0.85, 0.2, 0.2, 0.85), 1.0, 3.0, 0.0, -1.6)],
        ),
        Scene(
            "drop_step",
            "The floor ends 2 m ahead in a 0.18 m step down; approached with the head level the "
            "edge leaves the bottom of the view at about 1.5 m (under the 25 cm drop line).",
            [],
            [(0, -1.0, 0, 0, 0, 10), (5.5, 1.3, 0, 0, 0, 10), (7.0, 1.35, 0, 0, 0, 10)],
            slabs=[(-20.0, 2.0, 0.0)],
            edges=[2.0],
            ground_z=-0.18,
        ),
        Scene(
            "stairs_down",
            "Two steps down at 2.0 and 2.6 m (0.18 m each), head level.",
            [],
            [(0, -1.0, 0, 0, 0, 10), (5.5, 1.3, 0, 0, 0, 10), (7.0, 1.35, 0, 0, 0, 10)],
            slabs=[(-20.0, 2.0, 0.0), (2.0, 2.6, -0.18)],
            edges=[2.0, 2.6],
            ground_z=-0.36,
        ),
        Scene(
            "glass_pane",
            "A glass door the depth cannot see (a control: nothing to remember).",
            [],
            [(0, 0, 0, 0, 0, 15), (4.0, 1.2, 0, 0, 0, 15)],
            unseen=[(1.6, 0.0, 1.0, 0.01, 0.6, 1.0)],
        ),
    ]
}


def _colour(b: Box) -> str:
    # Walls pale, furniture wood: only the colour view cares, depth is the same.
    return "0.86 0.84 0.8 1" if max(b[3:5]) >= 1.5 and b[5] >= 1.0 else "0.66 0.47 0.3 1"


def _box_xml(b: Box, name: str = "") -> str:
    nm = f' name="{name}"' if name else ""
    return (
        f'<geom{nm} type="box" pos="{b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" '
        f'size="{b[3]:.4f} {b[4]:.4f} {b[5]:.4f}" rgba="{_colour(b)}"/>'
    )


def scene_xml(sc: Scene, w: int, h: int, hfov: float) -> str:
    fovy = 2 * math.degrees(math.atan(math.tan(math.radians(hfov) / 2) * h / w))
    geoms = [f'<geom type="plane" size="20 20 0.1" pos="0 0 {sc.ground_z}" material="floor"/>']
    for x0, x1, top in sc.slabs:
        geoms.append(
            f'<geom type="box" pos="{(x0 + x1) / 2:.4f} 0 {(top + sc.ground_z) / 2:.4f}" '
            f'size="{(x1 - x0) / 2:.4f} 6 {(top - sc.ground_z) / 2:.4f}" rgba="0.8 0.8 0.78 1"/>'
        )
    geoms += [_box_xml(b) for b in sc.boxes]
    bodies = "".join(
        f'<body name="mover{i}" mocap="true" pos="{m[0][0]} {m[0][1]} {m[0][2]}">'
        f'<geom type="box" size="{m[0][3]} {m[0][4]} {m[0][5]}" rgba="0.3 0.45 0.7 1"/></body>'
        for i, m in enumerate(sc.movers)
    )
    return (
        f'<mujoco><visual><global offwidth="{max(w, 540)}" offheight="{max(h, 960)}"/>'
        '<map znear="0.02" zfar="20"/></visual><asset>'
        '<texture name="chk" type="2d" builtin="checker" rgb1="0.82 0.82 0.8" '
        'rgb2="0.7 0.7 0.68" width="64" height="64"/>'
        '<material name="floor" texture="chk" texrepeat="20 20"/></asset><worldbody>'
        '<light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>'
        + "".join(geoms)
        + bodies
        + f'<camera name="eye" pos="0 0 {EYE_H}" fovy="{fovy:.4f}"/></worldbody></mujoco>'
    )


def _surface(b: Box, step: float = 0.02) -> np.ndarray:
    """Points on a box's surface, about `step` apart."""
    cx, cy, cz, sx, sy, sz = b
    out = []
    ax = [
        np.linspace(c - s, c + s, max(2, int(2 * s / step) + 1))
        for c, s in ((cx, sx), (cy, sy), (cz, sz))
    ]
    for i in range(3):
        for sgn in (-1, 1):
            o = [a for j, a in enumerate(ax) if j != i]
            g = np.meshgrid(*o, indexing="ij")
            c = np.full(g[0].shape, (cx, cy, cz)[i] + sgn * (sx, sy, sz)[i])
            pts = [None, None, None]
            pts[i] = c
            rest = [j for j in range(3) if j != i]
            pts[rest[0]], pts[rest[1]] = g
            out.append(np.stack([p.ravel() for p in pts], 1))
    return np.vstack(out)


def cam_quat(heading: float, pitch_deg: float) -> np.ndarray:
    p = math.radians(pitch_deg)
    fwd = np.array([math.cos(heading) * math.cos(p), math.sin(heading) * math.cos(p), -math.sin(p)])
    right = np.array([math.sin(heading), -math.cos(heading), 0.0])
    z = -fwd
    y = np.cross(z, right)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([right, y, z]).ravel())
    return q


def truth(pts: np.ndarray, x: float, y: float, yaw: float, eye_h: float = EYE_H):
    """Per belt sector, nearest hazard point (horizontal metres) in the head's frame, and the
    nearest in the walking path; points outside the body's height band are ignored."""
    fw = np.array([math.cos(yaw), math.sin(yaw)])
    rt = np.array([math.sin(yaw), -math.cos(yaw)])
    sec = [math.inf] * 8
    path = math.inf
    if not len(pts):
        return sec, path
    band = (pts[:, 2] > 0.05) & (pts[:, 2] < eye_h + 0.15)
    q = pts[band]
    d = q[:, :2] - [x, y]
    cf, cr = d @ fw, d @ rt
    dist = np.hypot(cf, cr)
    near = dist < REACH_M
    s = ((np.degrees(np.arctan2(cr, cf)) + 22.5) % 360 // 45).astype(int)
    for i in range(8):
        m = near & (s == i)
        if m.any():
            sec[i] = float(dist[m].min())
    pm = (np.abs(cr) < CORRIDOR_HALF_M) & (cf > 0) & (cf < REACH_M)
    if pm.any():
        path = float(cf[pm].min())
    return sec, path


def drop_truth(edges: list[float], x: float, y: float, yaw: float) -> float:
    """Metres along the walking path to the nearest floor edge ahead (inf: none)."""
    best = math.inf
    ys = np.arange(-3.0, 3.0001, 0.05)
    for ex in edges:
        d = np.column_stack([np.full(len(ys), ex), ys]) - [x, y]
        cf = d @ [math.cos(yaw), math.sin(yaw)]
        cr = d @ [math.sin(yaw), -math.cos(yaw)]
        m = (np.abs(cr) < CORRIDOR_HALF_M) & (cf > 0) & (cf < REACH_M)
        if m.any():
            best = min(best, float(cf[m].min()))
    return best


@dataclass
class Arms:
    """Which cue sources are scored. `pos`: the memory's position source, "true", "none" (the
    head is assumed to stay put) or "speed" (dead reckoning on a speed 15% off)."""

    memory: bool = True
    pos: str = "true"
    mask: bool = False
    keep_s: float = 6.0
    min_hits: int = 2
    pitch_sd: float = 2.0
    pitch_bias: float = 3.0
    yaw_drift_dps: float = 0.5
    fuse: bool = True  # learn the accelerometer's pitch offset from the floor while in view
    snap: bool = True  # with a floor in view, take its pitch as is
    low_slope: float = 0.04  # low cells need this much more height per metre of range
    band_lo: float | None = None  # lowest height (m) a cell is held at; None is HazardMemory's 0.08
    level: bool = True  # fovmem.level_floor from the pitch guess, not safety's floor plane
    people: bool = False  # a person detector's box is kept out of the memory (people move)
    gate: str | None = None  # HazardMemory report gate: None, "hip_pitch" or "moved"
    # Add new cells only on frames where the pitch is backed by a floor: None (always),
    # "level" (level_floor fit this frame) or "mask" (safety found its floor plane).
    trust: str | None = None
    max_range_m: float = 2.5  # new cells only from points nearer than this
    min_span: float | None = None  # level_floor: the floor seen must run this far (m)
    # trust "level" also on a frame with no fit of its own when a no-span fit agrees (L24)
    loose: bool = False
    # L27 cue gate on the remembered cues, letters of "sar": s still mute, a approach gate, r rear
    # mute (fovmem.gate_memory_cues). "" is the ungated memory.
    cue_gate: str = ""
    approach_m: float = 0.1
    still_min_s: float = 1.0


@dataclass
class Score:
    danger: int = 0
    missed: int = 0
    false: int = 0
    # A false cue with nothing in the neighbouring sectors either: not a hazard put one sector
    # over by a 0.1 m cell error at a sector edge (fovsim doorway_turn, the jamb behind).
    false_far: int = 0
    frames: int = 0
    path_danger: int = 0
    path_missed: int = 0
    front_danger: int = 0
    front_missed: int = 0
    # Drops (a floor edge ahead in the walking path), scored apart from the hazard counts.
    drop_danger: int = 0
    drop_missed: int = 0
    drop_false: int = 0

    def add(self, o: Score) -> None:
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(o, k))

    def rates(self) -> dict:
        def r(a, b):
            return round(a / b, 3) if b else None

        return {
            "miss": r(self.missed, self.danger),
            "front_miss": r(self.front_missed, self.front_danger),
            "path_miss": r(self.path_missed, self.path_danger),
            "false_per_min": round(self.false / (self.frames / FPS / 60), 1)
            if self.frames
            else None,
            "false_far_per_min": round(self.false_far / (self.frames / FPS / 60), 1)
            if self.frames
            else None,
            "drop_miss": r(self.drop_missed, self.drop_danger),
            "drop_false_per_min": round(self.drop_false / (self.frames / FPS / 60), 1)
            if self.frames
            else None,
            "drop_danger": self.drop_danger,
            "danger": self.danger,
            "frames": self.frames,
        }


def _miss(true: float, cue: float) -> bool:
    return true < DANGER_M and not cue <= true + MISS_TOL_M


def _merge(a: Arms, live, path_m, fc, still_s: float, eye_h: float):
    """Live cue merged with the (gated) remembered cues, as the server's `min` merge."""
    hid, path = fc.hidden, fc.path_m
    if a.cue_gate:
        g = a.cue_gate
        band = [band_name(h, eye_h) if math.isfinite(h) else None for h in fc.hidden_h]
        hid_n = [h if math.isfinite(h) else None for h in fc.hidden]
        close = [c if math.isfinite(c) else None for c in fc.hidden_close]
        gh, _, gp = gate_memory_cues(
            hid_n,
            band,
            close,
            still_s,
            path=fc.path_m,
            path_close=fc.path_close,
            mute_still="s" in g,
            approach="a" in g,
            mute_rear="r" in g,
            min_close_m=a.approach_m,
            still_min_s=a.still_min_s,
        )
        hid = [math.inf if h is None else h for h in gh]
        path = math.inf if gp is None else gp
    return [min(lv, hv) for lv, hv in zip(live, hid, strict=True)], min(path_m, path)


def run(
    sc: Scene,
    arms: list[Arms],
    seed: int = 0,
    hfov: float = HFOV,
    noise: float = 0.04,
    eye_h_cfg: float = EYE_H,
    trace: list | None = None,
    record: list | None = None,
) -> list[Score]:
    """Play one scene once; score the live cue alone (first) and live plus each memory arm.
    `eye_h_cfg` is the eye height the software believes (the glasses profile says 1.22). Each
    false cue is appended to `trace` as (t, arm index, sector, cue, truth) when one is given.
    With `record`, each frame appends a dict: the camera's colour view (540x960), the truth and
    every cue (sectors, path), for the demo renders."""
    rng = np.random.default_rng(seed)
    w, h = GRID[1], GRID[0]  # portrait, as the glasses stream
    m = mujoco.MjModel.from_xml_string(scene_xml(sc, w, h, hfov))
    dd = mujoco.MjData(m)
    r = mujoco.Renderer(m, h, w)
    r.enable_depth_rendering()
    rc = mujoco.Renderer(m, 960, 540) if record is not None else None
    pts = (
        np.vstack([_surface(b) for b in sc.boxes + sc.unseen])
        if sc.boxes or sc.unseen
        else np.zeros((0, 3))
    )
    mover_pts = [_surface(mv[0]) - np.array(mv[0][:3]) for mv in sc.movers]
    mover_geoms = [
        g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith("mover")
    ]
    jitter = rng.normal(0, [0.05, 0.05, 2.0, 3.0, 3.0])
    mems = [
        HazardMemory(
            hfov,
            eye_h_cfg,
            keep_s=a.keep_s,
            min_hits=a.min_hits,
            low_slope=a.low_slope,
            gate=a.gate,
            band=None if a.band_lo is None else (a.band_lo, eye_h_cfg + 0.15),
            max_range_m=a.max_range_m,
        )
        for a in arms
    ]
    yaw_err = [0.0] * len(arms)
    dr_pos = [np.zeros(2) for _ in arms]
    # Each arm draws its sensor errors from its own stream, the same for every arm, so arms are
    # paired: changing one arm's settings never changes what the others see.
    arng = [np.random.default_rng([seed, 1]) for _ in arms]
    p_bias = [g.normal(0, a.pitch_bias) for g, a in zip(arng, arms, strict=True)]
    fuses = [PitchFuse(snap=a.snap) for a in arms]
    dmems = [DropMemory(hfov, eye_h_cfg) for _ in arms]
    scores = [Score() for _ in range(len(arms) + 1)]
    prev_plane, plane_t, hold = None, -math.inf, None
    still_t0 = 0.0  # since when the walker has been under STILL_SPEED_MPS
    scale = 1.0  # the server's running depth scale (SafetyLoop._geometry): the floor is the ruler
    last = None
    t = 0.0
    while t <= sc.duration:
        x, y, hd, off, pitch = np.array(sc.at(t)) + jitter
        yaw = math.radians(hd + off)
        # movers
        all_pts = pts
        for i, mv in enumerate(sc.movers):
            b, t0, t1, dx, dy = mv
            a = float(np.clip((t - t0) / (t1 - t0), 0, 1))
            c = np.array([b[0] + a * dx, b[1] + a * dy, b[2]])
            m.body_pos[m.body(f"mover{i}").id] = c
            all_pts = np.vstack([all_pts, mover_pts[i] + c])
        dd.mocap_pos[:] = [
            m.body_pos[m.body(f"mover{i}").id] for i in range(len(sc.movers))
        ] or dd.mocap_pos
        m.cam_pos[0] = (x, y, EYE_H)
        m.cam_quat[0] = cam_quat(yaw, pitch)
        mujoco.mj_forward(m, dd)
        r.update_scene(dd, camera="eye")
        z = r.render().astype(np.float32)
        person = None
        if sc.movers and any(a.people for a in arms):
            # What a person detector would give: the box around every mover pixel.
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
            seg = r.render()[..., 0]
            r.disable_segmentation_rendering()
            r.enable_depth_rendering()
            person = np.zeros(z.shape, bool)
            ys, xs = np.nonzero(np.isin(seg, mover_geoms))
            if len(ys):
                person[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
        z[~(z <= 15.0)] = 0.0
        z *= rng.lognormal(0, noise) * rng.normal(1, 0.01, z.shape).astype(np.float32)
        rate = 0.0
        if last is not None:
            rate = abs(math.degrees(yaw - last[2])) * FPS
            if math.hypot(x - last[0], y - last[1]) * FPS >= STILL_SPEED_MPS:
                still_t0 = t
        still_s = t - still_t0
        if rate > sc.blur_dps:
            z[:] = 0.0
        prev = prev_plane if t - plane_t < 1.5 else None
        hold_n = hold if t - plane_t < 5.0 else None
        scene, plane = analyse(z, scale, 1.0, hfov, eye_h_cfg, prev, hold_n)
        z_m = z * scale
        if plane is not None:
            prev_plane, plane_t, hold = plane, t, plane[0]
            meas = float(np.clip(eye_h_cfg / plane[1], 0.8, 1.25))
            scale = 0.85 * scale + 0.15 * meas
            z_m = z * meas  # this frame's own floor scale, for the memory
            scene, _ = analyse(z, scale, 1.0, hfov, eye_h_cfg, plane)
        live = [math.inf] * 8
        for k, s in enumerate(CAMERA_SECTORS):
            live[s] = scene.sectors[k]
        true_sec, true_path = truth(all_pts, x, y, yaw)
        cues = [(live, scene.path_m)]
        drop_true = drop_truth(sc.edges, x, y, yaw)
        drop_cues = [scene.drop_m]
        shared: dict = {}  # arms that differ only in their cue gate share one memory run
        for k, (a, mem) in enumerate(zip(arms, mems, strict=True)):
            mkey = repr(replace(a, cue_gate=""))
            if mkey in shared:
                fc, dm, lf = shared[mkey]
                drop_cues.append(min(scene.drop_m, dm) if a.memory else scene.drop_m)
                cues.append(_merge(a, live, scene.path_m, fc, still_s, eye_h_cfg))
                continue
            yaw_err[k] += math.radians(a.yaw_drift_dps) / FPS * arng[k].choice([-1, 1])
            if a.pos == "true":
                pos = (y, x)
            elif a.pos == "none":
                pos = (float(sc.keys[0][2]), float(sc.keys[0][1]))
            else:
                if last is not None:
                    dx, dy = x - last[0], y - last[1]
                    dr_pos[k] += np.array([dy, dx]) * 1.15
                pos = tuple(dr_pos[k] + [sc.keys[0][2], sc.keys[0][1]])
            mask = (scene.grid < 255) if (a.mask and scene.floor) else None
            if a.people and person is not None:
                mask = ~person if mask is None else mask & ~person
            if a.trust == "mask" and not (a.mask and scene.floor):
                mask = np.zeros(z.shape, bool)
            imu = pitch + p_bias[k] + arng[k].normal(0, a.pitch_sd)
            zk = z_m
            lf = None
            if a.level:
                guess = imu + (fuses[k].offset or 0.0)
                if z.any():
                    eh = eye_h_cfg / scale
                    lf = level_floor(z, hfov, guess, eh, min_span=a.min_span)
                n = None
                if lf is not None:
                    p_r = math.radians(lf[0])
                    n = np.array([0.0, -math.cos(p_r), -math.sin(p_r)])
                    zk = z * float(np.clip(eye_h_cfg / lf[1], 0.8, 1.25))
                imu = fuses[k].update(imu, n) if a.fuse else imu
                if a.trust == "level" and lf is None:
                    if not (
                        a.loose and z.any() and loose_floor(z, hfov, guess, eye_h_cfg / scale, imu)
                    ):
                        mask = np.zeros(z.shape, bool)
            elif a.fuse:
                imu = fuses[k].update(imu, plane[0] if plane is not None else None)
            fc = mem.update(
                t,
                zk,
                imu,
                yaw + yaw_err[k],
                pos,
                mask,
            )
            lv_k, path_k = live, scene.path_m
            # The edge memory adds edges only while the pitch is backed by the floor: a fit this
            # frame, or a looser fit (no span check: a drop leaves only a strip of floor) that
            # agrees with the pitch in use.
            backed = lf is not None or not a.level
            if not backed and z.any():
                ld = level_floor(z, hfov, guess, eye_h_cfg / scale)
                backed = ld is not None and abs(ld[0] - imu) < 2.0
            dm = dmems[k].update(t, zk, imu, yaw + yaw_err[k], pos, add=backed)
            shared[mkey] = (fc, dm, lf)
            drop_cues.append(min(scene.drop_m, dm) if a.memory else scene.drop_m)
            if not a.memory:
                cues.append((lv_k, path_k))
                continue
            cues.append(_merge(a, lv_k, path_k, fc, still_s, eye_h_cfg))
        for ai, (sco, (sec, path)) in enumerate(zip(scores, cues, strict=True)):
            sco.frames += 1
            for s in range(8):
                if true_sec[s] < DANGER_M:
                    sco.danger += 1
                    sco.missed += _miss(true_sec[s], sec[s])
                    if s in CAMERA_SECTORS:
                        sco.front_danger += 1
                        sco.front_missed += _miss(true_sec[s], sec[s])
                if sec[s] < DANGER_M and true_sec[s] > sec[s] + FALSE_TOL_M:
                    sco.false += 1
                    near = min(true_sec[(s + j) % 8] for j in (-1, 0, 1))
                    sco.false_far += near > sec[s] + FALSE_TOL_M
                    if trace is not None:
                        trace.append((round(t, 2), ai, s, round(sec[s], 2), true_sec[s]))
            dc = drop_cues[ai]
            if drop_true < DANGER_M:
                sco.drop_danger += 1
                sco.drop_missed += _miss(drop_true, dc)
            if dc < DANGER_M and drop_true > dc + FALSE_TOL_M:
                sco.drop_false += 1
            if true_path < DANGER_M:
                sco.path_danger += 1
                sco.path_missed += _miss(true_path, path)
        if rc is not None:
            rc.update_scene(dd, camera="eye")
            record.append(
                {
                    "t": t,
                    "rgb": rc.render(),
                    "truth": (true_sec, true_path),
                    "cues": cues,
                    "blind": bool(rate > sc.blur_dps),
                }
            )
        last = (x, y, yaw)
        t += 1.0 / FPS
    if rc is not None:
        rc.close()
    r.close()
    return scores
