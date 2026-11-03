"""Render closed-loop walker runs (metasight.simwalk) to video for the demo cut.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Each frame: the blindfolded walker in the room (MuJoCo,
walls cut away), every object outlined with its label and size (green while the glasses camera
sees it), the spatial sound source drawn as rings on the floor (beacon pluck rate and gain from
`spatial_sound`), the glasses' view with the objects it sees boxed, and the belt ring with the
felt level of each motor (8, or the built belt's four diagonals). A metrics strip carries
collisions, min clearance, time and false buzzes a minute.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/render_sim_walk.py \
        avoid_chair --layout diag4 --seed 1000 --out data/demo-takes/sim/preview/walk01.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from metasight import hapsim as h
from metasight import simwalk as S
from metasight import spatial_sound as ss

ROOT = Path(__file__).resolve().parent.parent
W, H = 1280, 720
MAIN_W = 860
FPS = 25
BG = (16, 17, 20)
INK = (236, 236, 230)
DIM = (128, 128, 126)
SEEN = (90, 230, 120)
HOT = (255, 128, 40)
SOUND = (90, 180, 255)
TICK = (255, 80, 70)
LAYOUT_NAMES = {
    "eight": "8 motors",
    "diag4": "4 motors: D26 FR, D14 BR, D13 BL, D4 FL",
}


def _font(size: int, mono: bool = False):
    path = "/System/Library/Fonts/SFNSMono.ttf" if mono else "/System/Library/Fonts/SFNS.ttf"
    return ImageFont.truetype(path, size)


F_S, F_M, F_L, F_MONO = _font(14), _font(17), _font(24), _font(14, True)


class Cam:
    """Pinhole projection for a MuJoCo camera of the current frame."""

    def __init__(self, m, d, cam: str, w: int, hh: int):
        i = m.camera(cam).id
        self.pos = d.cam_xpos[i].copy()
        self.rot = d.cam_xmat[i].reshape(3, 3).copy()
        self.f = (hh / 2) / math.tan(math.radians(m.cam_fovy[i]) / 2)
        self.cx, self.cy = w / 2, hh / 2

    def __call__(self, p) -> tuple[float, float] | None:
        q = self.rot.T @ (np.asarray(p, float) - self.pos)
        if q[2] > -0.05:
            return None
        return (self.cx + self.f * q[0] / -q[2], self.cy - self.f * q[1] / -q[2])


def obb_corners(obb) -> np.ndarray:
    cx, cy, cz, hx, hy, hz, yaw = obb
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dz in (-hz, hz):
        for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            out.append((cx + c * dx - s * dy, cy + s * dx + c * dy, cz + dz))
    return np.array(out)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)] + [
    (i, i + 4) for i in range(4)
]


def _amp_col(a: float, base=(38, 38, 44), hot=HOT) -> tuple[int, int, int]:
    a = max(0.0, min(1.0, a))
    return tuple(int(b + (c - b) * a) for b, c in zip(base, hot, strict=True))


def belt_panel(draw: ImageDraw.ImageDraw, ox: int, oy: int, felt, cmd, layout: str) -> None:
    """Top-down belt: the eight planned cells as an inner ring, the physical motors outside,
    filled by the level the skin feels."""
    cx, cy = ox + 210, oy + 168
    draw.text((ox + 16, oy + 10), "Belt", font=F_L, fill=INK)
    draw.text((ox + 16, oy + 40), LAYOUT_NAMES[layout], font=F_S, fill=DIM)
    draw.ellipse([cx - 92, cy - 92, cx + 92, cy + 92], outline=(70, 70, 76), width=10)
    for i, (mode, inten) in enumerate(cmd):
        a0 = -90 + i * 45 - 20
        lvl = inten / 255 if inten else 0.0
        col = (110, 110, 200) if mode == h.MODES["guide"] else _amp_col(lvl, (32, 32, 36))
        draw.arc([cx - 62, cy - 62, cx + 62, cy + 62], a0, a0 + 40, fill=col, width=14)
    for k, d in enumerate(h.MOTOR_DEG[layout]):
        a = math.radians(d - 90)
        px, py = cx + 112 * math.cos(a), cy + 112 * math.sin(a)
        r = 20 + 10 * float(felt[k])
        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            fill=_amp_col(float(felt[k])),
            outline=(90, 90, 96),
            width=2,
        )
        draw.text((px, py), f"{int(round(100 * felt[k]))}", font=F_S, fill=INK, anchor="mm")
    # body and nose
    draw.ellipse([cx - 34, cy - 24, cx + 34, cy + 24], fill=(60, 80, 110))
    draw.polygon([(cx, cy - 40), (cx - 8, cy - 26), (cx + 8, cy - 26)], fill=INK)
    draw.text(
        (cx, cy + 150),
        "inner: planned cells   outer: felt motor level %",
        font=F_S,
        fill=DIM,
        anchor="mm",
    )


def metrics_strip(draw, x0: int, y0: int, items: list[tuple[str, str]]) -> None:
    x = x0
    for k, v in items:
        draw.text((x, y0), k, font=F_S, fill=DIM)
        draw.text((x, y0 + 18), v, font=F_M, fill=INK)
        x += max(110, int(draw.textlength(k, font=F_S)) + 26)


LABEL_M = 2.0  # label what the glasses see, the target and hazard, and anything this close


def _visible(cam: Cam, zbuf: np.ndarray, p, tol: float = 0.06) -> bool:
    """Is the 3D point in front of whatever the room camera rendered at its pixel?"""
    q = cam.rot.T @ (np.asarray(p, float) - cam.pos)
    uv = cam(p)
    if uv is None:
        return False
    u, v = int(uv[0]), int(uv[1])
    if not (0 <= u < zbuf.shape[1] and 0 <= v < zbuf.shape[0]):
        return False
    return -q[2] <= zbuf[v, u] + tol


def _edge(dr, cam: Cam, zbuf: np.ndarray, a, b, col, width: int) -> None:
    """A wireframe edge, hidden where the walker or furniture stands in front of it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / 0.04))
    run: list[tuple[float, float]] = []
    for k in range(n + 1):
        p = a + (b - a) * k / n
        if _visible(cam, zbuf, p):
            run.append(cam(p))
        else:
            if len(run) > 1:
                dr.line(run, fill=col, width=width)
            run = []
    if len(run) > 1:
        dr.line(run, fill=col, width=width)


def _label(dr, anchor, txt: str, font, fill, placed: list, force: bool) -> None:
    """A tag near `anchor` with a leader line, in the first free slot; dropped when every slot
    overlaps another tag (never for the target or the hazard)."""
    tw, th = dr.textlength(txt, font=font), font.size + 4
    ax, ay = anchor
    slots = [(dx, dy) for dy in (-46, -78, -110, 34, 66) for dx in (-tw / 2, 14, -tw - 14)]
    box = None
    for dx, dy in slots:
        bx = (ax + dx - 4, ay + dy - 2, ax + dx + tw + 4, ay + dy + th)
        inside = bx[0] >= 2 and bx[2] <= MAIN_W - 2 and bx[1] >= 2 and bx[3] <= H - 2
        if inside and not any(
            bx[0] < r[2] and bx[2] > r[0] and bx[1] < r[3] and bx[3] > r[1] for r in placed
        ):
            box = bx
            break
    if box is None:
        if not force:
            return
        dx, dy = slots[0]
        box = (ax + dx - 4, ay + dy - 2, ax + dx + tw + 4, ay + dy + th)
    placed.append(box)
    lx = min(max(ax, box[0]), box[2])
    ly = box[1] if box[1] > ay else box[3]
    dr.line([(ax, ay), (lx, ly)], fill=fill, width=1)
    dr.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], fill=fill)
    dr.rectangle(box, fill=(0, 0, 0, 165))
    dr.text((box[0] + 4, box[1] + 2), txt, font=font, fill=fill)


def hull_corners(o: S.Obj) -> np.ndarray:
    """The outline drawn for an object: its OBB axes, fitted to the boxes MuJoCo draws."""
    cx, cy, _, _, _, _, yaw = o.obb
    c, s = math.cos(yaw), math.sin(yaw)
    us, vs, zs = [], [], []
    for bx, by, bz, hx, hy, hz in o.boxes:
        for ex in (-hx, hx):
            for ey in (-hy, hy):
                px, py = bx + ex - cx, by + ey - cy
                us.append(c * px + s * py)
                vs.append(-s * px + c * py)
        zs += [bz - hz, bz + hz]
    u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
    uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
    return obb_corners(
        (
            cx + c * uc - s * vc,
            cy + s * uc + c * vc,
            (min(zs) + max(zs)) / 2,
            (u1 - u0) / 2,
            (v1 - v0) / 2,
            (max(zs) - min(zs)) / 2,
            yaw,
        )
    )


def _pose(f: int, t: float, x: float, y: float, z: float, hd: float, pitch: float) -> dict:
    p = math.radians(pitch)
    fwd = (math.cos(hd) * math.cos(p), math.sin(hd) * math.cos(p), -math.sin(p))
    return {
        "frame": f,
        "t": round(f / FPS, 3),
        "sim_t": round(t, 3),
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z, 4),
        "heading_deg": round(math.degrees(hd), 2),
        "pitch_down_deg": round(pitch, 2),
        "forward": [round(v, 4) for v in fwd],
    }


def _write_pose(path: Path, sc: S.Scenario, layout: str, poses: list[dict]) -> None:
    """Head pose per video frame, for the binaural track. Line 1 is the header: the frame, the
    objects (OBB centre, half sizes, yaw) and what the sound marks; then one line per frame."""
    objs = [
        {
            "name": o.name,
            "label": o.label,
            "centre": [round(v, 3) for v in o.obb[:3]],
            "half": [round(v, 3) for v in o.obb[3:6]],
            "yaw_deg": round(math.degrees(o.obb[6]), 1),
        }
        for o in sc.world.objs
    ]
    head = {
        "scenario": sc.name,
        "layout": layout,
        "fps": FPS,
        "first_frame_t": 0.0,
        "frame": (
            "simwalk floor frame: x, y on the floor, z up, metres; heading counter-clockwise "
            f"from +x. Home map (y up): X = x - {S.OFFSET[0]}, Y = z, Z = -(y - {S.OFFSET[1]})"
        ),
        "target": sc.target,
        "hazard": sc.hazard,
        "objects": objs,
    }
    with path.open("w") as fh:
        fh.write(json.dumps(head) + "\n")
        for p in poses:
            fh.write(json.dumps(p) + "\n")


def render(
    sc: S.Scenario,
    tr: S.Trace,
    layout: str,
    out: Path,
    label: str = "",
    max_s: float | None = None,
    card: list[str] | None = None,
) -> None:
    """`card`: lines for a box at the top right of the room view (the held-out numbers)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_string(S.world_xml(sc.world, body=True, layout=layout))
    d = mujoco.MjData(m)
    rr = mujoco.Renderer(m, H, MAIN_W)
    rd = mujoco.Renderer(m, H, MAIN_W)
    rd.enable_depth_rendering()
    re = mujoco.Renderer(m, 480, 270)
    rs = mujoco.Renderer(m, 192, 108)
    rs.enable_segmentation_rendering()
    opt_room = mujoco.MjvOption()
    opt_room.geomgroup[3] = 0  # walls cut away
    opt_eye = S.eye_option()
    geom_obj = np.full(m.ngeom, -1)
    for g in range(m.ngeom):
        nm = m.geom(g).name
        if nm.startswith("obj"):
            geom_obj[g] = int(nm[3:].split("_")[0])
    motor_ids = [m.geom(f"m{k}").id for k in range(len(h.MOTOR_DEG[layout]))]
    gid = {n: m.geom(n).id for n in ("legL", "legR", "thighL", "thighR", "shinL", "shinR")}
    wb = m.body("walker").mocapid[0]
    hulls = [hull_corners(o) for o in sc.world.objs]
    target = sc.world.obj(sc.target) if sc.target else None
    hazard = sc.world.obj(sc.hazard) if sc.hazard else None
    res = tr.result
    steps = tr.steps[::2]
    if max_s is not None:
        steps = [s for s in steps if s[0] <= max_s]
    frames: list[tuple] = [("walk", s) for s in steps]
    hold = steps[-1]
    if sc.sit and res is not None and res.success and target is not None:
        n_sit = int(1.6 * FPS)
        for k in range(n_sit):
            frames.append(("sit", hold, (k + 1) / n_sit))
    frames += [("end", hold)] * int(1.2 * FPS)
    trail: list[tuple[float, float]] = []
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-preset",
            "medium",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    after = 0
    poses: list[dict] = []
    for fr in frames:
        kind, st = fr[0], fr[1]
        t, x, y, hd, felt, cmd, coll, clear, fb = st
        if kind == "walk":
            after = 0
        else:  # arrived: the belt goes quiet
            after += 1
            felt = felt * math.exp(-after / 4)
            cmd = [(h.MODES["off"], 0)] * 8
        z_off, hd_draw, sit = 0.0, hd, 0.0
        if kind == "sit" or (kind == "end" and sc.sit and res and res.success and target):
            sit = fr[2] if kind == "sit" else 1.0
            cx, cy = target.obb[0], target.obb[1]
            face = math.atan2(y - cy, x - cx)
            a = min(1.0, sit * 1.6)
            x, y = x + (cx - x) * a, y + (cy - y) * a
            hd_draw = hd + ((face - hd + math.pi) % (2 * math.pi) - math.pi) * a
            z_off = -0.45 * max(0.0, (sit - 0.5) / 0.5)
        d.mocap_pos[wb] = (x, y, z_off)
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), hd_draw)
        d.mocap_quat[wb] = q
        seated = sit > 0.75
        for n in ("legL", "legR"):
            m.geom_rgba[gid[n], 3] = 0.0 if seated else 1.0
        for n in ("thighL", "thighR", "shinL", "shinR"):
            m.geom_rgba[gid[n], 3] = 1.0 if seated else 0.0
        for k, g in enumerate(motor_ids):
            col = np.array(_amp_col(float(felt[k]), (30, 30, 36))) / 255
            m.geom_rgba[g, :3] = col
        pitch = sc.pitch_at(t)
        poses.append(_pose(len(poses), t, x, y, S.EYE_H + z_off, hd_draw, pitch))
        m.cam_pos[0] = (x, y, S.EYE_H + z_off)
        m.cam_quat[0] = S.fovsim.cam_quat(hd_draw, pitch)
        mujoco.mj_forward(m, d)
        rr.update_scene(d, camera="room", scene_option=opt_room)
        room_img = rr.render()
        rd.update_scene(d, camera="room", scene_option=opt_room)
        room_z = rd.render()
        re.update_scene(d, camera="eye", scene_option=opt_eye)
        eye_img = re.render()
        rs.update_scene(d, camera="eye", scene_option=opt_eye)
        seg = rs.render()[..., 0]
        ids = geom_obj[np.clip(seg, 0, m.ngeom - 1)]
        ids[seg < 0] = -1
        seen = {int(i): (ids == i) for i in np.unique(ids) if i >= 0 and (ids == i).sum() >= 12}
        if kind == "walk":
            trail.append((x, y))
        cam = Cam(m, d, "room", MAIN_W, H)
        img = Image.new("RGB", (W, H), BG)
        img.paste(Image.fromarray(room_img), (0, 0))
        dr = ImageDraw.Draw(img, "RGBA")
        # trail and goal
        pts = [cam((px, py, 0.01)) for px, py in trail]
        pts = [p for p in pts if p is not None]
        if len(pts) > 1:
            dr.line(pts, fill=(255, 255, 255, 150), width=3)
        gp = cam((*sc.goal, 0.01))
        if gp:
            dr.ellipse(
                [gp[0] - 9, gp[1] - 6, gp[0] + 9, gp[1] + 6], outline=(255, 255, 255, 220), width=2
            )
            dr.text((gp[0] + 12, gp[1] - 8), "goal", font=F_S, fill=INK)
        # object outlines and labels
        for wb_ in sc.world.walls:
            fc = obb_corners((wb_[0], wb_[1], 0.0, wb_[3], wb_[4], 0.0, 0.0))[:4]
            fp = [cam(c) for c in fc]
            if all(p is not None for p in fp):
                dr.polygon(fp, fill=(70, 70, 76, 160), outline=(150, 150, 150, 200))
        placed: list[tuple[float, float, float, float]] = [
            (0, 0, 470, 82),  # title
            (0, H - 56, MAIN_W, H),  # metrics strip
        ]
        if card:
            placed.append((MAIN_W - 330, 0, MAIN_W, 26 + 19 * len(card)))
        for i, o in enumerate(sc.world.objs):
            corners = hulls[i]
            vis = i in seen
            col = (*SEEN, 255) if vis else (200, 200, 200, 110)
            for a, b in EDGES:
                _edge(dr, cam, room_z, corners[a], corners[b], col, 3 if vis else 1)
            near = math.hypot(o.obb[0] - x, o.obb[1] - y) - max(o.obb[3], o.obb[4]) < LABEL_M
            if not (vis or near or o in (target, hazard)):
                continue
            anchor = cam((*o.obb[:2], corners[4][2]))
            if anchor is None or not _visible(cam, room_z, (*o.obb[:2], corners[4][2]), 0.25):
                if o not in (target, hazard):
                    continue
            if anchor is None:
                continue
            fill = (*SEEN, 255) if vis else (215, 215, 215, 230)
            _label(dr, anchor, f"{o.label}  {o.dims}", F_S, fill, placed, o in (target, hazard))
        # spatial sound: the target's beacon (blue) or the hazard's ticks (red), as floor rings
        for src, colr in ((target, SOUND), (hazard, TICK)):
            if src is None:
                continue
            dist = math.hypot(src.obb[0] - x, src.obb[1] - y)
            if src is target:
                per, gain = ss.beacon_period(dist), ss.beacon_gain(dist)
                note = "beacon"
            else:
                per = ss.tick_period(max(0.0, dist - 0.3))
                gain = 1.0 if math.isfinite(per) else 0.0
                note = ss.band(max(0.0, dist - 0.3)) or "quiet"
            if math.isfinite(per) and gain > 0:
                ph = (t % per) / per
                for ring in (ph, (ph + 0.5) % 1.0):
                    rad = 0.15 + 0.6 * ring
                    circ = [
                        cam((src.obb[0] + rad * math.cos(a), src.obb[1] + rad * math.sin(a), 0.02))
                        for a in np.linspace(0, 2 * math.pi, 40)
                    ]
                    if all(c is not None for c in circ):
                        alpha = int(255 * gain * (1 - ring))
                        dr.line(circ + [circ[0]], fill=(*colr, alpha), width=3)
            sp = cam((src.obb[0], src.obb[1], 0.02))
            if sp:
                msg = f'{src.label}: "over here"' if src is target else f"{src.label}: tick {note}"
                _label(dr, sp, msg, F_M, (*colr, 255), placed, True)
        # metrics strip
        dr.rectangle([0, H - 54, MAIN_W, H], fill=(0, 0, 0, 170))
        metrics_strip(
            dr,
            14,
            H - 50,
            [
                ("time", f"{t:4.1f} s"),
                ("collisions", f"{coll}"),
                ("min clearance", f"{max(0.0, clear):.2f} m"),
                ("false buzzes/min", f"{(fb / max(t, 1.0) * 60):.1f}"),
                (
                    "result",
                    "" if kind == "walk" else ("reached" if res and res.success else "timed out"),
                ),
            ],
        )
        dr.text((14, 12), sc.title, font=F_L, fill=INK)
        dr.text((14, 42), "SIMULATED USER MODEL, NOT A HUMAN STUDY", font=F_S, fill=(255, 200, 90))
        if label:
            dr.text((14, 62), label, font=F_S, fill=DIM)
        if card:
            cw = max(dr.textlength(c, font=F_MONO) for c in card) + 20
            bx0 = MAIN_W - cw - 12
            dr.rectangle([bx0, 10, MAIN_W - 12, 18 + 19 * len(card)], fill=(0, 0, 0, 175))
            for k, c in enumerate(card):
                dr.text((bx0 + 10, 16 + 19 * k), c, font=F_MONO, fill=INK if k else HOT)
        # right column: glasses view with detections
        ex, ey = MAIN_W + 8, 8
        eye = Image.fromarray(eye_img).resize((202, 360))
        img.paste(eye, (ex, ey))
        sx, sy = 202 / 108, 360 / 192
        for i, mask in seen.items():
            o = sc.world.objs[i]
            ys, xs = np.nonzero(mask)
            box = [
                ex + xs.min() * sx,
                ey + ys.min() * sy,
                ex + (xs.max() + 1) * sx,
                ey + (ys.max() + 1) * sy,
            ]
            dr.rectangle(box, outline=(*SEEN, 255), width=2)
            dr.text((box[0] + 3, box[1] + 2), o.label, font=F_S, fill=(*SEEN, 255))
        dr.text((ex + 214, ey + 4), "Glasses camera", font=F_M, fill=INK)
        dr.text((ex + 214, ey + 28), f"{S.HFOV:.1f} deg wide", font=F_S, fill=DIM)
        dr.text((ex + 214, ey + 46), f"eye {S.EYE_H:.2f} m", font=F_S, fill=DIM)
        dr.text((ex + 214, ey + 64), f"pitch {pitch:.0f} deg down", font=F_S, fill=DIM)
        dr.text((ex + 214, ey + 100), "depth -> floor fit", font=F_S, fill=DIM)
        dr.text((ex + 214, ey + 118), "-> strips -> belt plan", font=F_S, fill=DIM)
        if target is not None:
            dist = math.hypot(target.obb[0] - x, target.obb[1] - y)
            dr.text((ex + 214, ey + 160), "Sound", font=F_M, fill=SOUND)
            dr.text((ex + 214, ey + 184), f"chair {dist:.1f} m", font=F_S, fill=INK)
            dr.text(
                (ex + 214, ey + 202),
                f"pluck every {ss.beacon_period(dist):.2f} s",
                font=F_S,
                fill=DIM,
            )
        belt_panel(dr, MAIN_W, 360, felt, cmd, layout)
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    _write_pose(out.with_suffix(".pose.jsonl"), sc, layout, poses)
    rr.close()
    re.close()
    rs.close()
    rd.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=S.SCENARIOS)
    ap.add_argument("--layout", default="eight", choices=("eight", "diag4"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    sc = S.scenario(a.scenario)
    tr = S.walk(sc, h.Belt(layout=a.layout, hfov=S.HFOV), seed=a.seed)
    print(a.scenario, a.layout, h.summarize([tr.result]))
    render(sc, tr, a.layout, a.out, a.label)


if __name__ == "__main__":
    main()
