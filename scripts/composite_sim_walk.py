"""Draw the overlay on the Blender render of a simulated walk (scripts/blender_sim_room.py).

Per frame: the floor the glasses camera can see as a soft cone, a thin oriented box around every
object (cyan while it is in view, amber once it has left the view and is only remembered), one
label per nearby object with its measured size and a leader line, sound rings from where the
spatial audio plays (the same schedule as scripts/render_sim_sound.py), the 3D belt panel, the
walk's numbers, and the corner label.

    PYTHONPATH=. .venv/bin/python scripts/composite_sim_walk.py SCENE.json RENDER.png \
        --frame 60 --belt BELT.png --out hero.png
    PYTHONPATH=. .venv/bin/python scripts/composite_sim_walk.py SCENE.json FRAMES_DIR \
        --belt-dir BELT_FRAMES --out OUT_DIR

The camera comes from the render's .cam.json (or cam.json in the frames folder).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from metasight import simwalk as S
from metasight import spatial_sound as ss

SS = 2  # supersampling for the line work
FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
IN_VIEW = (110, 214, 255)
REMEMBERED = (255, 178, 64)
RING = (255, 255, 255)
VFOV = 2 * math.degrees(math.atan(math.tan(math.radians(S.HFOV) / 2) * 9 / 16))
SEE_M = 4.5  # the depth range the glasses detector uses
LABEL_M = 2.0
VIEW_LABEL_M = 3.0
MAX_LABELS = 5
RING_S = 1.1
OUTLINE_M = 1.5  # remembered objects keep their outline this close to the walker
FADE_M = 0.5  # and fade out over this much further
HERO = (255, 236, 190)  # the warm glow under the scenario's target
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]  # belt cells, clockwise from ahead
SIT_DROP_M = 0.5
NAMES = {
    "bed_00": "bed",
    "desk_01": "standing desk",
    "desk_02": "folding table",
    "desk_18": "tower fan",
    "nightstand_15": "nightstand",
    "chair": "office chair",
    "box": "shoebox",
}
LABEL = "SIMULATED USER MODEL, NOT A HUMAN STUDY"


class Cam:
    def __init__(self, path: Path):
        c = json.loads(path.read_text())
        self.m = np.array(c["ndc_from_world"], float)
        self.w, self.h = c["w"], c["h"]

    def px(self, p) -> tuple[float, float]:
        v = self.m @ np.array([p[0], p[1], p[2], 1.0])
        x, y = v[0] / v[3], v[1] / v[3]
        return ((x * 0.5 + 0.5) * self.w * SS, (0.5 - y * 0.5) * self.h * SS)


def corners(obb) -> np.ndarray:
    cx, cy, cz, hx, hy, hz, yaw = obb
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for z in (cz - hz, cz + hz):
        for ax, ay in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = ax * hx, ay * hy
            out.append((cx + c * x - s * y, cy + s * x + c * y, z))
    return np.array(out)


def eye(fr: dict) -> np.ndarray:
    return np.array([fr["x"], fr["y"], S.EYE_H - SIT_DROP_M * fr["sit"]])


def in_view(fr: dict, pts: np.ndarray) -> bool:
    """Whether any of the points falls inside the glasses camera's view from this frame."""
    e = eye(fr)
    d = pts - e
    ground = np.hypot(d[:, 0], d[:, 1])
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) - math.degrees(fr["heading"])
    az = (az + 180) % 360 - 180
    el = np.degrees(np.arctan2(d[:, 2], ground)) + fr["pitch_deg"]
    ok = (np.abs(az) <= S.HFOV / 2) & (np.abs(el) <= VFOV / 2) & (ground <= SEE_M)
    return bool(ok.any())


def sample(obb) -> np.ndarray:
    c = corners(obb)
    return np.vstack([c, c.mean(axis=0), (c[:4] + c[4:]) / 2])


def gap_to(o: dict, x: float, y: float) -> float:
    """Floor distance from a point to an object's footprint (0 inside it)."""
    cx, cy, _, hx, hy, _, yaw = o["obb"]
    c, s = math.cos(yaw), math.sin(yaw)
    u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
    return math.hypot(max(abs(u) - hx, 0.0), max(abs(v) - hy, 0.0))


def tucked(doc: dict) -> set[str]:
    """Low clutter whose middle sits inside a desk's or bed's footprint: under the furniture, so
    a label or box on it would land on the desk instead."""
    hosts = [o for o in doc["objects"] if o["label"] in ("desk", "bed")]
    out = set()
    for o in doc["objects"]:
        if o["label"] in ("desk", "bed", "chair"):
            continue
        if any(gap_to(p, o["obb"][0], o["obb"][1]) == 0 for p in hosts):
            out.add(o["name"])
    return out


def heat(k: float) -> tuple[int, int, int]:
    """The belt's heat colour (web/src/beltlayout.ts heat)."""
    k = min(max(k, 0.0), 1.0)
    stops = [(0.0, (255, 226, 110)), (0.5, (255, 150, 40)), (1.0, (255, 48, 36))]
    (k0, c0), (k1, c1) = stops[:2] if k < 0.5 else stops[1:]
    f = (k - k0) / (k1 - k0)
    return tuple(round(a + (b - a) * f) for a, b in zip(c0, c1, strict=True))


def felt8(fr: dict) -> list[float]:
    """Felt levels per belt cell, clockwise from ahead; the 4-motor belt sits on the diagonals."""
    f = fr["felt"]
    return [0.0, f[0], 0.0, f[1], 0.0, f[2], 0.0, f[3]] if len(f) == 4 else list(f)


def direction_ring(levels: list[float], four: bool, size: int) -> Image.Image:
    """The belt from above, front up: eight cells round a head, each lit in its heat colour as
    hard as it buzzes, so a viewer reads which side is talking."""
    k = 4
    img = Image.new("RGBA", (size * k, size * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = size * k / 2
    r0, r1 = 0.24 * size * k, 0.37 * size * k
    for i, lv in enumerate(levels):
        if four and i % 2 == 0:
            continue
        a0, a1 = -90 + 45 * i - 19, -90 + 45 * i + 19  # PIL angles: clockwise from +x
        if lv > 0.05:
            col = (*heat(lv), int(110 + 145 * min(lv, 1.0)))
        else:
            col = (206, 210, 216, 255)
        bb = (c - r1, c - r1, c + r1, c + r1)
        d.pieslice(bb, a0, a1, fill=col)
        d.ellipse((c - r0, c - r0, c + r0, c + r0), fill=(0, 0, 0, 0))
    d.ellipse(
        (c - r0 + 2 * k, c - r0 + 2 * k, c + r0 - 2 * k, c + r0 - 2 * k), fill=(250, 250, 252, 255)
    )
    hr = 0.1 * size * k  # the head, nose up
    d.ellipse((c - hr, c - hr, c + hr, c + hr), fill=(60, 64, 72, 255))
    d.polygon(
        [(c, c - hr - 7 * k), (c - 5 * k, c - hr + 2 * k), (c + 5 * k, c - hr + 2 * k)],
        fill=(60, 64, 72, 255),
    )
    img = img.resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 11)
    for i, lab in enumerate(DIRS):
        if four and i % 2 == 0:
            continue
        a = math.radians(-90 + 45 * i)
        rr = 0.445 * size
        x, y = size / 2 + rr * math.cos(a), size / 2 + rr * math.sin(a)
        on = levels[i] > 0.05
        tw = d.textlength(lab, font=f)
        d.text((x - tw / 2, y - 7), lab, font=f, fill=(30, 32, 36) if on else (130, 134, 142))
    return img


def view_states(doc: dict) -> list[dict[str, str]]:
    """Per frame, each object's state: "view", "memory" (seen before, now out of view) or absent."""
    seen: set[str] = set()
    out = []
    pts = {o["name"]: sample(o["obb"]) for o in doc["objects"]}
    for fr in doc["frames"]:
        now = {n for n, p in pts.items() if in_view(fr, p)}
        seen |= now
        out.append({n: "view" if n in now else "memory" for n in seen})
    return out


def ray_room(x, y, a, bounds) -> float:
    """How far a floor ray from (x, y) at angle a runs before it meets the room's walls."""
    x0, x1, y0, y1 = bounds
    c, s = math.cos(a), math.sin(a)
    ts = [SEE_M]
    if c > 1e-6:
        ts.append((x1 - x) / c)
    if c < -1e-6:
        ts.append((x0 - x) / c)
    if s > 1e-6:
        ts.append((y1 - y) / s)
    if s < -1e-6:
        ts.append((y0 - y) / s)
    return max(0.0, min(ts))


def visible_floor(fr: dict) -> tuple[float, float]:
    """Near and far ground distance of the floor inside the camera's vertical view."""
    h = S.EYE_H - SIT_DROP_M * fr["sit"]
    p = fr["pitch_deg"]
    lo, hi = p - VFOV / 2, p + VFOV / 2  # degrees below the horizon at the top and bottom edge
    near = h / math.tan(math.radians(hi)) if hi > 0 else SEE_M
    far = h / math.tan(math.radians(lo)) if lo > 0.5 else SEE_M
    return min(near, SEE_M), min(far, SEE_M)


def wedge(fr: dict, bounds, near: float, far: float) -> list[tuple[float, float, float]]:
    """The floor inside the horizontal view between two ground distances, cut at the walls."""
    poly = []
    for r, sweep in ((near, 1), (far, -1)):
        for k in range(25):
            a = fr["heading"] + sweep * math.radians(S.HFOV / 2) * (1 - k / 12)
            rr = min(r, ray_room(fr["x"], fr["y"], a, bounds))
            poly.append((fr["x"] + rr * math.cos(a), fr["y"] + rr * math.sin(a), 0.005))
    return poly


def ring_events(pose_path: Path) -> list[tuple[float, str, np.ndarray]]:
    """When each sound plays and the world point it plays from, as the binaural render does."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "render_sim_sound", Path(__file__).with_name("render_sim_sound.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t, pos, fwd, up, target, hazards, _ = mod.sidecar(pose_path)
    out = []
    for e in ss.schedule(t, pos, fwd, up, target, hazards):
        k = int(np.clip(np.searchsorted(t, e.t, side="right") - 1, 0, len(t) - 1))
        if e.kind.startswith("beacon") or e.kind == "arrive":
            src = np.asarray(target, float)
        else:
            near = [hz.nearest(pos[k]) for hz in hazards]
            src = min(near, key=lambda q: math.hypot(q[0] - pos[k][0], q[1] - pos[k][1]))
        out.append((float(e.t - t[0]), e.kind, np.asarray(src, float)))
    return out


def font(size: int, mono: bool = False):
    return ImageFont.truetype(MONO if mono else FONT, size * SS)


def bbox_hit(a, b, pad=6 * SS) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def place_labels(draw, items, blocked, w, h, away, hull, memo=None):
    """Greedy placement: nearest objects first. Each label tries offsets pointing away from the
    room's middle, first outside the room (a callout in the margin, so it covers no furniture),
    then anywhere, and takes the first that covers no other label, anchor or panel."""
    f = font(15)
    fs = font(12, mono=True)
    ring = [(math.cos(q), math.sin(q)) for q in np.radians(np.arange(0, 360, 30))]
    anchors = [
        (it[2][0] - 6 * SS, it[2][1] - 6 * SS, it[2][0] + 6 * SS, it[2][1] + 6 * SS) for it in items
    ]
    placed = []
    for n, (name, dims, anchor, col) in enumerate(items):
        tw = max(draw.textlength(name, font=f), draw.textlength(dims, font=fs))
        bw, bh = tw + 20 * SS, 42 * SS
        vx, vy = anchor[0] - away[0], anchor[1] - away[1]
        order = sorted(ring, key=lambda u: -(u[0] * vx + u[1] * vy))
        radii = (110, 170, 240, 330, 430, 540)
        offsets = [(u[0] * r, u[1] * r * 0.7) for r in radii for u in order]
        others = anchors[:n] + anchors[n + 1 :]
        tries = [(ox, oy, True) for ox, oy in offsets] + [(ox, oy, False) for ox, oy in offsets]
        if memo is not None and name + dims in memo:  # keep last frame's spot while it still fits
            tries.insert(0, (*memo[name + dims], False))
        for ox, oy, outside in tries:
            cx, cy = anchor[0] + ox * SS, anchor[1] + oy * SS
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            inside = box[0] > 10 * SS and box[1] > 10 * SS and box[2] < w - 10 * SS
            inside = inside and box[3] < h - 10 * SS
            if outside and inside:
                pts = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
                inside = all(cv2.pointPolygonTest(hull, p, False) < 0 for p in pts)
            if inside and not any(bbox_hit(box, b) for b in blocked + placed + others):
                placed.append(box)
                if memo is not None:
                    memo[name + dims] = (ox, oy)
                yield name, dims, anchor, col, box
                break


def _cross(p1, p2, q1, q2) -> bool:
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    return side(p1, p2, q1) * side(p1, p2, q2) < 0 and side(q1, q2, p1) * side(q1, q2, p2) < 0


def untangle(labels: list) -> list:
    """Swap the boxes of two labels whose leader lines cross, when the boxes are the same size."""

    def mid(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    for _ in range(3):
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                if not _cross(a[2], mid(a[4]), b[2], mid(b[4])):
                    continue
                ca, cb = mid(a[4]), mid(b[4])
                ha, hb = (a[4][2] - a[4][0]) / 2, (b[4][2] - b[4][0]) / 2
                hh = (a[4][3] - a[4][1]) / 2
                labels[i] = (*a[:4], (cb[0] - ha, cb[1] - hh, cb[0] + ha, cb[1] + hh))
                labels[j] = (*b[:4], (ca[0] - hb, ca[1] - hh, ca[0] + hb, ca[1] + hh))
    return labels


def draw_frame(doc, k, base: Image.Image, cam: Cam, states, rings, belt, memo=None):
    fr = doc["frames"][k]
    hidden = tucked(doc)
    w, h = base.width * SS, base.height * SS
    walls = np.array(doc["walls"])
    room = (
        float(np.min(walls[:, 0] + walls[:, 3])),
        float(np.max(walls[:, 0] - walls[:, 3])),
        float(np.min(walls[:, 1] + walls[:, 4])),
        float(np.max(walls[:, 1] - walls[:, 4])),
    )
    inner = (room[0] + 0.24, room[1] - 0.24, room[2] + 0.24, room[3] - 0.24)
    room = (
        min(r for r in inner[:2]),
        max(r for r in inner[:2]),
        min(r for r in inner[2:]),
        max(r for r in inner[2:]),
    )

    # the view cone, soft-edged, under everything else
    cone = Image.new("L", (w, h), 0)
    cd = ImageDraw.Draw(cone)
    cd.polygon([cam.px(p) for p in wedge(fr, room, 0.25, SEE_M)], fill=60)
    near, far = visible_floor(fr)
    if near < far:  # the floor the camera actually sees, brighter
        cd.polygon([cam.px(p) for p in wedge(fr, room, near, far)], fill=130)
    cone = cone.filter(ImageFilter.GaussianBlur(8 * SS))
    layer = Image.new("RGBA", (w, h), (*IN_VIEW, 0))
    layer.putalpha(cone)
    out = base.convert("RGBA").resize((w, h), Image.LANCZOS)
    out = Image.alpha_composite(out, layer)

    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # sound rings: each sound grows a ring from its source and fades out
    t = fr["t"]
    for te, kind, src in rings:
        age = t - te
        if not 0 <= age < RING_S:
            continue
        a = age / RING_S
        for lag in (0.0, 0.22):
            b = a - lag
            if b <= 0:
                continue
            r = 0.12 + 0.75 * b
            pts = [
                cam.px((src[0] + r * math.cos(q), src[1] + r * math.sin(q), max(src[2], 0.02)))
                for q in np.linspace(0, 2 * math.pi, 64)
            ]
            alpha = int(200 * (1 - b) ** 1.5)
            col = (255, 214, 120) if kind in ("slow", "fast", "stop") else RING
            d.line(pts + [pts[0]], fill=(*col, alpha), width=3 * SS)

    # the walked path so far, as a faint dotted trail
    for p in doc["frames"][: k + 1 : 3]:
        x, y = cam.px((p["x"], p["y"], 0.01))
        d.ellipse((x - 2 * SS, y - 2 * SS, x + 2 * SS, y + 2 * SS), fill=(255, 255, 255, 120))

    # the scenario's target glows softly on the floor, so the hero object reads first
    objs = {o["name"]: o for o in doc["objects"]}
    goal = objs.get(doc.get("target") or doc.get("hazard") or "")
    if goal is not None:
        gx, gy = goal["obb"][0], goal["obb"][1]
        rad = max(goal["obb"][3:5]) + 0.28
        glow = Image.new("L", (w, h), 0)
        gpts = [
            cam.px((gx + rad * math.cos(q), gy + rad * math.sin(q), 0.01))
            for q in np.linspace(0, 2 * math.pi, 48)
        ]
        ImageDraw.Draw(glow).polygon(gpts, fill=150)
        glow = glow.filter(ImageFilter.GaussianBlur(22 * SS))
        gl = Image.new("RGBA", (w, h), (*HERO, 0))
        gl.putalpha(glow)
        out = Image.alpha_composite(out, gl)

    # oriented boxes: what is in view, and what is remembered close by; the rest fade away
    st = states[k]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
    edges += [(i, i + 4) for i in range(4)]
    for name, state in st.items():
        o = objs[name]
        if name in hidden:
            continue
        if state == "view":
            col, alpha = IN_VIEW, 235
        else:
            fade = 1 - (gap_to(o, fr["x"], fr["y"]) - OUTLINE_M) / FADE_M
            col, alpha = REMEMBERED, int(200 * min(1.0, fade))
            if alpha <= 0:
                continue
        c = [cam.px(p) for p in corners(o["obb"])]
        for i, j in edges:
            d.line([c[i], c[j]], fill=(*col, alpha), width=int(1.6 * SS))

    # labels: anything in view, or within reach of the walker
    me = eye(fr)
    items = []
    for o in sorted(doc["objects"], key=lambda o: math.dist(o["obb"][:2], me[:2])):
        state = st.get(o["name"])
        if o["name"] in hidden:
            continue
        gap = math.dist(o["obb"][:2], me[:2]) - max(o["obb"][3:5])
        big = max(o["obb"][3:6]) >= 0.1
        if not state or not (gap <= LABEL_M or (state == "view" and big and gap <= VIEW_LABEL_M)):
            continue
        if len(items) == MAX_LABELS:
            break
        cx, cy, cz, hx, hy, hz, _ = o["obb"]
        dims = f"{2 * hx:.2f} x {2 * hy:.2f} x {2 * hz:.2f} m"
        col = IN_VIEW if state == "view" else REMEMBERED
        items.append((NAMES.get(o["name"], o["label"]), dims, cam.px((cx, cy, cz + hz)), col))
    wx, wy = cam.px((fr["x"], fr["y"], 0.9))
    blocked = [(wx - 60 * SS, wy - 110 * SS, wx + 60 * SS, wy + 110 * SS)]
    blocked += [(w - 900 * SS, 0, w, 300 * SS), (0, 0, 700 * SS, 150 * SS)]
    blocked += [(0, h - 120 * SS, 620 * SS, h)]
    f, fs = font(15), font(12, mono=True)
    ox0, ox1 = float(np.min(walls[:, 0] - walls[:, 3])), float(np.max(walls[:, 0] + walls[:, 3]))
    oy0, oy1 = float(np.min(walls[:, 1] - walls[:, 4])), float(np.max(walls[:, 1] + walls[:, 4]))
    shell = [cam.px((x, y, z)) for x in (ox0, ox1) for y in (oy0, oy1) for z in (0.0, 2.6)]
    hull = cv2.convexHull(np.array(shell, np.float32))
    mid = cam.px(((ox0 + ox1) / 2, (oy0 + oy1) / 2, 0.5))
    labels = untangle(list(place_labels(d, items, blocked, w, h, mid, hull, memo)))
    for name, dims, anchor, col, box in labels:
        bx = min(max(anchor[0], box[0]), box[2])
        by = box[3] if anchor[1] > box[3] else box[1] if anchor[1] < box[1] else anchor[1]
        d.line([anchor, (bx, by)], fill=(*col, 220), width=int(1.4 * SS))
        d.ellipse(
            (anchor[0] - 3 * SS, anchor[1] - 3 * SS, anchor[0] + 3 * SS, anchor[1] + 3 * SS),
            fill=(*col, 255),
        )
        d.rounded_rectangle(box, radius=7 * SS, fill=(18, 20, 24, 200))
        d.rectangle((box[0], box[1] + 8 * SS, box[0] + 3 * SS, box[3] - 8 * SS), fill=(*col, 255))
        d.text((box[0] + 11 * SS, box[1] + 5 * SS), name, font=f, fill=(255, 255, 255, 255))
        d.text((box[0] + 11 * SS, box[1] + 23 * SS), dims, font=fs, fill=(200, 205, 212, 255))

    # title, numbers, legend, corner label
    m = doc["metrics"]
    layout = "8 motors" if doc["layout"] == "eight" else "4 motors (diagonals)"
    d.rounded_rectangle(
        (24 * SS, 24 * SS, 660 * SS, 128 * SS), radius=12 * SS, fill=(18, 20, 24, 190)
    )
    d.text((44 * SS, 38 * SS), doc["title"], font=font(24), fill=(255, 255, 255, 255))
    d.text(
        (44 * SS, 74 * SS),
        f"blindfolded walker  ·  {layout}  ·  seed {doc['seed']}",
        font=font(14),
        fill=(200, 205, 212, 255),
    )
    clock = f"t {t:5.2f} s"
    if doc.get("arm"):
        clock += f"   {doc['arm']}: {doc['arm_note']}"
    d.text((44 * SS, 98 * SS), clock, font=font(13, mono=True), fill=(170, 176, 186, 255))
    # the verdict holds through the end hold (walk over, any sit-down finished)
    done = k == len(doc["frames"]) - 1 or (not fr["walking"] and fr["sit"] in (0, 1))
    status = "sitting" if fr["sit"] > 0 else "walking"
    if done:
        status = "reached" if m["success"] else "not reached"
        if not m["success"] and fr["sit"] > 0:  # got to the chair, but touched something first
            status = f"sat after {m['collisions']} contacts, not counted"
        if doc.get("cut_s"):  # the clip stops early; the walk ran on to the time limit
            status = f"not reached in {m['time_s']:.0f} s"
    # live while walking (elapsed time, running minimum clearance), the result once it is over
    walked = min(t, m["time_s"])
    clear = fr.get("min_clear", m["min_clear_m"])
    if done:
        clock_s = f"{m['time_s']:.1f} s" if m["success"] else f"{m['time_s']:.0f} s, no goal"
        if not m["success"] and fr["sit"] > 0:  # at the seat, but the contacts void it
            clock_s = f"{m['time_s']:.1f} s, voided"
    else:
        clock_s = f"{walked:.1f} s"
    rows = [
        ("collisions", f"{fr['collisions']}"),
        ("min clearance", f"{m['min_clear_m'] if done else clear:.2f} m"),
        ("time to goal", clock_s),
        ("false buzzes", f"{m['false_per_min']:.1f} /min" if done else f"{fr.get('false', 0)}"),
        ("status", status),
    ]
    y0 = h - 112 * SS
    right = max(600 * SS, 44 * SS + 4 * 110 * SS + d.textlength(status, font=font(17)) + 20 * SS)
    d.rounded_rectangle(
        (24 * SS, y0 - 14 * SS, right, h - 24 * SS), radius=12 * SS, fill=(18, 20, 24, 190)
    )
    for i, (key, val) in enumerate(rows):
        x = 44 * SS + i * 110 * SS
        d.text((x, y0), key, font=font(11), fill=(170, 176, 186, 255))
        d.text((x, y0 + 18 * SS), val, font=font(17), fill=(255, 255, 255, 255))
    legend = [(IN_VIEW, "in view"), (REMEMBERED, "remembered"), ((255, 214, 120), "sound")]
    lx = 44 * SS
    for col, text in legend:
        d.rounded_rectangle(
            (lx, y0 + 50 * SS, lx + 14 * SS, y0 + 58 * SS), radius=3 * SS, fill=(*col, 255)
        )
        d.text((lx + 20 * SS, y0 + 46 * SS), text, font=font(11), fill=(200, 205, 212, 255))
        lx += 120 * SS
    lf = font(10)
    tw = d.textlength(LABEL, font=lf)
    d.text((w - tw - 22 * SS, h - 26 * SS), LABEL, font=lf, fill=(235, 238, 242, 170))

    out = Image.alpha_composite(out, ov).resize(base.size, Image.LANCZOS)

    if belt is not None:  # the 3D belt in a card, top right, with the cells from above
        card = belt_card(belt, felt8(fr), doc["layout"] != "eight")
        out.alpha_composite(card, (out.width - card.width - 24, 24))
    return out.convert("RGB")


def belt_card(belt: Image.Image, levels: list[float], four: bool) -> Image.Image:
    """The belt panel: the 3D belt screenshot beside the cells seen from above, on a card."""
    bw = 600
    bh = int(belt.height * bw / belt.width)
    rs = min(max(bh, 150), 190)
    card = Image.new("RGBA", (rs + bw + 24, max(bh, rs) + 44), (245, 245, 247, 236))
    card.paste(belt.convert("RGB").resize((bw, bh), Image.LANCZOS), (rs + 16, 36))
    ring = direction_ring(levels, four, rs)
    card.alpha_composite(ring, (8, 36 + (max(bh, rs) - rs) // 2))
    cd = ImageDraw.Draw(card)
    cd.text(
        (16, 10),
        "what the walker feels  ·  from above, front up  ·  belt from behind",
        font=ImageFont.truetype(FONT, 14),
        fill=(60, 64, 72),
    )
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *card.size), radius=14, fill=255)
    card.putalpha(Image.fromarray(np.minimum(np.asarray(card.getchannel("A")), np.asarray(mask))))
    return card


def crop_belt(img: Image.Image) -> Image.Image:
    """Trim the panel screenshot to the belt, with a little margin."""
    a = np.asarray(img.convert("RGB")).astype(int)
    bg = a[2, 2]
    diff = np.abs(a - bg).sum(axis=2) > 24
    ys, xs = np.nonzero(diff)
    pad = 30
    return img.crop(
        (
            max(xs.min() - pad, 0),
            max(ys.min() - pad, 0),
            min(xs.max() + pad, img.width),
            min(ys.max() + pad, img.height),
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene", type=Path)
    ap.add_argument("render", type=Path, help="a still PNG, or the folder of frames")
    ap.add_argument("--frame", type=int, help="the scene frame of a still")
    ap.add_argument("--belt", type=Path, help="belt panel still")
    ap.add_argument("--belt-dir", type=Path, help="belt panel frames, one per video frame")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    doc = json.loads(a.scene.read_text())
    states = view_states(doc)
    rings = ring_events(a.scene.with_suffix(".pose.jsonl"))
    if a.render.is_file():
        cam = Cam(a.render.with_suffix(".cam.json"))
        belt = crop_belt(Image.open(a.belt)) if a.belt else None
        img = draw_frame(doc, a.frame, Image.open(a.render), cam, states, rings, belt)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        img.save(a.out)
        print(a.out)
        return
    cam = Cam(a.render / "cam.json")
    a.out.mkdir(parents=True, exist_ok=True)
    frames = sorted(a.render.glob("*.png"))
    memo: dict = {}
    for k, p in enumerate(frames):
        belt = None
        if a.belt_dir:
            bp = a.belt_dir / f"{k + 1:05d}.png"
            belt = crop_belt(Image.open(bp)) if bp.exists() else None
        draw_frame(doc, k, Image.open(p), cam, states, rings, belt, memo).save(a.out / p.name)
    print(a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
