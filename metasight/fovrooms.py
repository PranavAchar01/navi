"""Reconstructed rooms as field-of-view scenes (map-recon's `objects.json`).

SIMULATED USER MODEL, NOT A HUMAN STUDY. `scene_boxes` turns the oriented boxes of a
reconstruction into the axis-aligned `fovsim.Box` list the MuJoCo scenes use, and
`bedroom_scenes` scripts head paths through the real bedroom that hit the catalogue's modes.

Mesh frame (y up, yaw about +y, box local x axis (cos yaw, 0, sin yaw)) to scene frame: X = x,
Y = -z (keeps handedness), Z = y - floor_y. Each box in the file is the object's bounding box;
`floor_clearance_m` is the height of its underside above the box bottom, so an object with
clearance is a slab from the clearance to the top, held up by thin legs (a small footprint gets
one post). Oriented footprints are cut into strips of axis-aligned boxes. Objects the file marks
`hazard: false` (a laptop or pillow lying on a bed or desk) are left out, and so are door leaves
and whiteboards, which sit flush in a wall. The file carries no walls: they are put at the
extent of the hazards plus 0.35 m, except on a side where a flush door or whiteboard gives the
wall plane. Doors named in `open_doors` leave a gap in their wall.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .fovsim import Box, Scene

ROOT = Path(__file__).resolve().parent.parent
# A worktree has no data/: fall back to the main checkout.
if not (ROOT / "data").exists() and ROOT.parent.name == "worktrees":
    ROOT = ROOT.parents[2]
BEDROOM_OBJECTS = ROOT / "data" / "rooms" / "bedroom" / "recon-v2" / "v2.0" / "objects.json"
FLUSH = ("door", "whiteboard")
STRIP_M = 0.05
LEG_HALF = 0.02
WALL_HALF_T = 0.05
MIN_TOP_M = 0.03


def _strips(cx: float, cy: float, sx: float, sy: float, yaw: float, z0: float, z1: float):
    """Boxes covering an oriented footprint (half sizes sx, sy along its local x and y, turned
    `yaw` counter-clockwise) between heights z0 and z1."""
    c, s = math.cos(yaw), math.sin(yaw)
    hz, cz = (z1 - z0) / 2, (z0 + z1) / 2
    if abs(s * c) < 0.02:  # a quarter turn or none: one box
        hx, hy = (sx, sy) if abs(c) > abs(s) else (sy, sx)
        return [(cx, cy, cz, hx, hy, hz)]
    long_x = sx >= sy
    n = min(60, max(1, math.ceil(2 * (sx if long_x else sy) / STRIP_M)))
    out: list[Box] = []
    for i in range(n):
        u0, u1 = (-1 + 2 * i / n), (-1 + 2 * (i + 1) / n)
        um = (u0 + u1) / 2
        ex, ey = (um * sx, 0.0) if long_x else (0.0, um * sy)
        hx_l, hy_l = (sx / n, sy) if long_x else (sx, sy / n)
        mx, my = cx + c * ex - s * ey, cy + s * ex + c * ey
        hx = abs(c) * hx_l + abs(s) * hy_l
        hy = abs(s) * hx_l + abs(c) * hy_l
        out.append((mx, my, cz, hx, hy, hz))
    return out


def _object_boxes(o: dict, floor_y: float) -> list[Box]:
    b = o["obb"]
    (px, py, pz), (sx, sh, sz) = b["center"], b["size"]
    yaw = -float(b["yaw_rad"])  # mesh z runs the other way (Y = -z)
    bottom = max(0.0, py - sh / 2 - floor_y)
    top = py + sh / 2 - floor_y
    if top < MIN_TOP_M:
        return []
    hx, hy = max(sx / 2, 0.01), max(sz / 2, 0.01)
    cx, cy = px, -pz
    lo = min(top - 0.02, bottom + max(0.0, float(o.get("floor_clearance_m", 0.0))))
    out = _strips(cx, cy, hx, hy, yaw, lo, top)
    if lo > bottom + 0.04:
        c, s = math.cos(yaw), math.sin(yaw)
        if min(hx, hy) < 0.15:
            posts = [(0.0, 0.0)]
        else:
            ix, iy = hx - 0.04, hy - 0.04
            posts = [(a * ix, d * iy) for a in (-1, 1) for d in (-1, 1)]
        for u, v in posts:
            out.append(
                (
                    cx + c * u - s * v,
                    cy + s * u + c * v,
                    (bottom + lo) / 2,
                    LEG_HALF,
                    LEG_HALF,
                    (lo - bottom) / 2,
                )
            )
    return out


def _walls(objs: list[dict], floor_y: float, ceil: float, open_doors: tuple[str, ...]):
    """Four wall boxes. Sides with a flush door or whiteboard sit at its plane."""
    ext = {"x0": 1e9, "x1": -1e9, "y0": 1e9, "y1": -1e9}
    flush = []
    for o in objs:
        b = o["obb"]
        (px, _, pz), (sx, _, sz) = b["center"], b["size"]
        if o["class"] in FLUSH and min(sx, sz) < 0.15:
            flush.append(o)
            continue
        if not o.get("hazard", True) or o.get("floor_clearance_m", 0) < 0 or sz <= 0:
            continue
        r = math.hypot(sx, sz) / 2 if abs(math.sin(2 * b["yaw_rad"])) > 0.02 else None
        hx, hy = (r, r) if r else (sx / 2, sz / 2)
        ext["x0"], ext["x1"] = min(ext["x0"], px - hx), max(ext["x1"], px + hx)
        ext["y0"], ext["y1"] = min(ext["y0"], -pz - hy), max(ext["y1"], -pz + hy)
    pad = 0.35
    side = {
        "x0": ext["x0"] - pad,
        "x1": ext["x1"] + pad,
        "y0": ext["y0"] - pad,
        "y1": ext["y1"] + pad,
    }
    seen: set[str] = set()
    gaps: dict[str, list[tuple[float, float]]] = {"x0": [], "x1": [], "y0": [], "y1": []}
    for o in flush:
        (px, _, pz), (sx, _, sz) = o["obb"]["center"], o["obb"]["size"]
        if sz < sx:  # thin in z: a wall of constant Y
            k, pos, lo, hi = ("y1" if -pz > 0 else "y0"), -pz, px - sx / 2, px + sx / 2
        else:
            k, pos, lo, hi = ("x1" if px > 0 else "x0"), px, -pz - sz / 2, -pz + sz / 2
        if k in seen:  # the innermost plane wins
            pos = min(pos, side[k]) if k.endswith("1") else max(pos, side[k])
        seen.add(k)
        side[k] = pos
        if o["id"] in open_doors:
            gaps[k].append((lo, hi))
    h = ceil - floor_y if ceil > 1 else 2.7
    h = min(h, 2.8)
    out: list[Box] = []
    x0, x1, y0, y1 = side["x0"], side["x1"], side["y0"], side["y1"]
    for k, fixed, a, b in (
        ("y0", y0, x0, x1),
        ("y1", y1, x0, x1),
        ("x0", x0, y0, y1),
        ("x1", x1, y0, y1),
    ):
        cuts = sorted(gaps[k])
        spans, cur = [], a
        for lo, hi in cuts:
            spans.append((cur, lo))
            cur = hi
        spans.append((cur, b))
        sign = -1 if k in ("y0", "x0") else 1  # the wall's body lies outside the face
        for p, q in spans:
            if q - p < 0.02:
                continue
            m = (p + q) / 2
            if k[0] == "y":
                out.append((m, fixed + sign * WALL_HALF_T, h / 2, (q - p) / 2, WALL_HALF_T, h / 2))
            else:
                out.append((fixed + sign * WALL_HALF_T, m, h / 2, WALL_HALF_T, (q - p) / 2, h / 2))
    return out


def scene_boxes(
    data: dict, open_doors: tuple[str, ...] = (), walls: bool = True
) -> tuple[list[Box], dict[str, list[Box]]]:
    """(all boxes, boxes by object id) for a parsed objects.json."""
    floor_y = float(data["floor_y"])
    by_id: dict[str, list[Box]] = {}
    for o in data["objects"]:
        if o["class"] in FLUSH or not o.get("hazard", True):
            continue
        bx = _object_boxes(o, floor_y)
        if bx:
            by_id[o["id"]] = bx
    boxes = [b for v in by_id.values() for b in v]
    if walls:
        boxes += _walls(
            data["objects"], floor_y, float(data.get("ceiling_height_m", 2.7)), open_doors
        )
    return boxes, by_id


def load_objects(path: Path | str = BEDROOM_OBJECTS) -> dict:
    return json.loads(Path(path).read_text())


def bedroom_scenes(path: Path | str | dict = BEDROOM_OBJECTS) -> dict[str, Scene]:
    """Five scripted paths through the reconstructed bedroom (scene frame: X = mesh x, Y = -z).
    Keys are (t, x, y, body heading deg, head yaw deg (+ left), pitch deg below level)."""
    data = path if isinstance(path, dict) else load_objects(path)
    boxes, _ = scene_boxes(data, open_doors=("door_03",))
    scenes = [
        Scene(
            "bedroom_desk_lookup",
            "Bedroom: walk west down the gap between the bed and the desk looking down at the "
            "desk edge and the chair at its end, then look up and step in.",
            boxes,
            [
                (0, 0.8, 0.35, 180, 0, 45),
                (1.5, 0.55, 0.35, 180, 0, 45),
                (2.3, 0.4, 0.35, 180, 0, 8),
                (5.0, 0.2, 0.35, 180, 0, 5),
            ],
        ),
        Scene(
            "bedroom_turn_back",
            "Bedroom: stand at the desk, turn the head 90 degrees away toward the south end of "
            "the room, then back toward the desk and step in.",
            boxes,
            [
                (0, 0.45, 0.2, 180, 0, 35),
                (1.0, 0.45, 0.2, 180, 0, 35),
                (1.6, 0.45, 0.2, 180, 90, 15),
                (3.0, 0.45, 0.2, 180, 90, 15),
                (3.6, 0.45, 0.2, 180, 30, 15),
                (5.0, 0.2, 0.2, 180, 30, 15),
            ],
        ),
        Scene(
            "bedroom_bed_corner",
            "Bedroom: walk north along the bed's west side with the head turned 60 degrees "
            "toward the bed, past its corner and on toward the far desk.",
            boxes,
            [
                (0, 0.72, -0.6, 90, 0, 12),
                (1.0, 0.72, -0.2, 90, -60, 12),
                (4.0, 0.72, 1.3, 90, -60, 12),
                (5.5, 0.72, 1.55, 90, 0, 12),
            ],
        ),
        Scene(
            "bedroom_doorway",
            "Bedroom: walk to the back doorway past the floor clutter and the foot of the bed, "
            "looking 40 degrees to the right, then ahead at the gap.",
            boxes,
            [
                (0, 2.1, 1.3, 90, 0, 15),
                (1.5, 2.1, 1.7, 90, -40, 15),
                (3.0, 2.15, 2.1, 90, -40, 15),
                (4.5, 2.2, 2.5, 90, 0, 15),
            ],
        ),
        Scene(
            "bedroom_low_table",
            "Bedroom: walk north toward the low desk against the back wall with the head "
            "level; its slab drops under the frame.",
            boxes,
            [(0, 0.2, 0.7, 90, 0, 15), (3.5, 0.2, 1.55, 90, 0, 15), (5.0, 0.2, 1.65, 90, 0, 12)],
        ),
    ]
    return {s.name: s for s in scenes}
