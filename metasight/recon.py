"""The room as objects: geometry helpers for scripts/recon_v2.py, kept free of models and meshes so
they are testable.

Frame: the room mesh's (y up, metres as the reconstruction gives them, floor near y = 0). Every
object is an upright box (yaw only), because furniture stands on the floor: a desk, a bed or a
shelf is a rectangle in plan with a height. Edges are the box's horizontal outlines at the
heights a walker can hit them: the top, and the underside when there is room beneath (a desk or
a shelf overhang).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Class priors: (material, typical mass kg, typical box volume m3, movable). Mass is scaled by the
# measured box volume against the typical one, clipped to half and twice the typical, so a big
# desk weighs more than a small one but a bad box cannot make a keyboard weigh 40 kg.
PRIORS: dict[str, tuple[str, float, float, bool]] = {
    "desk": ("particleboard and steel", 30.0, 1.2 * 0.6 * 0.75, False),
    "office chair": ("plastic, steel and fabric", 14.0, 0.65 * 0.65 * 1.1, True),
    "chair": ("wood or plastic", 6.0, 0.5 * 0.5 * 0.9, True),
    "computer monitor": ("plastic and glass", 5.0, 0.7 * 0.2 * 0.5, True),
    "bed": ("wood frame and foam mattress", 60.0, 2.0 * 1.4 * 0.6, False),
    "shelf": ("particleboard", 20.0, 0.8 * 0.35 * 1.8, False),
    "keyboard": ("plastic", 0.8, 0.45 * 0.15 * 0.04, True),
    "laptop": ("aluminium", 1.6, 0.31 * 0.22 * 0.02, True),
    "door": ("hollow-core wood", 12.0, 0.8 * 0.04 * 2.0, False),
    "trash can": ("plastic", 1.5, 0.3 * 0.3 * 0.45, True),
    "laundry basket": ("plastic", 1.5, 0.45 * 0.35 * 0.6, True),
    "whiteboard": ("melamine on fibreboard", 4.0, 0.9 * 0.02 * 0.6, False),
    "backpack": ("fabric", 3.0, 0.3 * 0.2 * 0.45, True),
    "bag": ("fabric", 1.5, 0.4 * 0.2 * 0.4, True),
    "lamp": ("metal and plastic", 2.0, 0.3 * 0.3 * 0.6, True),
    "dresser": ("wood", 45.0, 1.0 * 0.5 * 0.9, False),
    "cabinet": ("particleboard", 30.0, 0.8 * 0.5 * 1.0, False),
    "nightstand": ("particleboard", 10.0, 0.45 * 0.4 * 0.55, False),
    "box": ("cardboard", 2.0, 0.4 * 0.3 * 0.3, True),
    "robot arm": ("aluminium and servos", 1.5, 0.3 * 0.2 * 0.35, True),
    "speaker": ("plastic", 2.0, 0.15 * 0.15 * 0.25, True),
    "pillow": ("fabric and foam", 0.8, 0.6 * 0.4 * 0.15, True),
    "plant": ("ceramic pot and soil", 6.0, 0.4 * 0.4 * 1.0, True),
    "fan": ("plastic", 3.0, 0.35 * 0.3 * 0.9, True),
    "computer": ("aluminium", 1.3, 0.2 * 0.2 * 0.05, True),
    "mini fridge": ("steel and plastic", 18.0, 0.5 * 0.5 * 0.85, False),
    "shoe": ("rubber and fabric", 0.4, 0.3 * 0.12 * 0.12, True),
}
# Things a walker can strike with the body: their edges and overhangs matter for the belt.
HAZARD = {
    "desk",
    "office chair",
    "chair",
    "bed",
    "shelf",
    "door",
    "trash can",
    "laundry basket",
    "dresser",
    "cabinet",
    "nightstand",
    "box",
    "lamp",
    "plant",
    "fan",
    "mini fridge",
    "bag",
    "backpack",
    "shoe",
}


def upright_box(pts: np.ndarray, lo: float = 2.0, hi: float = 98.0, snap: float = 1.0) -> dict:
    """Smallest-area yaw-only box around points (n, 3), y up. Percentiles trim stray points.

    snap above 1 is a square-room prior (the mesh is squared up, so walls run along x and z):
    the wall-aligned box wins unless the best turned one is smaller by more than that factor.
    Furniture pushed against a wall is wall-aligned, and a chair or a mat in its masks would
    otherwise turn its box.

    Returns centre (x, y, z), size (along the box's own x, y, z) and yaw in radians: the box's
    local x axis is (cos yaw, 0, sin yaw) in the room."""
    xz = pts[:, [0, 2]]
    c = np.median(xz, 0)
    best, square = None, None
    for deg in range(0, 90, 2):
        a = math.radians(deg)
        rot = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = (xz - c) @ rot.T
        mn, mx = np.percentile(q, lo, 0), np.percentile(q, hi, 0)
        area = float(np.prod(mx - mn))
        if best is None or area < best[0]:
            best = (area, a, mn, mx, rot)
        if deg == 0:
            square = best
    if square[0] <= snap * best[0]:
        best = square
    _, a, mn, mx, rot = best
    mid = c + ((mn + mx) / 2) @ rot
    y0, y1 = np.percentile(pts[:, 1], lo), np.percentile(pts[:, 1], hi)
    sx, sz = mx - mn
    return {
        "center": [float(mid[0]), float((y0 + y1) / 2), float(mid[1])],
        "size": [float(sx), float(y1 - y0), float(sz)],
        "yaw": float(a),
    }


def box_corners(box: dict, y: float) -> np.ndarray:
    """The box's plan rectangle at height y, as 4 points (x, y, z) going round."""
    cx, _, cz = box["center"]
    sx, _, sz = box["size"]
    a = box["yaw"]
    ux, uz = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    out = []
    for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = np.array([cx, cz]) + i * sx / 2 * ux + j * sz / 2 * uz
        out.append([float(p[0]), float(y), float(p[1])])
    return np.array(out)


def footprint_grid(box: dict, step: float = 0.05, shrink: float = 0.15) -> np.ndarray:
    """Plan points (x, z) inside the box, its outer margin trimmed off, for casting rays."""
    cx, _, cz = box["center"]
    sx, _, sz = box["size"]
    a = box["yaw"]
    ux, uz = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    hx, hz = sx * (1 - shrink) / 2, sz * (1 - shrink) / 2
    ii = np.arange(-hx, hx + 1e-9, step) if hx > step else np.array([0.0])
    jj = np.arange(-hz, hz + 1e-9, step) if hz > step else np.array([0.0])
    g = np.array([[i, j] for i in ii for j in jj])
    return np.array([cx, cz]) + g[:, :1] * ux + g[:, 1:] * uz


def clearance(first_up: np.ndarray, floor_y: float, top_y: float) -> float:
    """Free height under an object, from rays cast straight up out of the floor inside its
    footprint (first hit height each, inf for none). A desk's rays mostly hit the underside of the
    top; the few that hit a leg or a bin underneath do not count. Under 3 cm is solid to the
    floor (a dresser, a bin)."""
    h = np.minimum(first_up, top_y) - floor_y
    h = h[np.isfinite(h)]
    if not len(h):
        return 0.0
    c = float(np.percentile(h, 60))
    return 0.0 if c < 0.03 else c


def edges(box: dict, floor_y: float, clear: float) -> list[dict]:
    """Horizontal outlines a walker can strike: the top, and the underside when the object
    overhangs free space (clearance above 10 cm). Closed polylines, metres."""
    top = box["center"][1] + box["size"][1] / 2
    out = [{"kind": "top", "height_m": round(top - floor_y, 3), "points": box_corners(box, top)}]
    if clear > 0.10:
        y = floor_y + clear
        out.append(
            {"kind": "underside", "height_m": round(clear, 3), "points": box_corners(box, y)}
        )
    for e in out:
        p = e["points"]
        e["points"] = [[round(float(v), 3) for v in q] for q in np.vstack([p, p[:1]])]
    return out


def mass_material(label: str, size: list[float]) -> tuple[float | None, str | None, bool]:
    """Estimated mass (kg) and material from the class prior and the measured box."""
    if label not in PRIORS:
        return None, None, True
    material, kg, vol, movable = PRIORS[label]
    ratio = float(np.clip(np.prod(size) / vol, 0.5, 2.0))
    return round(kg * ratio, 1), material, movable


def continuity(pos: np.ndarray, jump_m: float = 0.3) -> float:
    """Share of consecutive frames (time order) whose camera jumps more than jump_m: a head moves
    a few centimetres a frame, so a jump is a frame registered in the wrong place."""
    if len(pos) < 2:
        return 0.0
    return float((np.linalg.norm(np.diff(pos, axis=0), axis=1) > jump_m).mean())


def trusted(pos: np.ndarray, jump_m: float = 0.15) -> np.ndarray:
    """Frames whose camera sits within jump_m of both time neighbours (or of its one neighbour at
    the ends of the walk): the poses that agree with their own walk."""
    n = len(pos)
    if n < 2:
        return np.ones(n, bool)
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1) <= jump_m
    ok = np.zeros(n, bool)
    ok[0], ok[-1] = d[0], d[-1]
    ok[1:-1] = d[:-1] & d[1:]
    return ok


def load_prior(path: Path, xf: np.ndarray | None = None, min_frames: int = 3) -> dict:
    """A recon-v2 objects.json as boxes and edges in the live map's frame.

    The bedroom home (data/rooms/bedroom/home) is built in the room mesh's frame, the same frame
    objects.json uses, so xf is None there. For another home, xf (4x4, y stays up) takes mesh
    coordinates into it. Returns {"floor_y", "objects": [...]}, each object with id, class,
    hazard, movable, center (3,), size [sx, height, sz], yaw_rad, corners (8, 3: top ring then
    bottom ring), top_height_m, floor_clearance_m and edges [{kind, height_m, points (n, 3)}].
    Objects seen in fewer than min_frames frames are left out. Heights stay floor-relative.
    """
    doc = json.loads(Path(path).read_text())
    xf = np.eye(4) if xf is None else np.asarray(xf, float)

    def move(p) -> np.ndarray:
        p = np.asarray(p, float)
        return p @ xf[:3, :3].T + xf[:3, 3]

    floor_y = float(move([0.0, doc["floor_y"], 0.0])[1])
    objs = []
    for o in doc["objects"]:
        if o.get("seen", {}).get("frames", min_frames) < min_frames:
            continue
        b = o["obb"]
        top = np.asarray(b["corners_top"], float)
        bot = top.copy()
        bot[:, 1] = b["center"][1] - b["size"][1] / 2
        axis = xf[:3, :3] @ [math.cos(b["yaw_rad"]), 0.0, math.sin(b["yaw_rad"])]
        objs.append(
            {
                "id": o["id"],
                "class": o["class"],
                "hazard": bool(o.get("hazard")),
                "movable": bool(o.get("movable", True)),
                "center": move(b["center"]),
                "size": list(b["size"]),
                "yaw_rad": math.atan2(axis[2], axis[0]),
                "corners": move(np.vstack([top, bot])),
                "top_height_m": o.get("top_height_m"),
                "floor_clearance_m": o.get("floor_clearance_m", 0.0),
                "edges": [
                    {"kind": e["kind"], "height_m": e["height_m"], "points": move(e["points"])}
                    for e in o.get("edges", [])
                ],
            }
        )
    return {"floor_y": floor_y, "objects": objs}
