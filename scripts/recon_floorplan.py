"""A clean floor plan of the bedroom for the Blender set: walls, openings, floor and ceiling, and
every object's pose, box and dominant colour. Written as data/rooms/bedroom/floorplan.json.

Walls: the room is squared to the axes (room_mesh.square_up), so walls are the mesh triangles
whose normal is horizontal and close to +-x or +-z, between 0.3 m and 2.0 m above the floor.
Their offsets along the normal axis are histogrammed (2 cm bins); the outermost strong peak on
each of the four sides is that side's wall. Its extent runs between the two walls it meets.

Openings: along a wall, 5 cm columns are tested for wall surface at three bands (low 0.1 to
0.8 m, mid 0.8 to 1.6 m, high 1.6 to 2.0 m). A run of columns over 0.5 m wide with no
surface low and mid is an open doorway; no surface mid only, with wall below, is a window. A
closed door is flush with the wall, so doors come from the door objects whose box meets it.

Colour: the median, in Lab, of the largest of three k-means clusters of the mesh's vertex
colours inside the object's box (shrunk by 10%). Vertex colours come from the footage.

    PYTHONPATH=. .venv/bin/python scripts/recon_floorplan.py --version v2.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOM = Path(__file__).resolve().parent.parent / "data/rooms/bedroom"
BAND = (0.3, 2.0)
BIN = 0.02
COL = 0.05
MIN_OPEN = 0.5


def wall_peaks(off: np.ndarray, w: np.ndarray) -> list[float]:
    """Offsets (m) of strong planar peaks: bins with at least 15% of the strongest bin's area."""
    lo, hi = off.min(), off.max()
    h, e = np.histogram(off, bins=max(1, int((hi - lo) / BIN)), range=(lo, hi), weights=w)
    h = np.convolve(h, [0.25, 0.5, 0.25], "same")
    pk = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1]]
    strong = [i for i in pk if h[i] >= 0.15 * h.max()]
    return [float((e[i] + e[i + 1]) / 2) for i in strong]


def openings(along: np.ndarray, up: np.ndarray, a0: float, a1: float) -> list:
    """Gaps in one wall's surface: open doorway (open low and mid) or window (open mid only)."""
    n = max(1, int((a1 - a0) / COL))
    idx = np.clip(((along - a0) / COL).astype(int), 0, n - 1)
    bands = [(0.1, 0.8), (0.8, 1.6), (1.6, 2.0)]
    has = np.zeros((3, n), bool)
    for b, (lo, hi) in enumerate(bands):
        sel = (up >= lo) & (up < hi)
        has[b, np.unique(idx[sel])] = True
    out = []
    for kind, gap in [("door", ~has[0] & ~has[1]), ("window", has[0] & ~has[1])]:
        i = 0
        while i < n:
            if not gap[i]:
                i += 1
                continue
            j = i
            while j < n and gap[j]:
                j += 1
            s, t = a0 + i * COL, a0 + j * COL
            if t - s >= MIN_OPEN and i > 0 and j < n:  # an open end is the wall running out
                o = {"type": kind, "from_m": round(s - a0, 2), "to_m": round(t - a0, 2)}
                o["width_m"] = round(t - s, 2)
                if kind == "door":
                    o["height_m"], o["source"] = 2.0, "gap in wall surface"
                else:
                    o["sill_m"], o["top_m"] = 0.8, 1.6
                out.append(o)
            i = j
    return out


def dominant_colour(rgb: np.ndarray) -> list[int] | None:
    if len(rgb) < 30:
        return None
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab = lab.astype(np.float32)
    k = min(3, len(lab))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab_id, _ = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    big = np.bincount(lab_id.ravel()).argmax()
    m = np.median(lab[lab_id.ravel() == big], 0).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(m, cv2.COLOR_LAB2RGB).reshape(3).astype(int).tolist()


def inside(pts: np.ndarray, b: dict, shrink: float = 0.9) -> np.ndarray:
    c, s, yaw = np.asarray(b["center"]), np.asarray(b["size"]) * shrink / 2, b["yaw_rad"]
    d = pts - c
    ax = np.array([np.cos(yaw), 0, np.sin(yaw)])
    az = np.array([-np.sin(yaw), 0, np.cos(yaw)])
    return (np.abs(d @ ax) <= s[0]) & (np.abs(d[:, 1]) <= s[1]) & (np.abs(d @ az) <= s[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", default=str(ROOM / "floorplan.json"))
    a = ap.parse_args()
    import open3d as o3d

    vdir = ROOM / "recon-v2" / a.version
    doc = json.loads((vdir / "objects.json").read_text())
    mesh = o3d.io.read_triangle_mesh(doc["source"]["mesh"])
    mesh.compute_triangle_normals()
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    nrm = np.asarray(mesh.triangle_normals)
    cen = v[t].mean(1)
    area = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    floor_y = doc["floor_y"]
    up = cen[:, 1] - floor_y
    band = (up > BAND[0]) & (up < BAND[1])

    # Each side: triangles facing along that axis; the outermost strong peak is the wall.
    side = {}
    for name, ax, sign in [("x-", 0, -1), ("x+", 0, 1), ("z-", 2, -1), ("z+", 2, 1)]:
        sel = band & (np.abs(nrm[:, ax]) > 0.9)
        pk = wall_peaks(cen[sel, ax], area[sel])
        side[name] = (min(pk) if sign < 0 else max(pk)) if pk else None
    x0, x1, z0, z1 = side["x-"], side["x+"], side["z-"], side["z+"]
    doors = [o for o in doc["objects"] if o["class"] == "door"]
    walls = []
    for name, ax, off, (a0, a1) in [
        ("west", 0, x0, (z0, z1)),
        ("east", 0, x1, (z0, z1)),
        ("north", 2, z0, (x0, x1)),
        ("south", 2, z1, (x0, x1)),
    ]:
        other = 2 - ax
        sel = (np.abs(cen[:, ax] - off) < 0.06) & (np.abs(nrm[:, ax]) > 0.8) & (up > 0) & (up < 2.0)
        # A closed door is flush with the wall, so doors come from the door objects.
        ops = openings(cen[sel, other], up[sel], a0, a1)
        for d in doors:
            b = d["obb"]
            if abs(b["center"][ax] - off) > 0.4:
                continue
            c, sn = abs(np.cos(b["yaw_rad"])), abs(np.sin(b["yaw_rad"]))
            sx, sz = b["size"][0], b["size"][2]
            half = (c * sx + sn * sz) / 2 if ax == 2 else (sn * sx + c * sz) / 2
            lo, hi = b["center"][other] - half - a0, b["center"][other] + half - a0
            ops.append(
                {
                    "type": "door",
                    "from_m": round(lo, 2),
                    "to_m": round(hi, 2),
                    "width_m": round(hi - lo, 2),
                    "height_m": round(b["center"][1] + b["size"][1] / 2 - floor_y, 2),
                    "source": d["id"],
                    "seen_frames": d["seen"]["frames"],
                }
            )
        p0 = [off, a0] if ax == 0 else [a0, off]
        p1 = [off, a1] if ax == 0 else [a1, off]
        walls.append(
            {
                "name": name,
                "start_xz": np.round(p0, 3).tolist(),
                "end_xz": np.round(p1, 3).tolist(),
                "length_m": round(a1 - a0, 3),
                "openings": sorted(ops, key=lambda o: o["from_m"]),
            }
        )
    cols = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    objs = []
    for o in doc["objects"]:
        b = o["obb"]
        objs.append(
            {
                "id": o["id"],
                "class": o["class"],
                "center": b["center"],
                "size": b["size"],
                "yaw_rad": b["yaw_rad"],
                "top_height_m": o.get("top_height_m"),
                "floor_clearance_m": o.get("floor_clearance_m"),
                "movable": o.get("movable"),
                "hazard": o.get("hazard"),
                "material_est": o.get("material_est"),
                "colour_rgb": dominant_colour(cols[inside(v, b)]),
            }
        )
    plan = {
        "version": a.version,
        "frame": doc["frame"] + "; xz plan coordinates are (x, z)",
        "floor_y": floor_y,
        "ceiling_height_m": doc["ceiling_height_m"],
        "room_xz": {"x": [round(x0, 3), round(x1, 3)], "z": [round(z0, 3), round(z1, 3)]},
        "scale_note": "metres as reconstructed; door and desk heights suggest about 10% too "
        "large, unconfirmed until a tape measurement",
        "walls": walls,
        "objects": objs,
    }
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(json.dumps({k: plan[k] for k in ["room_xz", "ceiling_height_m"]}))
    for w in walls:
        print(
            w["name"], w["length_m"], [(o["type"], o["from_m"], o["to_m"]) for o in w["openings"]]
        )


if __name__ == "__main__":
    main()
