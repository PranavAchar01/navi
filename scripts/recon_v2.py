"""The bedroom as one metric map of objects, scored, under data/rooms/bedroom/recon-v2/<version>/.

Input is the dense run's mesh and cameras (room_dense.py, kept current by room_watch.py). Three
steps, each cached in the version folder:

  label   Open-vocabulary masks (YOLOE, text prompts for furniture and clutter) on frames from every
          fused walk, cast onto the mesh: each pixel's ray picks the triangle it sees, so each
          triangle collects votes per class over hundreds of views (the 2D-to-3D mask fusion of
          ConceptGraphs and OpenMask3D, with the mesh standing in for their point clouds). Only
          frames whose pose agrees with its time neighbours vote (recon.trusted).
  objects Triangles that win their votes are clustered per class into instances; each gets an
          upright box, its free height underneath and top height from rays cast up from the floor
          and down onto it, edge polylines, mass and material from class priors, and the frames
          that saw it. Written as objects.json.
  score   Map quality: pose continuity per walk, cross-walk reprojection (keypoints of one walk's
          frame lifted onto the mesh and projected into a frame of another walk), held-out
          localisation (frames of walks the map never fused, placed by PnP against it) and a
          scale check against standard sizes. Written as metrics.json.

    PYTHONPATH=. .venv/bin/python scripts/recon_v2.py --version v2.0 [--mesh DIR --tag s1]
        [--steps label objects score] [--frames 2000]

Heavy steps take a heavy-lock slot (room_watch.take_slot) and run under nice/taskpolicy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metasight import recon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROOM = ROOT / "data/rooms/bedroom"
WALKS = ROOT / "data/walks"
PACIFIC = ZoneInfo("America/Los_Angeles")
W, H = 294, 518  # the dense run's intrinsics are for this frame size
GW, GH = 147, 259  # ray grid per frame for voting
CLASSES = [
    "desk",
    "office chair",
    "chair",
    "computer monitor",
    "bed",
    "shelf",
    "keyboard",
    "laptop",
    "door",
    "trash can",
    "laundry basket",
    "whiteboard",
    "backpack",
    "bag",
    "lamp",
    "dresser",
    "cabinet",
    "nightstand",
    "box",
    "robot arm",
    "speaker",
    "pillow",
    "plant",
    "fan",
    "computer",
    "mini fridge",
    "shoe",
]
# Standing on the floor: the box goes down to it even when the masks only covered the top.
FLOOR_STANDING = {
    "desk",
    "office chair",
    "chair",
    "bed",
    "shelf",
    "dresser",
    "cabinet",
    "nightstand",
    "trash can",
    "laundry basket",
    "plant",
    "fan",
    "mini fridge",
    "door",
}
MIN_VOTES, MIN_SHARE = 2.0, 0.3
CLUSTER_M, MIN_TRIS = 0.05, 60
# Furniture is masked in pieces between the things on it (a desk around its keyboard and mat).
BIG = {"desk", "bed", "door", "shelf", "dresser", "cabinet", "whiteboard", "office chair", "chair"}
BIG_CLUSTER_M = 0.12
# Reference sizes for the scale check (US residential standards).
REF = {"desk_top_m": 0.74, "door_height_m": 2.03, "ceiling_8ft_m": 2.44, "ceiling_9ft_m": 2.74}


def ts_of(f: str) -> int:
    return int(f.split("-")[0].split(".")[0])


def walk_ts(walk: str, f: str) -> int:
    """Epoch ms of a frame. Phone and glasses frames are named by it; the video walk's frames
    are numbered, so they count from the walk's start (its name, Pacific time) at 30 fps."""
    t = ts_of(f)
    if t > 10**11:
        return t
    start = datetime.strptime(walk[:15], "%Y%m%d-%H%M%S").replace(tzinfo=PACIFIC)
    return int(start.timestamp() * 1000 + t * 1000 / 30)


def load_cams(mesh_dir: Path, tag: str, path: Path | None = None) -> dict[str, list[dict]]:
    """Cameras in the mesh frame, one per frame (the last pass wins), in time order per walk.
    path: a recon_reloc.py cams file instead of the dense run's (already in the mesh frame)."""
    xf = np.array(json.loads((mesh_dir / f"frame-{tag}.json").read_text()))
    if path:
        xf = np.eye(4)
    by: dict[str, dict[str, dict]] = {}
    for ln in (path or mesh_dir / "cams.jsonl").read_text().splitlines():
        r = json.loads(ln) if ln else None
        if r and "k" in r:
            r["pose"], r["k"] = xf @ np.array(r["pose"]), np.array(r["k"])
            by.setdefault(r["walk"], {})[r["file"]] = r
    out = {}
    for w, d in by.items():
        rows = sorted(d.values(), key=lambda r: ts_of(r["file"]))
        ok = recon.trusted(np.array([r["pose"][:3, 3] for r in rows]))
        for r, t in zip(rows, ok, strict=True):
            r["trusted"] = bool(t)
        out[w] = rows
    return out


def floor_layers(verts: np.ndarray) -> tuple[float, float]:
    """The floor height and how thick the floor came out. Walks registered a little off each other
    leave the floor as two sheets about 10 cm apart (v002), so the floor is the middle of the
    lowest and highest density peaks of the bottom 40 cm, and the gap between them is a quality
    number: 0 for a mesh whose walks agree."""
    y = verts[:, 1]
    lo = np.percentile(y, 0.5)
    h, e = np.histogram(y[(y > lo - 0.05) & (y < lo + 0.4)], bins=45)
    h = np.convolve(h, [1, 2, 1], "same")
    peaks = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1]]
    peaks = [i for i in peaks if h[i] > 0.3 * h.max()]
    mid = (e[:-1] + e[1:]) / 2
    a, b = mid[min(peaks)], mid[max(peaks)]
    if b - a > 0.25:  # the upper peak is furniture (a bed, a pile), not a second floor sheet
        b = a
    return float((a + b) / 2), float(b - a)


def ceiling_height(verts: np.ndarray, floor_y: float) -> float:
    y = verts[:, 1]
    hi = np.percentile(y, 99.5)
    h, e = np.histogram(y[(y > hi - 0.6) & (y < hi + 0.1)], bins=35)
    i = int(np.argmax(h))
    return float((e[i] + e[i + 1]) / 2 - floor_y)


def scene_of(mesh):
    import open3d as o3d

    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return s


def cast(scene, origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    rays = np.concatenate([origins, dirs], 1).astype(np.float32)
    r = scene.cast_rays(o3d.core.Tensor(rays))
    return r["t_hit"].numpy(), r["primitive_ids"].numpy().astype(np.int64)


def grid_rays(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    k, pose = cam["k"], cam["pose"]
    u, v = np.meshgrid((np.arange(GW) + 0.5) * W / GW, (np.arange(GH) + 0.5) * H / GH)
    d = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1)
    d = d.reshape(-1, 3) @ pose[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.repeat(pose[None, :3, 3], len(d), 0), d


def pick_frames(cams: dict[str, list[dict]], n: int) -> list[dict]:
    """Trusted frames spread evenly over each walk, walks weighted by their trusted count."""
    pool = {w: [r for r in rows if r["trusted"]] for w, rows in cams.items()}
    total = sum(len(v) for v in pool.values())
    out = []
    for _w, rows in pool.items():
        k = max(1, round(n * len(rows) / max(total, 1))) if rows else 0
        out += [rows[int(i)] for i in np.linspace(0, len(rows) - 1, min(k, len(rows)))]
    return out


def label(cams, mesh, scene, n: int, out: Path) -> None:
    from ultralytics import YOLOE

    model = YOLOE(str(ROOT / "yoloe-26l-seg.pt"))
    model.set_classes(CLASSES, model.get_text_pe(CLASSES))
    ntri = len(mesh.triangles)
    votes = np.zeros((ntri, len(CLASSES)), np.float32)
    vis = np.zeros(ntri, np.float32)
    frames = pick_frames(cams, n)
    seen = []  # (frame index, class, triangle ids)
    t0 = time.time()
    for b in range(0, len(frames), 16):
        chunk = frames[b : b + 16]
        paths = [str(WALKS / r["walk"] / "frames" / r["file"]) for r in chunk]
        res = model(paths, verbose=False, conf=0.3, device="mps", imgsz=640)
        for j, (r, det) in enumerate(zip(chunk, res, strict=True)):
            o, d = grid_rays(r)
            t, prim = cast(scene, o, d)
            hit = np.isfinite(t) & (t < 5.0)
            vis += np.bincount(prim[hit], minlength=ntri).astype(np.float32)
            if det.masks is None:
                continue
            for poly, c, p in zip(det.masks.xyn, det.boxes.cls, det.boxes.conf, strict=True):
                if len(poly) < 3:
                    continue
                m = np.zeros((GH, GW), np.uint8)
                cv2.fillPoly(m, [np.round(poly * [GW, GH]).astype(np.int32)], 1)
                sel = (m.reshape(-1) > 0) & hit
                ids = np.unique(prim[sel])
                if not len(ids):
                    continue
                votes[ids, int(c)] += float(p)
                seen.append((b + j, int(c), ids))
        if b % 320 == 0:
            print(f"label {b}/{len(frames)} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out / "votes.npz", votes=votes, vis=vis)
    rows = [{"walk": r["walk"], "file": r["file"]} for r in frames]
    (out / "voted_frames.json").write_text(json.dumps(rows))
    np.savez_compressed(
        out / "seen.npz",
        frame=np.array([s[0] for s in seen]),
        cls=np.array([s[1] for s in seen]),
        n=np.array([len(s[2]) for s in seen]),
        ids=np.concatenate([s[2] for s in seen]) if seen else np.zeros(0, np.int64),
    )


def objects(mesh, scene, out: Path, version: str, mesh_dir: Path, tag: str) -> dict:

    v = np.load(out / "votes.npz")
    votes, vis = v["votes"], v["vis"]
    s = dict(np.load(out / "seen.npz"))  # an NpzFile decompresses on every access
    frames = json.loads((out / "voted_frames.json").read_text())
    starts = np.concatenate([[0], np.cumsum(s["n"])])
    verts, tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    floor_y, floor_gap = floor_layers(verts)
    objs = [
        describe(name, ids, pts, box, votes, vis, floor_y, floor_gap, scene)
        for name, ids, pts, box in clusters(votes, vis, verts, tris)
    ]
    attach_seen(objs, len(tris), s, starts, frames)
    objs = merge(objs)
    objs.sort(key=lambda o: -o["seen"]["frames"])
    for i, o in enumerate(objs):
        o["id"] = f"{o['class'].replace(' ', '_')}_{i:02d}"
    ceil_h = ceiling_height(verts, floor_y)
    doc = {
        "version": version,
        "source": {"mesh": str(mesh_dir / f"mesh-{tag}.ply"), "frame": f"frame-{tag}.json"},
        "frame": "room mesh frame: y up, metres as reconstructed, yaw about +y; box local x axis "
        "is (cos yaw, 0, sin yaw)",
        "floor_y": round(floor_y, 3),
        "floor_sheet_gap_m": round(floor_gap, 3),
        "ceiling_height_m": round(ceil_h, 3),
        "objects": objs,
    }
    (out / "objects.json").write_text(json.dumps(doc, indent=1))
    return doc


def clusters(votes, vis, verts, tris, min_votes=MIN_VOTES, min_share=MIN_SHARE):
    """Triangles won by one class, split into separate instances: (class, ids, points, box)."""
    import open3d as o3d

    cent = verts[tris].mean(1)
    win = votes.argmax(1)
    share = votes.max(1) / np.maximum(vis, 1.0)
    keep = (votes.max(1) >= min_votes) & (share >= min_share)
    out = []
    for ci, name in enumerate(CLASSES):
        idx = np.flatnonzero(keep & (win == ci))
        if len(idx) < MIN_TRIS:
            continue
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cent[idx]))
        lab = np.array(pc.cluster_dbscan(BIG_CLUSTER_M if name in BIG else CLUSTER_M, 8))
        for c in range(lab.max() + 1 if len(lab) else 0):
            ids = idx[lab == c]
            if len(ids) < MIN_TRIS:
                continue
            pts = verts[tris[ids].reshape(-1)]
            box = recon.upright_box(pts, snap=1.35 if name in BIG else 1.0)
            if max(box["size"]) >= 0.05:
                out.append((name, ids, pts, box))
    return out


def describe(name, ids, pts, box, votes, vis, floor_y, floor_gap, scene) -> dict:
    if name in FLOOR_STANDING or box["center"][1] - box["size"][1] / 2 - floor_y < 0.25:
        top = box["center"][1] + box["size"][1] / 2
        box["center"][1], box["size"][1] = (top + floor_y) / 2, top - floor_y
    top = box["center"][1] + box["size"][1] / 2
    g = recon.footprint_grid(box)
    n = len(g)
    # Down from just above the box onto it: the top surface.
    o = np.column_stack([g[:, 0], np.full(n, top + 0.05), g[:, 1]])
    t, _ = cast(scene, o, np.tile([0.0, -1.0, 0.0], (n, 1)))
    top_y = float(np.median(top + 0.05 - t[np.isfinite(t)])) if np.isfinite(t).any() else top
    # Up out of the floor: the free height underneath.
    # Start above the upper floor sheet, so a doubled floor is not read as something underneath.
    y0 = floor_y + floor_gap / 2 + 0.03
    o = np.column_stack([g[:, 0], np.full(n, y0), g[:, 1]])
    t, _ = cast(scene, o, np.tile([0.0, 1.0, 0.0], (n, 1)))
    clear = recon.clearance(y0 + t, floor_y, top_y)
    mass, material, movable = recon.mass_material(name, box["size"])
    conf = float(np.median(votes[ids].max(1) / np.maximum(vis[ids], 1.0)))
    return {
        "class": name,
        # label and pos as home/objects.json has them, so roomprior.py can read this file as is
        "label": name,
        "pos": [round(x, 3) for x in box["center"]],
        "conf": round(conf, 2),
        "pose": {
            "position": [round(x, 3) for x in box["center"]],
            "yaw_rad": round(box["yaw"], 3),
        },
        "obb": {
            "center": [round(x, 3) for x in box["center"]],
            "size": [round(x, 3) for x in box["size"]],
            "yaw_rad": round(box["yaw"], 3),
            "corners_top": [[round(float(v), 3) for v in p] for p in recon.box_corners(box, top)],
        },
        "size_m": {
            "width": round(max(box["size"][0], box["size"][2]), 3),
            "depth": round(min(box["size"][0], box["size"][2]), 3),
            "height": round(box["size"][1], 3),
        },
        "top_height_m": round(top_y - floor_y, 3),
        "floor_clearance_m": round(clear, 3),
        "edges": recon.edges(box, floor_y, clear),
        "mass_kg_est": mass,
        "material_est": material,
        "movable": movable,
        "hazard": name in recon.HAZARD,
        "n_triangles": int(len(ids)),
        "_tris": ids,
    }


def attach_seen(objs, ntri, s, starts, frames) -> None:
    """Which voted frames saw each object: a frame's mask of the object's class covering at least
    10 of its triangles. One pass over the masks, with a triangle-to-object table."""
    owner = np.full(ntri, -1, np.int32)
    for i, o in enumerate(objs):
        owner[o["_tris"]] = i
    cls = [CLASSES.index(o["class"]) for o in objs]
    hits: list[list[tuple[int, int]]] = [[] for _ in objs]
    for r in range(len(s["cls"])):
        own = owner[s["ids"][starts[r] : starts[r + 1]]]
        own = own[own >= 0]
        for i, k in zip(*np.unique(own, return_counts=True), strict=True):
            if k >= 10 and cls[i] == s["cls"][r]:
                hits[i].append((int(k), int(s["frame"][r])))
    for o, h in zip(objs, hits, strict=True):
        walks: dict[str, int] = {}
        for _k, f in h:
            walks[frames[f]["walk"]] = walks.get(frames[f]["walk"], 0) + 1
        ts = [walk_ts(frames[f]["walk"], frames[f]["file"]) for _k, f in h]
        o["seen"] = {
            "frames": len(h),
            "walks": dict(sorted(walks.items())),
            "best_frames": [
                f"{frames[f]['walk']}/{frames[f]['file']}" for _k, f in sorted(h)[::-1][:8]
            ],
            "first_ts_ms": min(ts) if ts else None,
            "last_ts_ms": max(ts) if ts else None,
        }


def merge(objs: list[dict]) -> list[dict]:
    """Same class, plan centres closer than half the larger box: one object seen in pieces.
    Keeps the one with more triangles and adds the other's sightings."""
    objs = sorted(objs, key=lambda o: -o["n_triangles"])
    out: list[dict] = []
    for o in objs:
        c = np.array(o["obb"]["center"])[[0, 2]]
        for p in out:
            if p["class"] != o["class"]:
                continue
            d = np.linalg.norm(c - np.array(p["obb"]["center"])[[0, 2]])
            if d < 0.5 * max(p["size_m"]["width"], o["size_m"]["width"]):
                p["seen"]["frames"] += o["seen"]["frames"]
                for w, n in o["seen"]["walks"].items():
                    p["seen"]["walks"][w] = p["seen"]["walks"].get(w, 0) + n
                break
        else:
            out.append(o)
    for o in out:
        o.pop("_tris", None)
    # A laptop or pillow whose top is within 15 cm of the floor is clothing or clutter on the
    # floor called by the nearest prompt. It stays, as a trip hazard, under its own name.
    low_ok = {"shoe", "bag", "backpack", "box", "laundry basket", "trash can"}
    for o in out:
        if o["top_height_m"] < 0.15 and o["class"] not in low_ok:
            o["detector_class"], o["class"], o["label"] = (
                o["class"],
                "floor clutter",
                "floor clutter",
            )
            o["hazard"], o["movable"] = True, True
            o["mass_kg_est"], o["material_est"] = 0.5, "fabric or unknown"
    return [o for o in out if o["seen"]["frames"] >= 3]


def draw(doc: dict, mesh, path: Path) -> None:
    """Plan view: the mesh from above with the ceiling cut off, each object's box outline."""
    import open3d as o3d
    from map_to_mesh import to_px
    from room_mesh import render

    roof = o3d.geometry.TriangleMesh(mesh)
    roof.remove_vertices_by_mask(np.asarray(roof.vertices)[:, 1] > doc["floor_y"] + 2.0)
    roof.compute_vertex_normals()
    img = render(roof, "top")
    for o in doc["objects"]:
        c = np.array(o["obb"]["corners_top"])
        px = to_px(c[:, [0, 2]], roof).astype(np.int32)
        col = (40, 40, 220) if o["hazard"] else (200, 120, 30)
        cv2.polylines(img, [px], True, col, 3)
        cv2.putText(img, o["id"], tuple(px[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--mesh", help="room_dense.py output dir (default: current.json's)")
    ap.add_argument("--tag")
    ap.add_argument("--cams", help="recon_reloc.py cams file (default: the dense run's)")
    ap.add_argument("--steps", nargs="+", default=["label", "objects", "score"])
    ap.add_argument("--frames", type=int, default=2000)
    a = ap.parse_args()
    import open3d as o3d

    if a.mesh:
        mesh_dir, tag = Path(a.mesh), a.tag
    else:
        cur = json.loads((ROOM / "current.json").read_text())
        mesh_dir, tag = ROOM / cur["current"], cur["tag"]
    out = ROOM / "recon-v2" / a.version
    out.mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(str(mesh_dir / f"mesh-{tag}.ply"))
    scene = scene_of(mesh)
    cams = load_cams(mesh_dir, tag, Path(a.cams) if a.cams else None)
    from room_watch import take_slot

    slot = take_slot()
    try:
        if "label" in a.steps:
            label(cams, mesh, scene, a.frames, out)
        if "objects" in a.steps:
            doc = objects(mesh, scene, out, a.version, mesh_dir, tag)
            print(f"{len(doc['objects'])} objects, floor {doc['floor_y']}")
            draw(doc, mesh, out / "objects.png")
        if "score" in a.steps:
            import recon_score

            m = recon_score.score(cams, mesh, scene, out)
            (out / "metrics.json").write_text(json.dumps(m, indent=1))
            print(json.dumps(m["summary"], indent=1))
    finally:
        shutil.rmtree(slot, ignore_errors=True)
    shutil.copy(mesh_dir / f"frame-{tag}.json", out / "frame.json")
    splat = ROOM / "home/splat.ply"
    if splat.exists() and not (out / "splat.ply").is_symlink():
        os.symlink(splat.resolve(), out / "splat.ply")
    if not (out / "mesh.ply").is_symlink():
        os.symlink((mesh_dir / f"mesh-{tag}.ply").resolve(), out / "mesh.ply")


if __name__ == "__main__":
    main()
