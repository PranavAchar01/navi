"""Where things are in the room: detections in the walk frames, traced onto the room mesh.

Every frame the dense run fused has its camera in the mesh's frame (cams.jsonl). A detection's
box centre gives a ray from that camera; where the ray hits the mesh is the object, in metres,
in room coordinates. Many frames see the same keyboard, so the hits cluster: the cluster's
median is its position and the spread says how sure that is.

    python scripts/room_objects.py --mesh $S/denseall --tag s8 --every 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from map_to_mesh import to_px  # noqa: E402
from room_mesh import H, W, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLUSTER_M = 0.25  # hits of one label closer than this are the same object
MIN_VIEWS = 4  # an object needs this many frames that agree
MIN_CONF = 0.4
MOVE_MIN_M = 0.15  # still things sit 6 to 13 cm from their indexed spot (see moved_by)
MOVE_K = 1.5
FLAG_VIEWS = 10  # a move is only flagged from an object this many frames of the walk agree on
SKIP = {"person"}  # people are not part of the room


def load_cams(mesh_dir: Path, xf: np.ndarray, every: int) -> list[dict]:
    rows = [json.loads(ln) for ln in (mesh_dir / "cams.jsonl").read_text().splitlines() if ln]
    rows = [r for r in rows if "k" in r][::every]
    for r in rows:
        r["pose"] = xf @ np.array(r["pose"])
        r["k"] = np.array(r["k"])
    return rows


def detect(rows: list[dict], weights: Path, batch: int = 32) -> list[dict]:
    """YOLO boxes, as box-centre rays in the mesh frame."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = []
    for b in range(0, len(rows), batch):
        chunk = rows[b : b + batch]
        imgs = [
            cv2.imread(str(ROOT / "data/walks" / r["walk"] / "frames" / r["file"])) for r in chunk
        ]
        for r, img, res in zip(chunk, imgs, model(imgs, verbose=False, conf=MIN_CONF), strict=True):
            sx, sy = W / img.shape[1], H / img.shape[0]
            k, pose = r["k"], r["pose"]
            for box, cls, conf in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                strict=True,
            ):
                if model.names[int(cls)] in SKIP:
                    continue
                u, v = (box[0] + box[2]) / 2 * sx, (box[1] + box[3]) / 2 * sy
                d = pose[:3, :3] @ np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0])
                out.append(
                    dict(
                        label=model.names[int(cls)],
                        conf=float(conf),
                        origin=pose[:3, 3],
                        dir=d / np.linalg.norm(d),
                        walk=r["walk"],
                        file=r["file"],
                    )
                )
    return out


def place(dets: list[dict], mesh) -> list[dict]:
    """Cast each detection's ray onto the mesh and cluster the hits per label."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.array([np.concatenate([d["origin"], d["dir"]]) for d in dets], np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    objects = []
    for label in sorted({d["label"] for d in dets}):
        idx = [i for i, d in enumerate(dets) if d["label"] == label and np.isfinite(t[i])]
        if len(idx) < MIN_VIEWS:
            continue
        hits = np.array([dets[i]["origin"] + t[i] * dets[i]["dir"] for i in idx])
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hits))
        lab = np.array(pc.cluster_dbscan(CLUSTER_M, MIN_VIEWS))
        for c in range(lab.max() + 1 if len(lab) else 0):
            m = lab == c
            pos = np.median(hits[m], 0)
            spread = float(np.median(np.linalg.norm(hits[m] - pos, axis=1)))
            seen = [dets[idx[j]] for j in np.flatnonzero(m)]
            best = max(seen, key=lambda d: d["conf"])
            objects.append(
                dict(
                    label=label,
                    pos=[round(float(x), 3) for x in pos],
                    spread_m=round(spread, 3),
                    views=int(m.sum()),
                    walks=sorted({d["walk"] for d in seen}),
                    conf=round(float(np.mean([d["conf"] for d in seen])), 2),
                    best_frame=f"{best['walk']}/{best['file']}",
                )
            )
    return sorted(objects, key=lambda o: -o["views"])


def moved_by(a: dict, b: dict) -> float:
    """How far apart two sightings must be to count as a move: the two placements' spreads,
    widened, with a floor. Walk 20261031-004732-glasses against the index built without it put
    the bed, laptop, keyboard, mouse, phone and a chair 6 to 13 cm from where the index had them,
    and the desk monitor ("tv", a wide box whose centre lands anywhere on it, spreads 18 cm) 43 cm
    off, which a flat 30 cm called a move."""
    return max(MOVE_MIN_M, MOVE_K * (a.get("spread_m", 0.0) + b.get("spread_m", 0.0)))


def changes(ref: list[dict], new: list[dict]) -> list[dict]:
    """What a walk saw differently from the room's index: an object with no object of its label
    within moved_by in the index has moved there (or is new). Each one flags its spot for
    recapture."""
    out = []
    for o in new:
        if o["views"] < FLAG_VIEWS:
            continue
        same = [r for r in ref if r["label"] == o["label"]]
        dist = [float(np.linalg.norm(np.subtract(r["pos"], o["pos"]))) for r in same]
        if not same or all(d > moved_by(r, o) for r, d in zip(same, dist, strict=True)):
            near = same[int(np.argmin(dist))] if same else None
            out.append(
                {
                    **o,
                    "change": "moved" if same else "new",
                    "nearest_m": round(min(dist), 2) if same else None,
                    "from": near["pos"] if near else None,
                }
            )
    return out


def draw(objects: list[dict], mesh, path: Path) -> None:
    import open3d as o3d

    roof = o3d.geometry.TriangleMesh(mesh)
    xyz = np.asarray(roof.vertices)
    roof.remove_vertices_by_mask(xyz[:, 1] > min(np.percentile(xyz[:, 1], 98) - 0.15, 2.0))
    roof.compute_vertex_normals()
    img = render(roof, "top")
    for o, p in zip(
        objects, to_px(np.array([[o["pos"][0], o["pos"][2]] for o in objects]), roof), strict=True
    ):
        p = tuple(int(x) for x in p)
        cv2.circle(img, p, 10, (30, 30, 200), -1)
        cv2.putText(
            img, o["label"], (p[0] + 12, p[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2
        )
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="room_dense.py output dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--every", type=int, default=4, help="use every n-th fused frame")
    ap.add_argument("--weights", default=str(ROOT / "yolo26s.pt"))
    ap.add_argument("--walk", help="only this walk's frames, compared against the full index")
    ap.add_argument("--exclude", help="leave this walk's frames out (to score a prior on it)")
    a = ap.parse_args()
    import open3d as o3d

    mdir = Path(a.mesh)
    xf = np.array(json.loads((mdir / f"frame-{a.tag}.json").read_text()))
    mesh = o3d.io.read_triangle_mesh(str(mdir / f"mesh-{a.tag}.ply"))
    rows = load_cams(mdir, xf, a.every)
    if a.walk:
        rows = [r for r in rows if r["walk"] == a.walk]
    if a.exclude:
        rows = [r for r in rows if r["walk"] != a.exclude]
    dets = detect(rows, Path(a.weights))
    objects = place(dets, mesh)
    if a.walk:
        ref = json.loads((mdir / f"objects-{a.tag}.json").read_text())
        flagged = changes(ref, objects)
        (mdir / f"objects-{a.walk}.json").write_text(json.dumps(objects, indent=1))
        (mdir / f"changes-{a.walk}.json").write_text(json.dumps(flagged, indent=1))
        print(f"{a.walk}: {len(objects)} objects, {len(flagged)} changed")
        for o in flagged:
            print(f"  {o['change']:5s} {o['label']:12s} {o['pos']} ({o['nearest_m']} m from index)")
        return
    out = f"{a.tag}-without-{a.exclude}" if a.exclude else a.tag
    (mdir / f"objects-{out}.json").write_text(json.dumps(objects, indent=1))
    draw(objects, mesh, mdir / f"objects-{out}.png")
    print(f"{len(rows)} frames, {len(dets)} detections, {len(objects)} objects")
    for o in objects[:25]:
        print(f"  {o['label']:12s} {o['pos']} +-{o['spread_m'] * 100:.0f} cm, {o['views']} views")


if __name__ == "__main__":
    main()
