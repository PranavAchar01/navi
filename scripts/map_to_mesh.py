"""Put the step map (the live MemoryEngine trail) into the room mesh's frame.

The room mesh's anchor views are one MapAnything pass over the walk, so their camera positions
are the best reference this room has. Fitting the live trail onto them (Sim3 on the keyframes
both share by timestamp) gives the transform from map to mesh coordinates and, from the
residuals, how far the step map is from the mesh: its error in metres on a real walk.

    python scripts/map_to_mesh.py data/walks/20261031-004732-glasses \
        bench/results/traj-20261031-004732-glasses-roomC.npz \
        --anchors $S/room160 --views 160 --mesh $S/denseall --tag s8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from room_dense import umeyama  # noqa: E402
from room_mesh import pool_frames, render  # noqa: E402

MATCH_S = 0.3  # a trail keyframe this close in time to an anchor view is the same moment


def anchor_times(walk: Path, views: int) -> np.ndarray:
    rows = {
        r["file"]: r["ts"] / 1000
        for r in map(json.loads, (walk / "frames.jsonl").read_text().splitlines())
    }
    return np.array([rows[p.name] for p in pool_frames([walk], views)])


def fit(trail: np.ndarray, t: np.ndarray, cams: np.ndarray, tc: np.ndarray):
    """Sim3 trail to anchor cameras on time-matched pairs, refit twice without the worst 20%."""
    j = np.abs(t[None, :] - tc[:, None]).argmin(1)
    ok = np.abs(t[j] - tc) < MATCH_S
    src, dst = trail[j[ok]], cams[ok]
    keep = np.ones(len(src), bool)
    for _ in range(3):
        s, r, tr = umeyama(src[keep], dst[keep])
        err = np.linalg.norm(s * src @ r.T + tr - dst, axis=1)
        keep = err <= np.quantile(err, 0.8)
    return s, r, tr, err, int(ok.sum())


def to_px(xz: np.ndarray, mesh, size: int = 1400) -> np.ndarray:
    """Mesh x/z to pixels of room_mesh.render(..., 'top'): +x right, +z down, ortho."""
    box = mesh.get_axis_aligned_bounding_box()
    c, ext = box.get_center(), box.get_extent()
    half = 0.52 * max(ext[0], ext[2])
    return np.stack(
        [size / 2 * (1 + (xz[:, 0] - c[0]) / half), size / 2 * (1 + (xz[:, 1] - c[2]) / half)], 1
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("trail", help="bench_map traj-*.npz of this walk")
    ap.add_argument("--anchors", required=True, help="room_mesh.py output dir (views.npz)")
    ap.add_argument("--views", type=int, required=True)
    ap.add_argument("--mesh", required=True, help="room_dense.py output dir")
    ap.add_argument("--tag", required=True, help="pass tag of the mesh, e.g. s8")
    a = ap.parse_args()
    walk, mdir = Path(a.walk), Path(a.mesh)
    tr = np.load(a.trail)
    trail, t = tr["est"], tr["t"]
    cams = np.load(Path(a.anchors) / "views.npz")["pose"][:, :3, 3].astype(float)
    s, r, tt, err, n = fit(trail, t, cams, anchor_times(walk, a.views))
    xf = np.array(json.loads((mdir / f"frame-{a.tag}.json").read_text()))
    # map -> anchor frame -> mesh frame
    m2a = np.eye(4)
    m2a[:3, :3], m2a[:3, 3] = s * r, tt
    m2m = xf @ m2a
    stats = {
        "trail": a.trail,
        "pairs": n,
        "scale": round(float(s), 3),
        "median_err_m": round(float(np.median(err)), 3),
        "p90_err_m": round(float(np.quantile(err, 0.9)), 3),
        "map_to_mesh": m2m.tolist(),
    }
    out = mdir / f"map-{Path(a.trail).stem}.json"
    out.write_text(json.dumps(stats, indent=1))
    print({k: v for k, v in stats.items() if k != "map_to_mesh"})

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mdir / f"mesh-{a.tag}.ply"))
    xyz = np.asarray(mesh.vertices)
    mesh.remove_vertices_by_mask(xyz[:, 1] > min(np.percentile(xyz[:, 1], 98) - 0.15, 2.0))
    mesh.compute_vertex_normals()
    img = render(mesh, "top")
    path = trail @ m2m[:3, :3].T + m2m[:3, 3]
    anc = cams @ xf[:3, :3].T + xf[:3, 3]
    for p in to_px(anc[:, [0, 2]], mesh).astype(int):
        cv2.circle(img, tuple(p), 5, (60, 160, 60), -1)
    pts = to_px(path[:, [0, 2]], mesh).astype(np.int32)
    cv2.polylines(img, [pts], False, (40, 40, 220), 3, cv2.LINE_AA)
    cv2.putText(
        img,
        f"step map on mesh: median {stats['median_err_m'] * 100:.0f} cm off ({n} matched)",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (40, 40, 40),
        3,
    )
    cv2.imwrite(str(mdir / f"trail-{Path(a.trail).stem}.png"), img)


if __name__ == "__main__":
    main()
