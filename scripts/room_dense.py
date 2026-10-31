"""Every frame of every walk into one room mesh, on top of room_mesh.py's pooled anchors.

room_mesh.py places a few hundred chosen frames in one MapAnything pass. Here the remaining
frames go through in chunks, each chunk carrying anchors from the same walk; the anchors'
known depth pins the chunk into the room's frame (a similarity fit), and every frame's depth
is fused into one GPU TSDF. Passes go coarse to fine (every 8th frame, then the 4th, 2nd,
rest), so each pass is a whole-room improvement and a render comes out after each.

    .venv/bin/python scripts/room_dense.py <walk> [<walk> ...] --anchors <room_mesh out dir>
        --views 280 --out <dir> [--voxel 0.015] [--strides 8 4 2 1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from room_mesh import H, W, level, load_model, pool_frames, reconstruct, render, square_up  # noqa: E402

NEW, ANCHORS = 112, 16  # 128 views a chunk: about two and a half minutes on the mini
FIT_MAX_M = 0.06  # anchors farther than this from their own room position: chunk is misplaced


def embed(rgbs: list[np.ndarray]) -> np.ndarray:
    """Unit DINOv2 image vectors: frames of the same corner of the room point the same way."""
    global _dino
    if "_dino" not in globals():
        _dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to("mps").eval()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    x = np.stack([cv2.resize(im, (224, 392)) / 255.0 for im in rgbs])
    x = torch.tensor((x - mean) / std, dtype=torch.float32).permute(0, 3, 1, 2)
    with torch.no_grad():
        e = torch.cat([_dino(b.to("mps")).cpu() for b in x.split(32)]).numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Scale, rotation and translation taking src points onto dst."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    e = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ e @ vt
    s = float(np.trace(np.diag(d) @ e) / a.var(0).sum())
    return s, r, md - s * r @ ms


def world_pts(depth, k, pose, mask, step=6) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    d = depth[::step, ::step]
    ok = mask[::step, ::step] & (d > 0.1) & (d < 5.0)
    z = d
    cam = np.stack([(xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z], -1)
    return cam @ pose[:3, :3].T + pose[:3, 3], ok


def fit_chunk(
    views: list[dict], anchors: list[dict]
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Similarity from the chunk's frame to the room's, from the anchors' depth in both."""
    src, dst = [], []
    for v, a in zip(views, anchors, strict=True):
        ps, oks = world_pts(v["depth"], v["k"], v["pose"], v["mask"])
        pd, okd = world_pts(a["depth"], a["k"], a["pose"], a["mask"].astype(bool))
        ok = oks & okd
        src.append(ps[ok])
        dst.append(pd[ok])
    src, dst = np.concatenate(src), np.concatenate(dst)
    keep = np.ones(len(src), bool)
    for _ in range(3):  # refit on the points that agree
        s, r, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm(src * s @ r.T + t - dst, axis=1)
        keep = res < max(2.5 * np.median(res[keep]), 0.02)
    return s, r, t, float(np.median(res[keep]))


class Volume:
    """TSDF on the GPU: one frame integrates in milliseconds instead of a second in numpy."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray, voxel: float):
        self.lo, self.voxel = lo, voxel
        self.dims = np.ceil((hi - lo) / voxel).astype(int) + 1
        axes = [
            torch.arange(n, dtype=torch.float32) * voxel + float(o)
            for n, o in zip(self.dims, lo, strict=True)
        ]
        self.centres = (
            torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).to("mps")
        )
        n = len(self.centres)
        self.tsdf = torch.ones(n, device="mps")
        self.w = torch.zeros(n, device="mps")
        self.col = torch.zeros(n, 3, device="mps")
        self.trunc = 4 * voxel

    def add(self, depth, k, pose, mask, rgb) -> None:
        r = torch.tensor(pose[:3, :3], device="mps", dtype=torch.float32)
        t = torch.tensor(pose[:3, 3], device="mps", dtype=torch.float32)
        cam = (self.centres - t) @ r
        z = cam[:, 2]
        zc = z.clamp(min=1e-3)
        u = torch.round(cam[:, 0] * float(k[0, 0]) / zc + float(k[0, 2])).long()
        v = torch.round(cam[:, 1] * float(k[1, 1]) / zc + float(k[1, 2])).long()
        idx = torch.nonzero((z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).squeeze(1)
        u, v, z = u[idx], v[idx], z[idx]
        d = torch.tensor(depth, device="mps", dtype=torch.float32)[v, u]
        ok = torch.tensor(mask, device="mps")[v, u] & (d > 0.1) & (d < 5.0)
        sdf = d - z
        ok &= sdf > -self.trunc  # far behind the surface: unknown, not empty
        idx, sdf, d, u, v = idx[ok], (sdf[ok] / self.trunc).clamp(max=1.0), d[ok], u[ok], v[ok]
        wt = 1.0 / d.clamp(min=0.5)  # near depth is the sharper depth
        w0 = self.w[idx]
        self.tsdf[idx] = (self.tsdf[idx] * w0 + sdf * wt) / (w0 + wt)
        close = sdf.abs() < 0.5
        ci, wc = idx[close], wt[close, None]
        c = torch.tensor(rgb, device="mps", dtype=torch.float32)[v[close], u[close]]
        self.col[ci] = (self.col[ci] * self.w[ci, None] + c * wc) / (self.w[ci, None] + wc)
        self.w[idx] = w0 + wt

    def save(self, path: Path, **extra) -> None:
        """Checkpoint, written whole then renamed so a crash mid-write leaves the last one."""
        tmp = path.with_suffix(".tmp")
        state = dict(tsdf=self.tsdf.cpu(), w=self.w.cpu(), col=self.col.cpu(), **extra)
        torch.save(state, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> dict:
        state = torch.load(path, weights_only=False)
        self.tsdf, self.w = state.pop("tsdf").to("mps"), state.pop("w").to("mps")
        self.col = state.pop("col").to("mps")
        return state

    def mesh(self, min_weight: float):
        import open3d as o3d
        from skimage.measure import marching_cubes

        w = self.w.cpu().numpy()
        vol = np.where(w >= min_weight, self.tsdf.cpu().numpy(), 1.0).reshape(self.dims)
        verts, faces, _, _ = marching_cubes(vol, 0.0)
        vi = np.clip(np.round(verts).astype(int), 0, self.dims - 1)
        col = self.col.cpu().numpy().reshape(*self.dims, 3)[vi[:, 0], vi[:, 1], vi[:, 2]] / 255
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.lo + verts * self.voxel),
            o3d.utility.Vector3iVector(faces[:, ::-1].copy()),
        )
        m.vertex_colors = o3d.utility.Vector3dVector(col)
        labels, counts, _ = m.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        m.remove_triangles_by_mask(counts[labels] < max(500, 0.002 * len(labels)))
        m.remove_unreferenced_vertices()
        m.compute_vertex_normals()
        return m


def snapshot(
    vol: Volume, anchors: list[dict], out: Path, tag: str, frames: int, xf: np.ndarray | None
) -> None:
    """The mesh in room coordinates (floor y = 0, walls on the axes), with renders. xf, when
    given, is a room frame fixed earlier, so coordinates stay the same from version to version."""
    mesh = vol.mesh(min_weight=3.0)
    if xf is None:
        lv = level(mesh, anchors)
        mesh.rotate(lv, center=(0, 0, 0))
        floor = np.percentile(np.asarray(mesh.vertices)[:, 1], 2)
        mesh.translate((0, -floor, 0))
        sq = square_up(mesh)
        mesh.rotate(sq, center=(0, 0, 0))
        xf = np.eye(4)
        xf[:3, :3], xf[:3, 3] = sq @ lv, sq @ [0, -floor, 0]
    else:
        mesh.transform(xf)
    mesh.compute_vertex_normals()
    import open3d as o3d

    o3d.io.write_triangle_mesh(str(out / f"mesh-{tag}.ply"), mesh)
    # Anchor frame to mesh frame, so step trails and object positions can be drawn on it.
    (out / f"frame-{tag}.json").write_text(json.dumps(xf.tolist()))
    xyz = np.asarray(mesh.vertices)
    cut = min(np.percentile(xyz[:, 1], 98) - 0.15, 2.0)
    roof = o3d.geometry.TriangleMesh(mesh)
    roof.remove_vertices_by_mask(xyz[:, 1] > cut)
    roof.compute_vertex_normals()
    top = render(roof, "top")
    cv2.putText(top, f"{frames} frames", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 40, 40), 3)
    cv2.imwrite(str(out / f"top-{tag}.png"), top)
    tiles = [cv2.resize(render(roof, a), (900, 700)) for a in (45, 135, 225, 315)]
    cv2.imwrite(
        str(out / f"orbit-{tag}.png"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk", nargs="+")
    ap.add_argument("--anchors", required=True, help="room_mesh.py output dir for these walks")
    ap.add_argument("--views", type=int, required=True, help="--views given to room_mesh.py")
    ap.add_argument("--anchor-walks", nargs="+", help="walks --anchors came from (default: walk)")
    ap.add_argument("--start-from", help="state.pt of a finished run of the same anchors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.015)
    ap.add_argument("--strides", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--frame", help="frame-*.json of the room, to keep its coordinates")
    a = ap.parse_args()
    walks, out = [Path(w) for w in a.walk], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    anchor_walks = [Path(w) for w in a.anchor_walks] if a.anchor_walks else walks
    hfov = {
        w: json.loads((w / "camera.json").read_text()).get("hfov") for w in walks + anchor_walks
    }

    z = np.load(Path(a.anchors) / "views.npz")
    keys = ("pose", "depth", "mask", "k", "rgb")
    anchor_views = [{k: z[k][i] for k in keys} for i in range(len(z["pose"]))]
    anchor_paths = pool_frames(anchor_walks, a.views)
    assert len(anchor_paths) == len(anchor_views), "anchors came from other walks or --views"
    anchor_of = dict(zip(anchor_paths, anchor_views, strict=True))

    clouds = [
        world_pts(v["depth"], v["k"], v["pose"], v["mask"].astype(bool), 4) for v in anchor_views
    ]
    pts = np.concatenate([p[ok] for p, ok in clouds])
    vol = Volume(np.percentile(pts, 0.5, 0) - 0.3, np.percentile(pts, 99.5, 0) + 0.3, a.voxel)
    ckpt = out / "state.pt"
    if ckpt.exists() or a.start_from:  # a restart after a crash, or new walks on a finished room
        state = vol.load(ckpt if ckpt.exists() else Path(a.start_from))
        if not ckpt.exists():
            state["passes"] = []
        done, fused, passes = {Path(p) for p in state["done"]}, state["fused"], state["passes"]
        print(f"resumed: {fused} frames fused, passes done {passes}", flush=True)
    else:
        for v in anchor_views:
            vol.add(v["depth"], v["k"], v["pose"], v["mask"].astype(bool), v["rgb"])
        done, fused, passes = set(anchor_paths), len(anchor_paths), []

    anchor_emb = embed([v["rgb"] for v in anchor_views])
    model = load_model()
    log = (out / "chunks.jsonl").open("a")
    # Every fused frame's camera in the anchor frame: the reference trajectory of each walk.
    cams = (out / "cams.jsonl").open("a")
    t0 = time.time()
    for stride in a.strides:
        if stride in passes:
            continue
        for w in walks:
            rows = [json.loads(ln) for ln in (w / "frames.jsonl").read_text().splitlines() if ln]
            ts = {w / "frames" / r["file"]: r["ts"] for r in rows}
            todo = [p for p in list(ts)[::stride] if p not in done]
            # Each chunk samples the whole walk: a run of neighbouring frames sees one corner
            # from one spot and fits the room badly (half the every-frame chunks did).
            n = -(-len(todo) // NEW)
            todo = [q for i in range(n) for q in todo[i::n]]
            for c in range(0, len(todo), NEW):
                new = todo[c : c + NEW]
                # Anchors: the room frames that look most like this chunk.
                rgbs = [cv2.cvtColor(cv2.resize(cv2.imread(str(q)), (W, H)), 4) for q in new]
                sim = (embed(rgbs) @ anchor_emb.T).max(0)
                anc = [anchor_paths[i] for i in np.argsort(-sim)[:ANCHORS]]
                views = reconstruct(anc + new, hfov, model)
                s, r, t, res = fit_chunk(views[: len(anc)], [anchor_of[p] for p in anc])
                ok = res < FIT_MAX_M and 0.5 < s < 2.0
                log.write(
                    json.dumps(
                        dict(
                            walk=w.name,
                            stride=stride,
                            start=c,
                            n=len(new),
                            scale=s,
                            res_m=res,
                            used=ok,
                        )
                    )
                    + "\n"
                )
                log.flush()
                print(
                    f"stride {stride} {w.name} {c}+{len(new)}: fit {res * 100:.1f} cm,"
                    f" scale {s:.2f}, {'used' if ok else 'DROPPED'}",
                    flush=True,
                )
                done.update(new)
                if not ok:
                    continue
                for q, v in zip(new, views[len(anc) :], strict=True):
                    pose = np.eye(4)
                    pose[:3, :3] = r @ v["pose"][:3, :3]
                    pose[:3, 3] = s * r @ v["pose"][:3, 3] + t
                    vol.add(v["depth"] * s, v["k"], pose, v["mask"], v["rgb"])
                    row = dict(walk=w.name, file=q.name, pose=pose.tolist(), k=v["k"].tolist())
                    cams.write(json.dumps(row) + "\n")
                cams.flush()
                fused += len(new)
                vol.save(ckpt, done=[str(p) for p in done], fused=fused, passes=passes)
        xf = np.array(json.loads(Path(a.frame).read_text())) if a.frame else None
        snapshot(vol, anchor_views, out, f"s{stride}", fused, xf)
        passes.append(stride)
        vol.save(ckpt, done=[str(p) for p in done], fused=fused, passes=passes)
        print(f"pass stride {stride} done: {fused} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
