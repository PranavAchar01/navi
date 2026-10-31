"""A room as one mesh, offline, from a recorded walk: the backend's picture of the place.

    .venv/bin/python scripts/room_mesh.py data/walks/<walk> --views 96 --out bench/rooms/<name>

Frames are picked for sharpness and new viewpoints, MapAnything reconstructs all of them in one
pass (no window stitching, which is where the live map's noise comes from), and the depth maps
are fused into a TSDF volume and meshed. Writes mesh.ply, top.png (straight down, ceiling cut off)
and orbit.png (four views from above the corners).
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

W, H = 294, 518  # portrait glasses frames at the model's patch grid (14 px)


def pick_frames(walk: Path, n: int) -> list[Path]:
    """Sharp frames that look somewhere new: standing still at the desk for minutes adds nothing."""
    rows = [json.loads(ln) for ln in (walk / "frames.jsonl").read_text().splitlines() if ln]
    cands, last_t = [], -1e9
    for r in rows:
        if r["ts"] - last_t >= 250:  # 4 per second is plenty to choose from
            cands.append(r)
            last_t = r["ts"]
    thumbs, sharp = [], []
    for r in cands:
        g = cv2.imread(str(walk / "frames" / r["file"]), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        sharp.append(cv2.Laplacian(g, cv2.CV_32F).var())
        thumbs.append(np.float32(cv2.resize(g, (48, 84))))
    sharp = np.array(sharp)
    ok = sharp >= np.quantile(sharp, 0.3)
    kept: list[int] = []
    for i in np.flatnonzero(ok):
        if kept:
            (dx, dy), resp = cv2.phaseCorrelate(thumbs[kept[-1]], thumbs[i])
            moved = math.hypot(dx, dy) > 6 or resp < 0.25
            if not moved:
                if sharp[i] > sharp[kept[-1]]:  # same view: keep the sharper copy
                    kept[-1] = i
                continue
        kept.append(int(i))
    if n and len(kept) > n:
        kept = [kept[k] for k in np.linspace(0, len(kept) - 1, n).round().astype(int)]
    return [walk / "frames" / cands[i]["file"] for i in kept]


def pool_frames(walks: list[Path], n: int) -> list[Path]:
    """Novel frames from every walk of the room, each walk's share set by how much new it saw."""
    novel = [pick_frames(w, 0) for w in walks]
    total = sum(map(len, novel))
    picked = []
    for w, paths in zip(walks, novel, strict=True):
        k = min(len(paths), max(1, round(n * len(paths) / total)))
        picked += [paths[i] for i in np.linspace(0, len(paths) - 1, k).round().astype(int)]
        print(f"{w.name}: {k} of {len(paths)} novel", flush=True)
    return picked


def load_model():
    from mapanything.models import MapAnything

    return MapAnything.from_pretrained("facebook/map-anything-apache").to("mps").eval()


def reconstruct(paths: list[Path], hfov: dict[Path, float | None], model=None) -> list[dict]:
    import torch
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

    model = model or load_model()
    norm = IMAGE_NORMALIZATION_DICT["dinov2"]
    mean, std = np.array(norm.mean, np.float32), np.array(norm.std, np.float32)
    views, rgbs = [], []
    for i, p in enumerate(paths):
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (W, H)), cv2.COLOR_BGR2RGB)
        rgbs.append(rgb)
        x = torch.from_numpy((rgb.astype(np.float32) / 255 - mean) / std).permute(2, 0, 1)[None]
        v = dict(img=x, true_shape=np.int32([[H, W]]), idx=i, instance=str(i))
        v["data_norm_type"] = ["dinov2"]
        if fov := hfov[p.parent.parent]:
            f = 0.5 * W / math.tan(math.radians(fov) / 2)
            k = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
            v["intrinsics"] = torch.from_numpy(k)[None]
        views.append(v)
    with torch.no_grad():
        preds = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="fp16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            confidence_percentile=20,
        )
    out = []
    for p, rgb in zip(preds, rgbs, strict=True):
        out.append(
            dict(
                pts=p["pts3d"][0].float().cpu().numpy(),
                mask=p["mask"][0].squeeze(-1).cpu().numpy().astype(bool),
                pose=p["camera_poses"][0].float().cpu().numpy(),
                depth=p["depth_z"][0].squeeze(-1).float().cpu().numpy(),
                k=p["intrinsics"][0].float().cpu().numpy(),
                rgb=rgb,
            )
        )
    return out


def _cloud(v: dict, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """A view's valid depth as world points and colours."""
    d, k = v["depth"][::step, ::step], v["k"]
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    ok = v["mask"][::step, ::step] & (d > 0.1) & (d < 5.0)
    z = d[ok]
    cam = np.stack([(xs[ok] - k[0, 2]) * z / k[0, 0], (ys[ok] - k[1, 2]) * z / k[1, 1], z], 1)
    return cam @ v["pose"][:3, :3].T + v["pose"][:3, 3], v["rgb"][::step, ::step][ok]


def fuse(views: list[dict], voxel: float):
    """TSDF fusion and marching cubes (open3d's legacy TSDF integrates nothing on this Mac)."""
    import open3d as o3d
    from skimage.measure import marching_cubes

    pts = np.concatenate([_cloud(v, 4)[0] for v in views])
    lo, hi = np.percentile(pts, 0.5, 0) - 0.1, np.percentile(pts, 99.5, 0) + 0.1
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    gx, gy, gz = (lo[a] + voxel * np.arange(dims[a]) for a in range(3))
    centres = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
    trunc = 4 * voxel
    tsdf = np.ones(len(centres), np.float32)
    wsum = np.zeros(len(centres), np.float32)
    col = np.zeros((len(centres), 3), np.float32)
    for v in views:
        r, t, k = v["pose"][:3, :3], v["pose"][:3, 3], v["k"]
        cam = (centres - t) @ r  # world to camera
        z = cam[:, 2]
        front = z > 0.1
        u = np.full(len(z), -1)
        w_ = np.full(len(z), -1)
        u[front] = np.round(cam[front, 0] * k[0, 0] / z[front] + k[0, 2]).astype(int)
        w_[front] = np.round(cam[front, 1] * k[1, 1] / z[front] + k[1, 2]).astype(int)
        inside = front & (u >= 0) & (u < W) & (w_ >= 0) & (w_ < H)
        idx = np.flatnonzero(inside)
        d = v["depth"][w_[idx], u[idx]]
        ok = v["mask"][w_[idx], u[idx]] & (d > 0.1) & (d < 5.0)
        idx, d = idx[ok], d[ok]
        sdf = d - z[idx]
        near = sdf > -trunc  # behind the surface by more than the band: unknown, not empty
        idx, sdf, d = idx[near], np.minimum(sdf[near] / trunc, 1.0), d[near]
        wt = 1.0 / np.maximum(d, 0.5)  # near depth is the sharper depth
        tsdf[idx] = (tsdf[idx] * wsum[idx] + sdf * wt) / (wsum[idx] + wt)
        rgb = v["rgb"][w_[idx], u[idx]].astype(np.float32)
        close = np.abs(sdf) < 0.5
        ci = idx[close]
        col[ci] = (col[ci] * wsum[ci, None] + rgb[close] * wt[close, None]) / (
            wsum[ci, None] + wt[close, None]
        )
        wsum[idx] += wt
    vol = np.where(wsum >= 2.0, tsdf, 1.0).reshape(dims)  # seen by two views at least
    verts, faces, _, _ = marching_cubes(vol, 0.0)
    verts = lo + verts * voxel
    vi = np.clip(np.round((verts - lo) / voxel).astype(int), 0, dims - 1)
    vcol = col.reshape(*dims, 3)[vi[:, 0], vi[:, 1], vi[:, 2]] / 255
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces[:, ::-1].copy())
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(vcol)
    # Floaters: keep the pieces that make up the room, drop specks from bad depth.
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[labels] < max(200, 0.002 * len(labels)))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def level(mesh, views: list[dict]) -> np.ndarray:
    """Rotation putting the floor flat (y up): the biggest plane facing the cameras' 'down'."""
    import open3d as o3d

    # MapAnything cameras are OpenCV (y down), so the mean camera y axis points roughly down.
    down = np.mean([v["pose"][:3, 1] for v in views], 0)
    down /= np.linalg.norm(down)
    pc = o3d.geometry.PointCloud(mesh.vertices)
    best = None
    for _ in range(6):
        if len(pc.points) < 500:
            break
        (a, b, c, d), idx = pc.segment_plane(0.02, 3, 500)
        n = np.array([a, b, c])
        if abs(n @ down) > 0.9:
            up = -n if n @ down > 0 else n
            best = up
            break
        pc = pc.select_by_index(idx, invert=True)
    up = best if best is not None else -down
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(up, y), float(up @ y)
    kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kx + kx @ kx / (1 + c)


def square_up(mesh) -> np.ndarray:
    """Yaw putting the walls on the image axes, like a plan: the biggest vertical plane
    (normals vote badly here, the bed and clutter outnumber the wall)."""
    import open3d as o3d

    pc = o3d.geometry.PointCloud(mesh.vertices)
    yaw = 0.0
    for _ in range(8):
        if len(pc.points) < 500:
            break
        (a, b, c, _d), idx = pc.segment_plane(0.02, 3, 1000)
        if abs(b) < 0.1:  # vertical: a wall
            yaw = math.atan2(c, a)
            break
        pc = pc.select_by_index(idx, invert=True)
    c, s_ = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def render(mesh, view, size=(1400, 1400)) -> np.ndarray:
    """GPU render of the mesh: 'top' looks straight down (orthographic), a number is the yaw of
    a corner view 40 degrees down."""
    import open3d as o3d
    from open3d.visualization import rendering

    r = rendering.OffscreenRenderer(*size)
    r.scene.set_background([0.96, 0.96, 0.96, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.add_geometry("room", mesh, mat)
    r.scene.scene.set_sun_light([-0.3, -1.0, -0.2], [1.0, 1.0, 1.0], 60000)
    r.scene.scene.enable_sun_light(True)
    box = mesh.get_axis_aligned_bounding_box()
    c, ext = box.get_center(), box.get_extent()
    if view == "top":
        half = 0.52 * max(ext[0], ext[2])
        aspect = size[0] / size[1]
        r.scene.camera.set_projection(
            rendering.Camera.Projection.Ortho, -half * aspect, half * aspect, -half, half, 0.1, 50
        )
        r.scene.camera.look_at(c, c + [0, 10, 0], [0, 0, -1])
    else:
        yaw, dist = math.radians(view), 1.15 * max(ext[0], ext[2])
        eye = c + dist * np.array([math.cos(yaw), 0.85, math.sin(yaw)])
        r.setup_camera(55, c, eye, [0, 1, 0])
    img = np.asarray(r.render_to_image())
    del r
    o3d.utility.reset_print_function()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk", nargs="+", help="one or more walks of the same room")
    ap.add_argument("--views", type=int, default=96)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--hfov", type=float, default=None, help="glasses lens; default camera.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse", action="store_true", help="mesh the views.npz already in --out")
    a = ap.parse_args()
    walks, out = [Path(w) for w in a.walk], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hfov = {w: a.hfov or json.loads((w / "camera.json").read_text()).get("hfov") for w in walks}

    t0 = time.time()
    keys = ("pose", "depth", "mask", "k", "rgb")
    if a.reuse:
        z = np.load(out / "views.npz")
        views = [{k: z[k][i] for k in keys} for i in range(len(z["pose"]))]
    else:
        paths = pool_frames(walks, a.views)
        print(f"{len(paths)} views picked", flush=True)
        views = reconstruct(paths, hfov)
        np.savez_compressed(out / "views.npz", **{k: np.stack([v[k] for v in views]) for k in keys})
    t1 = time.time()
    print(f"reconstructed in {t1 - t0:.0f}s", flush=True)

    import open3d as o3d

    mesh = fuse(views, a.voxel)
    rot = level(mesh, views)
    mesh.rotate(rot, center=(0, 0, 0))
    floor = np.percentile(np.asarray(mesh.vertices)[:, 1], 2)
    mesh.translate((0, -floor, 0))
    mesh.compute_vertex_normals()
    mesh.rotate(square_up(mesh), center=(0, 0, 0))
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
    xyz = np.asarray(mesh.vertices)
    height = np.percentile(xyz[:, 1], 98)
    roofless = o3d.geometry.TriangleMesh(mesh)  # lift the roof off
    roofless.remove_vertices_by_mask(xyz[:, 1] > min(height - 0.15, 2.0))
    cv2.imwrite(str(out / "top.png"), render(roofless, "top"))
    tiles = [render(roofless, yaw, (900, 700)) for yaw in (45, 135, 225, 315)]
    cv2.imwrite(str(out / "orbit.png"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    stats = dict(
        views=len(views),
        recon_s=round(t1 - t0),
        triangles=len(mesh.triangles),
        extent_m=np.ptp(xyz, 0).round(2).tolist(),
        room_height_m=round(float(height), 2),
    )
    (out / "stats.json").write_text(json.dumps(stats))
    print(stats)


if __name__ == "__main__":
    main()
