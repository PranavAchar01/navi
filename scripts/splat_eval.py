"""Score any trained splat on a dataset's held-out frames: render it from those cameras and compare
with the real frames (PSNR, SSIM, LPIPS), the room only when the dataset has masks.

Brush scores its own run on its eval split (--eval-split-every N writes eval_<step>/<index>.png),
and has no render-only mode for a finished splat on another run's cameras. So this file has its
own renderer, the 3D Gaussian splatting forward pass (EWA projection with a 0.3 px low-pass, SH up
to degree 3, front-to-back alpha compositing per 16 px tile) in torch; its renders of a Brush
export match Brush's own eval renders to 41 to 45 dB on a 3000-step splat and about 31 dB on a 50k
step, 500k-splat one, where it scores about 1.3 dB under Brush (x7 rerun on splat-x3: 25.08 here,
26.41 by splat_score.py on Brush's renders). The live splat, Brush's new ones and any other ply
are all scored the same way on the same cameras: compare splats with each other here, not with
Brush's numbers.

    python scripts/splat_eval.py DATASET SPLAT.ply [SPLAT2.ply=TRAIN_DATASET ...] [--every 20] \
        [--still out.jpg]

Frames: when a dataset has a colmap/ model (splat_refine_poses.py writes one), Brush trains on that
model, not on transforms.json, so its splat lives in the BA's own gauge, a sim3 away from the room
frame (colmap/gauge.npz: room = s R x + t, about 6 degrees and 2% for splat-x3). Each splat is
scored in the frame it was trained in: SPLAT.ply=TRAIN_DATASET names the dataset it came from
(default DATASET), and the held-out cameras go room frame -> that dataset's gauge.

The held-out frames are dataset frames 0, every, 2 * every... in image-name order (Brush's split;
room_splat.py puts frames there that no splat under test trained on).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from plyfile import PlyData
from skimage.metrics import structural_similarity

GL = np.diag([1.0, -1.0, -1.0, 1.0])
TILE = 16
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
         0.5462742152960396)  # fmt: skip
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)  # fmt: skip


def device() -> torch.device:
    """CPU: the per-tile loop is launch-bound on MPS, about 5 s a frame here against 20 s there."""
    return torch.device("cpu")


def load_splat(path: Path, dev: torch.device) -> dict[str, torch.Tensor]:
    """A 3DGS ply (Brush's export): log scales, logit opacity, wxyz quaternion, SH channel-major."""
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    col = lambda *ks: np.stack([np.asarray(v[k], np.float32) for k in ks], 1)  # noqa: E731
    n_rest = sum(1 for k in names if k.startswith("f_rest_"))
    sh = col("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :]  # (N, 1, 3)
    if n_rest:
        rest = col(*(f"f_rest_{i}" for i in range(n_rest))).reshape(len(sh), 3, n_rest // 3)
        sh = np.concatenate([sh, rest.transpose(0, 2, 1)], 1)  # (N, K, 3)
    q = col("rot_0", "rot_1", "rot_2", "rot_3")
    t = {
        "xyz": col("x", "y", "z"),
        "scale": np.exp(col("scale_0", "scale_1", "scale_2")),
        "quat": q / np.linalg.norm(q, axis=1, keepdims=True),
        "opacity": 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float32))),
        "sh": sh,
    }
    return {k: torch.from_numpy(np.ascontiguousarray(a)).to(dev) for k, a in t.items()}


def load_gauge(dataset: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """The sim3 from a dataset's COLMAP gauge to the room frame (identity without one)."""
    p = dataset / "colmap" / "gauge.npz"
    if not p.exists():
        return 1.0, np.eye(3), np.zeros(3)
    g = np.load(p)
    return float(g["s"]), g["R"], g["t"]


def load_frames(dataset: Path) -> list[dict]:
    """The cameras Brush trains and evaluates on, sorted by image name, with room-frame poses
    (OpenGL camera-to-world, as transforms.json): the COLMAP model when there is one."""
    tf = json.loads((dataset / "transforms.json").read_text())
    frames = {f["file_path"]: f for f in tf["frames"]}
    if not (dataset / "colmap" / "images.bin").exists():
        return [frames[k] for k in sorted(frames)]
    s, r, t = load_gauge(dataset)
    cams, ims = read_colmap(dataset / "colmap")
    out = []
    for name in sorted(ims):
        cam_id, w2c = ims[name]
        c2w = np.linalg.inv(w2c)
        c2w[:3, :3], c2w[:3, 3] = r @ c2w[:3, :3], s * r @ c2w[:3, 3] + t
        fx, fy, cx, cy = cams[cam_id]
        f = dict(frames[name], fl_x=fx, fl_y=fy, cx=cx, cy=cy)
        f["transform_matrix"] = (c2w @ GL).tolist()
        out.append(f)
    return out


def read_colmap(model: Path) -> tuple[dict, dict]:
    """PINHOLE cameras.bin and images.bin, read directly: pycolmap and torch each bring their own
    OpenMP and abort when loaded together. Returns {camera id: fx fy cx cy} and
    {image name: (camera id, 4x4 world-to-camera)}."""
    cams, ims = {}, {}
    b = (model / "cameras.bin").read_bytes()
    o = 8
    for _ in range(struct.unpack_from("<Q", b)[0]):
        cid, mid = struct.unpack_from("<ii", b, o)
        if mid != 1:
            raise SystemExit(f"camera {cid}: model {mid}, only PINHOLE (1) is read")
        cams[cid] = struct.unpack_from("<4d", b, o + 24)
        o += 24 + 32
    b = (model / "images.bin").read_bytes()
    o = 8
    for _ in range(struct.unpack_from("<Q", b)[0]):
        qw, qx, qy, qz, tx, ty, tz = struct.unpack_from("<7d", b, o + 4)
        (cid,) = struct.unpack_from("<i", b, o + 60)
        end = b.index(b"\0", o + 64)
        name = b[o + 64 : end].decode()
        (npts,) = struct.unpack_from("<Q", b, end + 1)
        o = end + 9 + 24 * npts
        w2c = np.eye(4)
        w2c[:3, :3] = quat_to_rot(torch.tensor([[qw, qx, qy, qz]], dtype=torch.float64))[0].numpy()
        w2c[:3, 3] = tx, ty, tz
        ims[name] = (cid, w2c)
    return cams, ims


def into_gauge(frame: dict, gauge: tuple[float, np.ndarray, np.ndarray]) -> dict:
    """A room-frame camera moved into a splat's training gauge (the inverse of room = s R x + t)."""
    s, r, t = gauge
    c2w = np.array(frame["transform_matrix"], float) @ GL
    c2w[:3, :3], c2w[:3, 3] = r.T @ c2w[:3, :3], r.T @ (c2w[:3, 3] - t) / s
    return dict(frame, transform_matrix=(c2w @ GL).tolist())


def moved(frame: dict, sim: tuple[float, np.ndarray, np.ndarray]) -> dict:
    """A camera carried by x -> s R x + t."""
    s, r, t = sim
    c2w = np.array(frame["transform_matrix"], float) @ GL
    c2w[:3, :3], c2w[:3, 3] = r @ c2w[:3, :3], s * r @ c2w[:3, 3] + t
    return dict(frame, transform_matrix=(c2w @ GL).tolist())


def umeyama(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The sim3 taking points a onto b in the least-squares sense."""
    ma, mb = a.mean(0), b.mean(0)
    u, d, vt = np.linalg.svd((b - mb).T @ (a - ma) / len(a))
    e = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ e @ vt
    s = float((d * e.diagonal()).sum() / ((a - ma) ** 2).sum(1).mean())
    return s, r, mb - s * r @ ma


def sources(dataset: Path, given: Path | None = None) -> list[str]:
    return (given or dataset / "sources.txt").read_text().split()  # line n is images/n.jpg


def align(frames: list, src: list, train: Path, train_src: list) -> tuple:
    """The sim3 from this dataset's room frame to another dataset's, fitted on the source frames
    both have: each camera's centre plus points 0.5 m along its three axes, so orientation counts
    as well as position (centres alone leave a walk's roll free: 5 degrees off between splat-x3
    and splat-B). Frames off by more than 3x the median are dropped and the fit redone. Two dense
    runs (seed and v003) and their BAs each put the room a few cm and degrees apart."""

    def pts(fs: list, ss: list) -> dict:
        out = {}
        for f in fs:
            m = np.array(f["transform_matrix"])
            out[ss[int(Path(f["file_path"]).stem)]] = m[:3, 3] + 0.5 * np.vstack(
                [0 * m[:3, 0], m[:3, :3].T]
            )
        return out

    here, there = pts(frames, src), pts(load_frames(train), train_src)
    keys = sorted(set(here) & set(there))
    if len(keys) < 10:
        raise SystemExit(f"{train}: only {len(keys)} shared frames to align on")
    a, b = np.array([here[k] for k in keys]), np.array([there[k] for k in keys])  # (K, 4, 3)
    ok = np.ones(len(keys), bool)
    for _ in range(2):
        sim = umeyama(a[ok].reshape(-1, 3), b[ok].reshape(-1, 3))
        err = np.linalg.norm(sim[0] * a @ sim[1].T + sim[2] - b, axis=2).mean(1)
        ok = err < 3 * np.median(err)
    print(f"aligned to {train} on {ok.sum()}/{len(keys)} shared frames, "
          f"median {100 * np.median(err):.1f} cm", flush=True)  # fmt: skip
    return sim


def sh_colour(sh: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Evaluate SH (N, K, 3) at unit directions d (N, 3), as 3DGS does (plus 0.5, clamped at 0)."""
    c = SH_C0 * sh[:, 0]
    k = sh.shape[1]
    if k >= 4:
        x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        c = c - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if k >= 9:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            c = (
                c
                + SH_C2[0] * xy * sh[:, 4]
                + SH_C2[1] * yz * sh[:, 5]
                + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6]
                + SH_C2[3] * xz * sh[:, 7]
                + SH_C2[4] * (xx - yy) * sh[:, 8]
            )
            if k >= 16:
                c = (
                    c
                    + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9]
                    + SH_C3[1] * xy * z * sh[:, 10]
                    + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                    + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                    + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
                    + SH_C3[5] * z * (xx - yy) * sh[:, 14]
                    + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15]
                )
    return torch.clamp(c + 0.5, min=0.0)


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        1,
    ).reshape(-1, 3, 3)  # fmt: skip


@torch.no_grad()
def render(g: dict[str, torch.Tensor], frame: dict) -> np.ndarray:
    """BGR uint8 render of the splat from a nerfstudio frame (OpenGL camera-to-world)."""
    dev = g["xyz"].device
    w, h = int(frame["w"]), int(frame["h"])
    fx, fy, cx, cy = (float(frame[k]) for k in ("fl_x", "fl_y", "cx", "cy"))
    c2w = np.array(frame["transform_matrix"], float) @ GL
    w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)
    p = g["xyz"] @ w2c[:3, :3].T + w2c[:3, 3]
    z = p[:, 2]
    keep = z > 0.01
    # loose frustum cull before the covariance work
    lim = 1.3
    keep &= (p[:, 0].abs() < lim * z * (max(cx, w - cx) / fx + 0.5)) & (
        p[:, 1].abs() < lim * z * (max(cy, h - cy) / fy + 0.5)
    )
    idx = keep.nonzero().squeeze(1)
    p, z = p[idx], z[idx]
    r = quat_to_rot(g["quat"][idx])
    m = r * g["scale"][idx][:, None, :]
    cov3 = m @ m.transpose(1, 2)
    tx = torch.clamp(p[:, 0] / z, -lim * (cx / fx + 0.5), lim * ((w - cx) / fx + 0.5)) * z
    ty = torch.clamp(p[:, 1] / z, -lim * (cy / fy + 0.5), lim * ((h - cy) / fy + 0.5)) * z
    jac = torch.zeros(len(p), 2, 3, device=dev)
    jac[:, 0, 0], jac[:, 0, 2] = fx / z, -fx * tx / (z * z)
    jac[:, 1, 1], jac[:, 1, 2] = fy / z, -fy * ty / (z * z)
    t = jac @ w2c[:3, :3]
    cov2 = t @ cov3 @ t.transpose(1, 2)
    a, b, c = cov2[:, 0, 0] + 0.3, cov2[:, 0, 1], cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    ok = det > 0
    inv = torch.stack([c / det, -b / det, a / det], 1)
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    rad = torch.ceil(3 * torch.sqrt(lam))
    u = fx * p[:, 0] / z + cx
    v = fy * p[:, 1] / z + cy
    ok &= (u + rad > 0) & (u - rad < w) & (v + rad > 0) & (v - rad < h)
    sel = ok.nonzero().squeeze(1)
    u, v, rad, inv, z = u[sel], v[sel], rad[sel], inv[sel], z[sel]
    gi = idx[sel]
    cam = torch.tensor(c2w[:3, 3], dtype=torch.float32, device=dev)
    d = g["xyz"][gi] - cam
    col = sh_colour(g["sh"][gi], d / d.norm(dim=1, keepdim=True))
    op = g["opacity"][gi]
    order = torch.argsort(z)
    u, v, rad, inv, col, op = (x[order] for x in (u, v, rad, inv, col, op))
    out = torch.zeros(h, w, 3, device=dev)
    tx0 = torch.clamp(((u - rad) / TILE).floor(), min=0).long()
    tx1 = torch.clamp(((u + rad) / TILE).floor(), max=math.ceil(w / TILE) - 1).long()
    ty0 = torch.clamp(((v - rad) / TILE).floor(), min=0).long()
    ty1 = torch.clamp(((v + rad) / TILE).floor(), max=math.ceil(h / TILE) - 1).long()
    ys, xs = torch.meshgrid(
        torch.arange(TILE, device=dev, dtype=torch.float32) + 0.5,
        torch.arange(TILE, device=dev, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    # every (tile, splat) overlap, sorted by tile and then depth (the splats are in depth order)
    nx = math.ceil(w / TILE)
    cw, ch = tx1 - tx0 + 1, ty1 - ty0 + 1
    gid = torch.repeat_interleave(torch.arange(len(u), device=dev), cw * ch)
    off = torch.arange(len(gid), device=dev) - torch.repeat_interleave(
        torch.cumsum(cw * ch, 0) - cw * ch, cw * ch
    )
    tile = (ty0[gid] + off // cw[gid]) * nx + tx0[gid] + off % cw[gid]
    order = torch.argsort(tile * len(u) + gid)
    tile, gid = tile[order], gid[order]
    ends = torch.cumsum(torch.bincount(tile, minlength=nx * math.ceil(h / TILE)), 0).tolist()
    for t_id, end in enumerate(ends):
        start = ends[t_id - 1] if t_id else 0
        ty, tx_ = divmod(t_id, nx)
        k = gid[start:end]
        if len(k) == 0:
            continue
        px = (xs + tx_ * TILE).reshape(-1)
        py = (ys + ty * TILE).reshape(-1)
        dx = px[None] - u[k][:, None]
        dy = py[None] - v[k][:, None]
        ik = inv[k]
        power = -0.5 * (ik[:, 0:1] * dx * dx + ik[:, 2:3] * dy * dy) - ik[:, 1:2] * dx * dy
        alpha = torch.clamp(op[k][:, None] * torch.exp(torch.clamp(power, max=0)), max=0.99)
        alpha = torch.where(alpha < 1 / 255, torch.zeros_like(alpha), alpha)
        trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1 - alpha[:-1]]), 0)
        wgt = alpha * trans * (trans > 1e-4)
        pix = (wgt.T @ col[k]).reshape(TILE, TILE, 3)
        y1, x1 = min(TILE, h - ty * TILE), min(TILE, w - tx_ * TILE)
        out[ty * TILE : ty * TILE + y1, tx_ * TILE : tx_ * TILE + x1] = pix[:y1, :x1]
    img = (torch.clamp(out, 0, 1) * 255).round().byte().cpu().numpy()
    return img[..., ::-1].copy()


def score(out: np.ndarray, gt: np.ndarray, keep: np.ndarray, net) -> tuple[float, float, float]:
    mse = np.mean(((out.astype(np.float64) - gt) ** 2)[keep])
    _, smap = structural_similarity(out, gt, channel_axis=2, full=True)
    t = [
        torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
        for x in (out, gt)
    ]
    with torch.no_grad():
        lp = float(net(*t)[0, 0].numpy()[keep].mean())
    return float(10 * np.log10(255**2 / mse)), float(smap.mean(axis=2)[keep].mean()), lp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("splats", nargs="+", help="SPLAT.ply, or SPLAT.ply=TRAIN_DATASET")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--renders", type=Path, help="write each render as <dir>/<splat stem>/<i>.png")
    ap.add_argument("--still", type=Path, help="ground truth and each splat for 3 held-out frames")
    ap.add_argument("--json", type=Path, help="write the scores here")
    ap.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="sources.txt of another splat's dataset: held-out frames it trained on (all but its "
        "own every-Nth) are dropped, so every splat is scored on frames none of them saw",
    )
    ap.add_argument(
        "--sources",
        type=Path,
        help="DATASET's sources.txt when it has none (splat-x3 is splat/ after BA)",
    )
    a = ap.parse_args()
    import lpips

    net = lpips.LPIPS(net="alex", spatial=True, verbose=False)
    frames = load_frames(a.dataset)
    held = list(range(0, len(frames), a.every))
    if a.exclude:
        src = sources(a.dataset, a.sources)
        seen = set()
        for p in a.exclude:
            s = p.read_text().split()
            seen |= set(s) - set(s[:: a.every])
        held = [i for i in held if src[int(Path(frames[i]["file_path"]).stem)] not in seen]
    dev = device()
    gts, keeps = {}, {}
    for i in held:
        f = frames[i]
        gts[i] = cv2.imread(str(a.dataset / f["file_path"]))
        keeps[i] = np.ones(gts[i].shape[:2], bool)
        if "mask_path" in f:
            keeps[i] = cv2.imread(str(a.dataset / f["mask_path"]), 0) > 127
    results, tiles = {}, {i: [gts[i]] for i in held[:: max(1, len(held) // 3)][:3]}
    for spec in a.splats:
        sp, _, trained = spec.partition("=")
        sp = Path(sp)
        train, _, train_src = trained.partition(",")
        train = Path(train) if train else a.dataset
        sim = (1.0, np.eye(3), np.zeros(3))
        if train.resolve() != a.dataset.resolve():
            sim = align(
                frames,
                sources(a.dataset, a.sources),
                train,
                sources(train, Path(train_src) if train_src else None),
            )
        gauge = load_gauge(train)
        t0 = time.time()
        g = load_splat(sp, dev)
        rows = []
        for i in held:
            img = render(g, into_gauge(moved(frames[i], sim), gauge))
            if a.renders:
                (a.renders / sp.stem).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(a.renders / sp.stem / f"{i:04d}.png"), img)
            rows.append(score(img, gts[i], keeps[i], net))
            if i in tiles:
                tiles[i].append(img)
        psnr, ssim, lp = np.mean(rows, axis=0)
        results[str(sp)] = {
            "frames": len(rows), "psnr": psnr, "ssim": ssim, "lpips": lp,
            "per_frame": rows,
        }  # fmt: skip
        print(
            f"{sp}: {len(rows)} frames  PSNR {psnr:.2f}  SSIM {ssim:.3f}  LPIPS {lp:.3f}"
            f"  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))
    if a.still:
        cv2.imwrite(str(a.still), np.vstack([np.hstack(t) for t in tiles.values()]))


if __name__ == "__main__":
    main()
