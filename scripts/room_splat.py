"""The room as Gaussian splats: the dense run's cameras and frames as a nerfstudio dataset, then
Brush (github.com/ArthurBrussee/brush, runs on the Mac's GPU) trains on it.

The cameras are the ones the dense run fused (cams.jsonl), so no COLMAP pass: each pose is
already in the room frame and the mesh's vertices seed the splats. The splat lands in the same
frame as room.ply and the saved objects. The walks wander into the bathroom, the stairs and the
other rooms, and the dense run registered some of those on top of the bedroom, so a frame is only
kept when the room matcher places it against the home's saved views, as the live app would.

    python scripts/room_splat.py --mesh data/rooms/bedroom/seed --tag s1 \
        --out data/rooms/bedroom/splat
    ~/helloworld/tools/brush/brush_app data/rooms/bedroom/splat --total-steps 30000 --sh-degree 1 \
        --max-splats 500000 --export-every 30000 --export-path data/rooms/bedroom/home \
        --export-name splat.ply

The server hands home/splat.ply to the browser at /api/splat.ply (web/splat.html shows it).

Only frames whose camera can be trusted are used: the dense run's trusted frames (camera within
15 cm of both time neighbours, recon_v2.load_cams), or with --cams a recon_reloc.py cams file,
whose rows are trusted dense frames or frames re-placed by PnP. Frames the dense run threw across
the room are what blurred the early splats.

Held-out evaluation: every --eval-holdout'th frame (Brush's --eval-split-every slots, 0, 20, 40...)
is a frame the live splat (x7, trained on --baseline-sources) never saw either, so the new splat
and the live one can be scored on the same unseen cameras (scripts/splat_eval.py).
heldout.txt lists them.

    python scripts/room_splat.py --mesh data/rooms/bedroom/v003 --tag s1 \
        --cams data/rooms/bedroom/recon-v2/v2.1/cams.jsonl \
        --out data/rooms/bedroom/recon-v2/v2.1/splat-B/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))
from recon_v2 import load_cams  # noqa: E402
from room_mesh import H, W  # noqa: E402  the size the dense run's K is for

from metasight import home as homemod  # noqa: E402
from metasight.relocalize import RoomMatcher  # noqa: E402

GL = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes to nerfstudio's OpenGL ones
SEED_POINTS = 200_000
MIN_SHARP = 10.0  # Laplacian variance; the median frame is near 70, a smeared turn near 2
MIN_INLIERS = 220  # server.ROOM_VIEW_MIN_INLIERS


def sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def source(r: dict) -> str:
    return f"{r['walk']}/{r['file']}"


def holdout_order(picked: list, every: int, baseline: Path) -> list:
    """Reorder so that slots 0, every, 2 * every... hold frames spread evenly over the run that the
    live splat did not train on either (its own eval split, every 20th, it never saw)."""
    seen = baseline.read_text().split() if baseline.exists() else []
    seen = set(seen) - set(seen[::20])
    free = [i for i, (r, _) in enumerate(picked) if source(r) not in seen]
    n = -(-len(picked) // every)  # ceil: slots 0, every, ... below len(picked)
    if len(free) < n:
        raise SystemExit(f"only {len(free)} frames the live splat never trained on, need {n}")
    held = {free[round(j)] for j in np.linspace(0, len(free) - 1, n)}
    todo = [picked[i] for i in sorted(held)]
    rest = [p for i, p in enumerate(picked) if i not in held]
    order = []
    while todo or rest:
        order.append(todo.pop(0) if len(order) % every == 0 else rest.pop(0))
    return order


def write_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Float positions and byte colours: Brush's reader stops early on Open3D's doubles."""
    row = np.dtype([*((c, "<f4") for c in "xyz"), *((c, "u1") for c in "rgb")])
    v = np.empty(len(xyz), row)
    v["x"], v["y"], v["z"] = xyz.T
    v["r"], v["g"], v["b"] = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8).T
    head = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
        + "".join(f"property float {c}\n" for c in "xyz")
        + "".join(f"property uchar {c}\n" for c in ("red", "green", "blue"))
        + "end_header\n"
    )
    path.write_bytes(head.encode() + v.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="room_dense.py output dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--home", default=str(ROOT / "data/rooms/bedroom/home"))
    ap.add_argument("--frames", type=int, default=1600)
    ap.add_argument("--long-side", type=int, default=960)
    ap.add_argument(
        "--per-walk",
        action="store_true",
        help="spread --frames evenly over the walks instead of over all fused frames",
    )
    ap.add_argument(
        "--hold-out",
        help="sources.txt of an earlier dataset: its every-20th frames (Brush's eval split) stay "
        "at the same slots here, so both runs are scored on the same held-out frames",
    )
    ap.add_argument(
        "--min-inliers",
        type=int,
        default=MIN_INLIERS,
        help="keep a frame only if the room matcher places it against the home's saved views "
        "with this many inliers; 0 skips the matcher (the saved views miss parts of the room, "
        "e.g. every frame of the 11-02 walk, so with trusted or PnP poses it is optional)",
    )
    ap.add_argument("--cams", help="recon_reloc.py cams file (mesh frame, dense and pnp rows)")
    ap.add_argument(
        "--eval-holdout",
        type=int,
        default=20,
        help="every Nth frame is held out (pass Brush --eval-split-every N); 0 turns it off",
    )
    ap.add_argument(
        "--baseline-sources",
        default=str(ROOT / "data/rooms/bedroom/splat/sources.txt"),
        help="sources.txt the live splat trained on; none of its training frames is held out",
    )
    a = ap.parse_args()
    if a.hold_out and a.eval_holdout:
        ap.error("--hold-out pins Brush's 20-frame split itself; pass --eval-holdout 0 with it")
    import open3d as o3d

    mdir, out = Path(a.mesh), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    by_walk = load_cams(mdir, a.tag, Path(a.cams) if a.cams else None)  # poses in the mesh frame
    rows = [
        r
        for w in sorted(by_walk)
        for r in by_walk[w]
        if r.get("src") in ("dense", "pnp") or (not a.cams and r["trusted"])
    ]
    if a.min_inliers:
        h = homemod.load(Path(a.home))
        views = [cv2.imdecode(np.frombuffer(v.jpg, np.uint8), 1) for v in h.views]
        m = RoomMatcher(views, h.views)
        mh, mw = m.shape
    elsewhere = 0
    # Evenly through the fused frames, the sharper of each neighbouring pair: walking blurs most.
    starts = range(0, len(rows) - 1, max(1, len(rows) // a.frames))
    if a.per_walk:
        walks = sorted({r["walk"] for r in rows})
        starts = []
        for w in walks:
            idx = [i for i, r in enumerate(rows[:-1]) if r["walk"] == w]
            starts += idx[:: max(1, len(idx) * len(walks) // a.frames)]
    picked = []  # (row, image) in sample order
    for i in starts:
        best = None
        for r in rows[i : i + 2]:
            img = cv2.imread(str(ROOT / "data/walks" / r["walk"] / "frames" / r["file"]))
            if img is not None and (best is None or sharpness(img) > best[0]):
                best = (sharpness(img), r, img)
        if best is None or best[0] < MIN_SHARP:
            continue
        _, r, img = best
        if a.min_inliers:
            hit = m.locate(cv2.resize(img, (mw, mh), interpolation=cv2.INTER_AREA))
            if hit is None or hit[1] < a.min_inliers:
                elsewhere += 1
                continue
        picked.append((r, img))
    if a.hold_out:
        # The earlier run's held-out frames go back into the slots Brush holds out.
        old = Path(a.hold_out).read_text().split()
        held = {s: None for s in old[::20]}
        for r in rows:
            if (s := source(r)) in held:
                held[s] = (r, cv2.imread(str(ROOT / "data/walks" / s.replace("/", "/frames/", 1))))
        rest = [p for p in picked if source(p[0]) not in held]
        picked, todo = [], list(held.values())
        while todo or rest:
            picked.append(todo.pop(0) if todo and len(picked) % 20 == 0 else rest.pop(0))
    if a.eval_holdout:
        picked = holdout_order(picked, a.eval_holdout, Path(a.baseline_sources))
        held = picked[:: a.eval_holdout]
        (out / "heldout.txt").write_text("".join(f"{source(r)}\n" for r, _ in held))
    frames = []
    for r, img in picked:
        k = min(1.0, a.long_side / max(img.shape[:2]))
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
        h, w = img.shape[:2]
        name = f"{len(frames):04d}.jpg"
        cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        kk = np.array(r["k"])
        sx, sy = w / W, h / H  # the dense run saw each frame resized to W x H
        frames.append(
            {
                "file_path": f"images/{name}",
                "transform_matrix": (np.array(r["pose"]) @ GL).tolist(),
                "fl_x": kk[0, 0] * sx,
                "fl_y": kk[1, 1] * sy,
                "cx": kk[0, 2] * sx,
                "cy": kk[1, 2] * sy,
                "w": w,
                "h": h,
            }
        )
    (out / "sources.txt").write_text("".join(f"{source(r)}\n" for r, _ in picked))
    mesh = o3d.io.read_triangle_mesh(str(mdir / f"mesh-{a.tag}.ply"))
    pc = mesh.sample_points_uniformly(SEED_POINTS)  # the mesh is in the room frame already
    write_points(out / "points.ply", np.asarray(pc.points), np.asarray(pc.colors))
    (out / "transforms.json").write_text(
        json.dumps({"camera_model": "OPENCV", "ply_file_path": "points.ply", "frames": frames})
    )
    print(f"{len(frames)} frames ({elsewhere} not in this room), {SEED_POINTS} seeds -> {out}")


if __name__ == "__main__":
    main()
