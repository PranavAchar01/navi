"""MapAnything (Meta, Apache-2.0 checkpoint) speed / memory / point-cloud benchmark on MPS.

Usage: python bench/recon_mapanything.py
Runs a grid of (resolution, window size) and one full-sequence map for the screenshot.
"""

import sys

import numpy as np
import torch
from common import SHOTS, frame_paths, render_cloud, save_result, timed
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

DEV = torch.device("mps")
model = MapAnything.from_pretrained("facebook/map-anything-apache").to(DEV).eval()


def views_for(paths, size):
    if size == 518:
        return load_images(paths)  # default fixed_mapping: 16:9 -> 518x294
    return load_images(paths, resize_mode="longest_side", size=size)


def run(views):
    with torch.no_grad():
        return model.infer(
            views,
            memory_efficient_inference=False,
            use_amp=True,
            amp_dtype="fp16",
            apply_mask=True,
            mask_edges=True,
        )


grid = [
    (1274, 2),
    (1274, 4),
    (518, 2),
    (518, 4),
    (518, 8),
    (518, 16),
    (518, 32),
    (336, 4),
    (336, 8),
    (336, 16),
    (252, 8),
]
if len(sys.argv) > 1 and sys.argv[1] == "--quick":
    grid = grid[2:4]

for size, n in grid:
    paths = frame_paths(fps=2)[:n]
    views = views_for(paths, size)
    h, w = views[0]["img"].shape[-2:]
    try:
        med, times, _, mem = timed(lambda v=views: run(v), warmup=1, reps=3)
    except RuntimeError as e:  # MPS OOM or unsupported op
        save_result(
            f"mapanything_{size}_{n}",
            {"model": "MapAnything", "size": [w, h], "views": n, "error": str(e)[:300]},
        )
        continue
    save_result(
        f"mapanything_{size}_{n}",
        {
            "model": "MapAnything-apache",
            "size": [int(w), int(h)],
            "views": n,
            "latency_s": round(med, 3),
            "views_per_s": round(n / med, 2),
            "peak_mps_gb": round(mem.peak_mps / 2**30, 2),
            "peak_rss_gb": round(mem.peak_rss / 2**30, 2),
            "times": [round(t, 3) for t in times],
        },
    )
    torch.mps.empty_cache()

# Full map: 60 keyframes (2 fps over 30 s) at default 518 for the screenshot.
paths = frame_paths(fps=2)
views = load_images(paths)
import time

t0 = time.perf_counter()
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
torch.mps.synchronize()
full_s = time.perf_counter() - t0
pts, cols, cams, depths = [], [], [], []
for p in preds:
    m = p["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
    pts.append(p["pts3d"][0].float().cpu().numpy()[m])
    cols.append(p["img_no_norm"][0].float().cpu().numpy()[m])
    cams.append(p["camera_poses"][0, :3, 3].float().cpu().numpy())
    depths.append(float(np.median(p["depth_z"][0].float().cpu().numpy())))
pts, cols, cams = np.concatenate(pts), np.concatenate(cols), np.stack(cams)
np.savez_compressed(
    SHOTS.parent / "bench" / "results" / "mapanything_full60.npz",
    pts=pts[::4],
    cols=cols[::4],
    cams=cams,
)
extent = np.percentile(pts, 98, 0) - np.percentile(pts, 2, 0)
path_len = float(np.linalg.norm(np.diff(cams, axis=0), axis=1).sum())
save_result(
    "mapanything_full60",
    {
        "model": "MapAnything-apache",
        "views": len(paths),
        "seconds": round(full_s, 2),
        "views_per_s": round(len(paths) / full_s, 2),
        "points": len(pts),
        "scene_extent_m_xyz": [round(float(x), 2) for x in extent],
        "camera_path_m": round(path_len, 2),
        "median_depth_m": round(float(np.median(depths)), 2),
    },
)
render_cloud(
    pts,
    cols,
    f"MapAnything (Apache), 60 keyframes @518x294, {full_s:.1f}s on M5 Pro MPS",
    SHOTS / "mapanything_full60.png",
    cams,
)
