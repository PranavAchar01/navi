"""VGGT-1B and StreamVGGT speed / memory / point-cloud benchmark on MPS.

Usage: python bench/recon_vggt.py vggt|stream
"""

import sys
import time

import numpy as np
import torch
from common import ROOT, SHOTS, frame_paths, render_cloud, save_result, timed
from PIL import Image

DEV = torch.device("mps")
WHICH = sys.argv[1]


def load(paths, width):
    """Resize 16:9 frames to `width` x (multiple of 14), [0,1] tensor (N,3,H,W)."""
    h = round(width * 9 / 16 / 14) * 14
    ims = [
        np.asarray(Image.open(p).convert("RGB").resize((width, h), Image.BICUBIC))
        for p in paths
    ]
    return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().div(255).to(DEV)


if WHICH == "vggt":
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEV).eval()

    def run(imgs):
        with torch.no_grad(), torch.autocast("mps", dtype=torch.float16):
            return model(imgs)
else:
    sys.path.insert(0, str(ROOT / "third_party" / "StreamVGGT" / "src"))
    from huggingface_hub import hf_hub_download
    from streamvggt.models.streamvggt import StreamVGGT

    model = StreamVGGT()
    ck = torch.load(
        hf_hub_download("lch01/StreamVGGT", "checkpoints.pth"), map_location="cpu"
    )
    model.load_state_dict(ck, strict=True)
    model = model.to(DEV).eval()

    def run(imgs):
        """Causal streaming: one frame at a time with cached past tokens (the real-time mode)."""
        frames = [{"img": imgs[i : i + 1]} for i in range(imgs.shape[0])]
        with torch.no_grad(), torch.autocast("mps", dtype=torch.float16):
            return model.inference(frames)


tag = "VGGT-1B" if WHICH == "vggt" else "StreamVGGT"
grid = [
    (1274, 2),
    (518, 2),
    (518, 4),
    (518, 8),
    (518, 16),
    (518, 32),
    (336, 8),
    (336, 16),
    (252, 8),
]
if WHICH == "stream":
    grid = [(1274, 4), (518, 8), (518, 32), (518, 60), (336, 8), (336, 32), (252, 32)]
for width, n in grid:
    imgs = load(frame_paths(fps=2)[:n], width)
    try:
        med, times, _, mem = timed(lambda x=imgs: run(x), warmup=1, reps=3)
    except RuntimeError as e:
        save_result(
            f"{WHICH}_{width}_{n}",
            {
                "model": tag,
                "size": list(imgs.shape[-1:-3:-1]),
                "views": n,
                "error": str(e)[:300],
            },
        )
        continue
    save_result(
        f"{WHICH}_{width}_{n}",
        {
            "model": tag,
            "size": [imgs.shape[-1], imgs.shape[-2]],
            "views": n,
            "latency_s": round(med, 3),
            "views_per_s": round(n / med, 2),
            "per_frame_ms": round(1000 * med / n, 1),
            "peak_mps_gb": round(mem.peak_mps / 2**30, 2),
            "peak_rss_gb": round(mem.peak_rss / 2**30, 2),
            "times": [round(t, 3) for t in times],
        },
    )
    torch.mps.empty_cache()

if WHICH == "vggt":
    # Full 60-keyframe map for the screenshot (VGGT runs all views jointly).
    paths = frame_paths(fps=2)
    imgs = load(paths, 518)
    t0 = time.perf_counter()
    pred = run(imgs)
    torch.mps.synchronize()
    full_s = time.perf_counter() - t0
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
    depth = pred["depth"][0].float().cpu().numpy()
    conf = pred["depth_conf"][0].float().cpu().numpy()
    pts = unproject_depth_map_to_point_map(
        depth, extri[0].float().cpu().numpy(), intri[0].float().cpu().numpy()
    )
    cols = imgs.permute(0, 2, 3, 1).cpu().numpy()
    keep = conf > np.percentile(conf, 30)
    ex = extri[0].float().cpu().numpy()  # world2cam 3x4
    cams = np.stack([-e[:, :3].T @ e[:, 3] for e in ex])
    np.savez_compressed(
        ROOT / "bench" / "results" / "vggt_full60.npz",
        pts=pts[keep][::4],
        cols=cols[keep][::4],
        cams=cams,
    )
    save_result(
        "vggt_full60",
        {
            "model": tag,
            "views": len(paths),
            "seconds": round(full_s, 2),
            "views_per_s": round(len(paths) / full_s, 2),
            "points": int(keep.sum()),
        },
    )
    render_cloud(
        pts[keep],
        cols[keep],
        f"VGGT-1B, 60 keyframes @518x294, {full_s:.1f}s on M5 Pro MPS (up to scale)",
        SHOTS / "vggt_full60.png",
        cams,
    )
