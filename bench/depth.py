"""Monocular depth for the fast obstacle path: Depth Anything V2 (Small) and Apple Depth Pro on MPS.

Per-frame latency at several input sizes, peak memory, and a side-by-side depth screenshot.
Usage: python bench/depth.py da2|depthpro
"""

import sys

import numpy as np
import torch
from common import SHOTS, frame_paths, save_result, timed
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEV = torch.device("mps")
WHICH = sys.argv[1]
REPO = {
    "da2": "depth-anything/Depth-Anything-V2-Small-hf",
    "depthpro": "apple/DepthPro-hf",
    "da2metric": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
}[WHICH]

if WHICH == "depthpro":
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

    proc = DepthProImageProcessorFast.from_pretrained(REPO)
    model = (
        DepthProForDepthEstimation.from_pretrained(REPO, torch_dtype=torch.float16)
        .to(DEV)
        .eval()
    )
else:
    proc = AutoImageProcessor.from_pretrained(REPO)
    model = (
        AutoModelForDepthEstimation.from_pretrained(REPO, torch_dtype=torch.float16)
        .to(DEV)
        .eval()
    )

frame = Image.open(frame_paths(fps=2)[20]).convert(
    "RGB"
)  # 720p frame, t=10 s (kitchen)


def tensor_at(h, w):
    """Model-ready normalised tensor at an explicit (h, w), skipping the processor's own resize."""
    arr = np.asarray(frame.resize((w, h), Image.BICUBIC)).astype(np.float32) / 255
    mean = np.array(proc.image_mean, dtype=np.float32)
    std = np.array(proc.image_std, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(DEV, torch.float16)


def run(x):
    with torch.no_grad():
        return model(pixel_values=x).predicted_depth


sizes = {
    "da2": [(714, 1274), (518, 924), (294, 518), (196, 350)],
    "depthpro": [(1536, 1536)],
    "da2metric": [(518, 924), (294, 518), (196, 350)],
}[WHICH]
last = None
for h, w in sizes:
    x = tensor_at(h, w)
    med, times, out, mem = timed(
        lambda x=x: run(x), warmup=2, reps=3 if WHICH == "depthpro" else 10
    )
    save_result(
        f"depth_{WHICH}_{w}x{h}",
        {
            "model": REPO,
            "input": [w, h],
            "latency_ms": round(1000 * med, 1),
            "fps": round(1 / med, 1),
            "peak_mps_gb": round(mem.peak_mps / 2**30, 2),
            "peak_rss_gb": round(mem.peak_rss / 2**30, 2),
        },
    )
    if (h, w) in [(294, 518), (1536, 1536)]:
        last = out[0].float().cpu().numpy()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), dpi=110)
ax[0].imshow(frame)
ax[0].set_title("input frame (720p)")
ax[0].axis("off")
im = ax[1].imshow(last, cmap="turbo")
ax[1].axis("off")
ax[1].set_title(
    f"{REPO.split('/')[-1]} output ({'metric m' if WHICH != 'da2' else 'relative inverse depth'})"
)
fig.colorbar(im, ax=ax[1], fraction=0.03)
fig.tight_layout()
fig.savefig(SHOTS / f"depth_{WHICH}.png")
plt.close(fig)
