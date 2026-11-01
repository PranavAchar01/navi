"""Score a Brush run on its held-out frames: PSNR, SSIM and LPIPS of each eval render against the
dataset image it was held out from (Brush names a render after the frame's index). With --masks,
only the room counts: pixels a mask blacks out (hands, laptop, people) are left out of all three.

    python scripts/splat_score.py RUN/eval_30000 data/rooms/bedroom/splat \
        [--masks DIR] [--still out.jpg]

Every experiment in docs/render-experiments.md is scored this way, on the same frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity

STILLS = ("0000", "0100", "0200")  # the frames the side-by-side shows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("evals")
    ap.add_argument("dataset")
    ap.add_argument("--masks")
    ap.add_argument("--still")
    a = ap.parse_args()
    net = lpips.LPIPS(net="alex", spatial=True, verbose=False)
    rows, tiles = [], []
    for render in sorted(Path(a.evals).glob("*.png")):
        out = cv2.imread(str(render))
        gt = cv2.imread(str(Path(a.dataset) / "images" / f"{render.stem}.jpg"))
        if gt.shape != out.shape:
            gt = cv2.resize(gt, out.shape[1::-1], interpolation=cv2.INTER_AREA)
        keep = np.ones(out.shape[:2], bool)
        if a.masks:
            m = cv2.imread(str(Path(a.masks) / f"{render.stem}.png"), 0)
            keep = cv2.resize(m, out.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 127
        mse = np.mean(((out.astype(np.float64) - gt) ** 2)[keep])
        _, smap = structural_similarity(out, gt, channel_axis=2, full=True)
        ssim = smap.mean(axis=2)[keep].mean()
        t = [
            torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
            for x in (out, gt)
        ]
        with torch.no_grad():
            lp = float(net(*t)[0, 0].numpy()[keep].mean())
        rows.append((10 * np.log10(255**2 / mse), ssim, lp))
        if render.stem in STILLS:
            tiles.append(np.hstack([gt, out]))
    psnr, ssim, lp = np.mean(rows, axis=0)
    print(f"{len(rows)} frames  PSNR {psnr:.2f}  SSIM {ssim:.3f}  LPIPS {lp:.3f}")
    if a.still and tiles:
        h = min(t.shape[0] for t in tiles)
        cv2.imwrite(
            a.still, np.hstack([cv2.resize(t, (t.shape[1] * h // t.shape[0], h)) for t in tiles])
        )


if __name__ == "__main__":
    main()
