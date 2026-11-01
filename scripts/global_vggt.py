"""Build a walk's map in one global solve with VGGT-1B instead of stitching short windows.

python scripts/global_vggt.py data/walk.mp4 --out sessions/walk-vggt.npz [--views 96]

The live engine stitches windows of a few keyframes, so turning error adds up and a room seen
twice lands twice. Offline nobody waits: VGGT takes every chosen keyframe at once and places
them all jointly. Keyframes are the sharpest frame of each equal slice of the walk. VGGT's
units are arbitrary, so the map is scaled by MapAnything's metric depth on a few of the same
frames (median depth ratio). Level it afterwards with level_map.py.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metasight.align import map_resolution  # noqa: E402
from metasight.memory import MapAnythingRunner  # noqa: E402

KEEP_CONF_PCT = 40  # drop the least confident share of depth pixels
SCALE_VIEWS = 8  # frames MapAnything also sees, for the metric scale


def keyframes(video: str, n: int) -> list[np.ndarray]:
    """The sharpest frame (Laplacian variance) of each of n equal slices of the video."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    edges = np.linspace(0, total, n + 1).astype(int)
    best: list[tuple[float, np.ndarray | None]] = [(-1.0, None)] * n
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        b = min(int(np.searchsorted(edges, i, side="right")) - 1, n - 1)
        sharp = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if sharp > best[b][0]:
            best[b] = (sharp, frame)
        i += 1
    return [f for _, f in best if f is not None]


def to_tensor(frames: list[np.ndarray], long_side: int = 518) -> torch.Tensor:
    h, w = frames[0].shape[:2]
    s = long_side / max(h, w)
    size = (round(w * s / 14) * 14, round(h * s / 14) * 14)
    ims = [cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
           for f in frames]  # fmt: skip
    return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().div(255)


def metric_scale(frames: list[np.ndarray], depth: np.ndarray, conf: np.ndarray) -> float:
    """Median of MapAnything's metric depth over VGGT's depth, on SCALE_VIEWS spread frames."""
    pick = np.linspace(0, len(frames) - 1, SCALE_VIEWS).astype(int)
    h, w = frames[0].shape[:2]
    mw, mh = map_resolution(w, h)
    preds = MapAnythingRunner().infer(
        [cv2.resize(frames[i], (mw, mh), interpolation=cv2.INTER_AREA) for i in pick]
    )
    ratios = []
    for i, pr in zip(pick, preds, strict=True):
        dz = pr["depth_z"]
        dv = cv2.resize(depth[i][..., 0], dz.shape[::-1], interpolation=cv2.INTER_NEAREST)
        cv = cv2.resize(conf[i], dz.shape[::-1], interpolation=cv2.INTER_NEAREST)
        ok = pr["mask"] & (dv > 1e-3) & (cv > np.percentile(conf, KEEP_CONF_PCT))
        ratios.append(dz[ok] / dv[ok])
    return float(np.median(np.concatenate(ratios)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=96)
    a = ap.parse_args()
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dev = torch.device("mps")
    frames = keyframes(a.video, a.views)
    imgs = to_tensor(frames).to(dev)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    with torch.no_grad(), torch.autocast("mps", dtype=torch.float16):
        pred = model(imgs)
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
    ex, inn = extri[0].float().cpu().numpy(), intri[0].float().cpu().numpy()
    depth = pred["depth"][0].float().cpu().numpy()
    conf = pred["depth_conf"][0].float().cpu().numpy()
    pts = unproject_depth_map_to_point_map(depth, ex, inn)
    keep = conf > np.percentile(conf, KEEP_CONF_PCT)
    cols = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    pts, cols = pts[keep].astype(np.float32), cols[keep]
    cams = np.stack([-e[:, :3].T @ e[:, 3] for e in ex])  # camera centres, world = first camera

    k = metric_scale(frames, depth, conf)
    pts, cams = pts * k, cams * k

    trail = np.concatenate([cams, np.arange(len(cams))[:, None]], 1)
    np.savez_compressed(a.out, pts=pts, cols=cols, trail=trail, pose=np.eye(4))
    print(f"{len(cams)} views, {len(pts)} points, scale x{k:.2f}")


if __name__ == "__main__":
    main()
