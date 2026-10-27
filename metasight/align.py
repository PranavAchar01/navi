"""Stitching windows into one map: robust similarity alignment and voxel bookkeeping."""

from __future__ import annotations

import numpy as np

# MapAnything's 518-wide fixed resolutions, keyed by aspect ratio (width / height).
RESOLUTIONS = {
    1.000: (518, 518),
    1.321: (518, 392),
    1.542: (518, 336),
    1.762: (518, 294),
    2.056: (518, 252),
    0.757: (392, 518),
    0.649: (336, 518),
    0.567: (294, 518),
    0.486: (252, 518),
}


def map_resolution(w: int, h: int) -> tuple[int, int]:
    """(width, height) the models run at for a frame of this aspect ratio. Multiples of 14."""
    ar = w / h
    key = min(RESOLUTIONS, key=lambda k: abs(k - ar))
    return RESOLUTIONS[key]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform with dst ~ s * R @ src + t. Returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rot = u @ sign @ vt
    var = (xs**2).sum() / len(src)
    s = float((d * np.diag(sign)).sum() / var) if with_scale and var > 1e-12 else 1.0
    t = mu_d - s * rot @ mu_s
    return s, rot, t


def robust_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    iters: int = 3,
    keep: float = 0.8,
    scale_bounds: tuple[float, float] = (0.8, 1.25),
):
    """Umeyama with iterative trimming of the worst residuals. None if the fit is implausible."""
    if len(src) < 50:
        return None
    idx = np.arange(len(src))
    s, rot, t = umeyama(src, dst)
    for _ in range(iters):
        res = np.linalg.norm(dst - (s * src @ rot.T + t), axis=1)
        idx = np.argsort(res)[: max(50, int(keep * len(src)))]
        s, rot, t = umeyama(src[idx], dst[idx])
    if not scale_bounds[0] <= s <= scale_bounds[1]:
        s, rot, t = umeyama(src[idx], dst[idx], with_scale=False)
    res = np.linalg.norm(dst[idx] - (s * src[idx] @ rot.T + t), axis=1)
    return s, rot, t, float(np.median(res))


def apply_sim3(points: np.ndarray, s: float, rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * points @ rot.T + t


def pose_through_sim3(pose: np.ndarray, s: float, rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot @ pose[:3, :3]
    out[:3, 3] = s * rot @ pose[:3, 3] + t
    return out


_OFF, _M = 4096, 8192


def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    """Integer key per voxel (int64), valid within +-4096 voxels of the origin."""
    v = np.floor(points / voxel).astype(np.int64) + _OFF
    v = np.clip(v, 0, _M - 1)
    return (v[:, 0] * _M + v[:, 1]) * _M + v[:, 2]


class VoxelSet:
    """Occupied voxels, so each window only sends points the viewer has not seen."""

    def __init__(self, voxel: float):
        self.voxel = voxel
        self._keys = np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._keys)

    def add_new(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask over `points`: the first point in each not-yet-occupied voxel."""
        keys = voxel_keys(points, self.voxel)
        uniq, first = np.unique(keys, return_index=True)
        fresh = ~np.isin(uniq, self._keys, assume_unique=True)
        sel = np.zeros(len(points), dtype=bool)
        sel[first[fresh]] = True
        self._keys = np.union1d(self._keys, uniq[fresh])
        return sel
