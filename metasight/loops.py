"""Loop closure: notice a place seen before, measure the drift, and bend the map back.

Windows are stitched only to the window before, so error builds up along the walk and a room
seen twice lands twice, a little apart, with its items pinned twice. A keyframe is compared with
older ones by a DINOv2 image descriptor; a candidate also has to be near on the map and share
enough SIFT matches under a fundamental-matrix check. MapAnything then sees the old and new
frame together, which says where the new frame really is relative to the old one, and the gap to
where the chain put it is the drift.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SIM_MIN = 0.75  # descriptor cosine for a candidate (TUM revisits: recall ~0.9, many false ones)
GAP_S = 15.0  # a candidate is at least this much older
NEAR_M = 8.0  # and at most this far on the current (drifted) map
MIN_INLIERS = 40  # SIFT matches surviving the fundamental-matrix check
MAX_SHIFT_M = 2.0  # larger corrections need a second keyframe to agree
MAX_TURN_DEG = 30.0  # likewise
MIN_SHIFT_M = 0.05  # smaller drift is not worth a rebuild
# A larger correction stands only when another keyframe of the same window lands within this
# share of it (and this many degrees): one look-alike room is easy, two in a row agreeing is not.
AGREE_SHARE = 0.15
AGREE_TURN_DEG = 10.0


class Describer:
    """DINOv2-small: normalised class token plus normalised mean patch token."""

    def __init__(self, device: str = "mps"):
        import torch
        from transformers import AutoModel

        self._torch, self.device = torch, device
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        self._mean = np.array([0.485, 0.456, 0.406], np.float32)
        self._std = np.array([0.229, 0.224, 0.225], np.float32)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        torch = self._torch
        h = 168 if bgr.shape[1] >= bgr.shape[0] else 224
        w = 224 if h == 168 else 168
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        x = torch.from_numpy((rgb - self._mean) / self._std).permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad():
            tok = self.model(pixel_values=x).last_hidden_state[0].float().cpu().numpy()
        cls, mean = tok[0], tok[1:].mean(0)
        v = np.concatenate([cls / np.linalg.norm(cls), mean / np.linalg.norm(mean)])
        return (v / np.linalg.norm(v)).astype(np.float32)


_sift = None


def sift_features(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """SIFT keypoint positions and descriptors of one frame (the slow half of a match)."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=1500)
    kp, des = _sift.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)
    return np.float32([k.pt for k in kp]).reshape(-1, 2), des


def matches_between(fa, fb) -> int:
    """SIFT matches between two frames' features that agree with one fundamental matrix."""
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 15:  # with 8 points the RANSAC mask comes back as garbage bytes
        return 0
    pa = ka[[m.queryIdx for m in good]]
    pb = kb[[m.trainIdx for m in good]]
    _, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.999)
    return 0 if inl is None else int(np.count_nonzero(inl))


def shared_matches(a: np.ndarray, b: np.ndarray) -> int:
    """SIFT matches between two frames that agree with one fundamental matrix."""
    return matches_between(sift_features(a), sift_features(b))


# ---------------------------------------------------------------------- similarity transforms
Sim3 = tuple[float, np.ndarray, np.ndarray]  # x -> s * R @ x + t


def compose(a: Sim3, b: Sim3) -> Sim3:
    """a after b."""
    return a[0] * b[0], a[1] @ b[1], a[0] * a[1] @ b[2] + a[2]


def turn_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1) / 2))))


def inverse(a: Sim3) -> Sim3:
    return 1.0 / a[0], a[1].T, -a[1].T @ a[2] / a[0]


def _log(a: Sim3) -> np.ndarray:
    rv, _ = cv2.Rodrigues(a[1])
    return np.concatenate([[math.log(a[0])], rv.ravel(), a[2]])


def _exp(v: np.ndarray) -> Sim3:
    rot, _ = cv2.Rodrigues(v[1:4].reshape(3, 1))
    return math.exp(v[0]), rot, v[4:7].copy()


LOOP_WEIGHT = 3.0  # a checked loop is worth a few windows of chaining
# MapAnything measures in metres, so windows rarely disagree on scale; without this a loop's
# correction shrinks the map at one weak join instead of turning it.
SCALE_WEIGHT = 10.0
_AXIS_W = np.array([SCALE_WEIGHT, 1, 1, 1, 1, 1, 1])


def optimize(
    sims: list[Sim3],
    seq_rel: list[Sim3 | None],
    loops: list[tuple[int, int, Sim3]],
    seq_w: list[float] | None = None,
) -> list[Sim3]:
    """Pose graph over windows: keep each window's measured step from the one before and each
    loop's measured offset, with window 0 held still. `seq_w` says how much each step is
    trusted; a loop's correction goes mostly into the steps trusted least. Residuals are
    scale-free in rotation and log-scale and in metres for translation."""
    from scipy.optimize import least_squares

    n = len(sims)
    if n < 2 or not loops:
        return sims
    wts = seq_w or [1.0] * n
    edges = [(w - 1, w, seq_rel[w], wts[w]) for w in range(1, n) if seq_rel[w] is not None]
    edges += [(a, b, m, LOOP_WEIGHT) for a, b, m in loops]
    x0 = np.concatenate([_log(s) for s in sims[1:]])

    def unpack(x):
        return [sims[0]] + [_exp(x[7 * i : 7 * i + 7]) for i in range(n - 1)]

    def resid(x):
        t = unpack(x)
        out = []
        for a, b, m, wt in edges:
            e = compose(inverse(m), compose(inverse(t[a]), t[b]))
            out.append(wt * _AXIS_W * _log(e))
        return np.concatenate(out)

    # Each edge touches two windows only; telling the solver so keeps a long walk's solve
    # near linear instead of a dense finite-difference Jacobian over every window.
    from scipy.sparse import lil_matrix

    sp = lil_matrix((7 * len(edges), len(x0)), dtype=np.int8)
    for e, (a, b, _m, _w) in enumerate(edges):
        for v in (a, b):
            if v > 0:
                sp[7 * e : 7 * e + 7, 7 * (v - 1) : 7 * v] = 1
    sol = least_squares(resid, x0, method="trf", jac_sparsity=sp, ftol=1e-6)
    return unpack(sol.x)
