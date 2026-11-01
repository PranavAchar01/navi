"""The safety loop: one frame in, per-sector distances and labelled obstacles out.

Metric depth (Depth Anything V2 Metric-Indoor Small) gives metres; YOLO26n names what is there.
Monocular metric scale drifts, so the memory loop feeds back a scale correction measured against
the MapAnything map (`apply_scale_hint`).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .align import map_resolution
from .config import Settings
from .roomprior import CONFIRM_CONF, LIVE_CONF

DEPTH_REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
# The same depth net converted for Core ML (scripts/coreml_depth.py), fixed at 294 x 518. On the
# Mac it runs in about 9 ms against 15 to 30 ms in PyTorch on MPS, and leaves the GPU to the map.
# Depth matches PyTorch within 0.6% (median) on 123 walk frames.
COREML_DEPTH = (
    Path(__file__).resolve().parent.parent
    / "ios/CameraAccess/MetaSight/Models/ReflexDepth.mlpackage"
)


@dataclass
class Obstacle:
    label: str
    score: float
    box: tuple[float, float, float, float]  # normalised x0, y0, x1, y1
    dist_m: float


@dataclass
class SegObject:
    label: str
    score: float
    poly: list[float]  # normalised x, y pairs, simplified outline of the mask
    box: tuple[float, float, float, float]
    dist_m: float


@dataclass
class SafetyResult:
    sectors: list[float]  # front-left, front, front-right, metres
    obstacles: list[Obstacle]
    depth: np.ndarray  # float16 metres at network resolution, after scale correction
    objects: list[SegObject] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    scene: Scene | None = None
    ttc_s: float = math.inf  # time to contact with what is in the path, from its closing speed
    floor_scale: float | None = None  # this frame's scale measured from the floor plane
    depth_m: np.ndarray | None = field(default=None, repr=False)  # float32 `depth`, for labels
    weak: list[Obstacle] = field(default_factory=list)  # below the live threshold, for the room


@dataclass
class Scene:
    """What the floor-aligned geometry of one frame says, in metres."""

    sectors: list[float]  # front-left, front, front-right: nearest horizontal range
    grid: np.ndarray  # uint8 range in cm / 2 for obstacle pixels, 255 elsewhere (floor, far)
    corridor: np.ndarray  # bool, pixels inside the 0.9 m wide walking path ahead
    path_m: float  # nearest obstacle in the walking path
    trip_m: float  # nearest low obstacle (5 to 40 cm tall) in the path
    drop_m: float  # nearest drop below the floor (stairs down, a step) in the path
    floor: bool  # a floor plane was found this frame (or reused from the last 1.5 s)
    cam_h_raw: float  # camera height above the floor in raw (unscaled) depth units


GRID = (192, 108)
BODY_HALF_M = 0.45  # half the width of the path a walking person sweeps
SECTOR_HALF_DEG = 22.5


def _fit_floor(
    P: np.ndarray, prev: tuple[np.ndarray, float] | None, tol: float, c_lo: float, c_hi: float
):
    """RANSAC plane through the lower part of the view, normal near vertical, oriented up.

    Returns (n, c) with height(P) = P @ n + c, or None. Only planes whose distance below the
    camera `c` (raw units) is within [c_lo, c_hi] can be the floor: a bed or table top is level
    too, but it is not an eye height below you. The previous plane is one candidate, which keeps
    the floor steady from frame to frame.
    """
    h = P.shape[0]
    low = P[int(h * 0.55) :].reshape(-1, 3)
    low = low[np.isfinite(low).all(1) & (low[:, 2] > 0.1)]
    if len(low) < 50:
        return None
    rng = np.random.default_rng(0)
    # 48 random triples, drawn one at a time as before so the planes stay the same, then
    # their normals and support counted in one go.
    tri = low[np.stack([rng.choice(len(low), 3, replace=False) for _ in range(48)])]
    a = tri[:, 0]
    n = np.cross(tri[:, 1] - a, tri[:, 2] - a)
    nn = np.linalg.norm(n, axis=1)
    keep = nn >= 1e-6
    n = n[keep] / nn[keep, None]
    a = a[keep]
    level = np.abs(n[:, 1]) >= 0.75  # camera y is down: the floor normal is mostly vertical
    n, a = n[level], a[level]
    n[n[:, 1] > 0] *= -1  # point up
    N = np.vstack([prev[0][None], n]) if prev is not None else n
    C = -np.einsum("ij,ij->i", a, n)
    if prev is not None:
        C = np.concatenate([[prev[1]], C])
    inr = (C > c_lo) & (C < c_hi)
    sup = np.zeros(len(C), int)
    if inr.any():
        sup[inr] = (np.abs(low @ N[inr].T + C[inr]) < tol).sum(0)
    # Of the level planes with real support, the lowest is the floor: a bed or a table top can
    # fill more of the view, but it is never under the floor.
    best = None
    for i in np.nonzero(inr & (sup >= 0.1 * len(low)))[0]:
        if best is None or C[i] > best[1] + tol:
            best = (N[i], float(C[i]))
    if best is None:
        return None
    # Refine on the inliers (least squares), then keep the camera above it.
    n, c = best
    inl = low[np.abs(low @ n + c) < tol]
    cen = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - cen, full_matrices=False)
    n2 = vt[2] if vt[2][1] < 0 else -vt[2]
    if abs(n2[1]) >= 0.75:
        n, c = n2, -float(cen @ n2)
    return (n, c) if c_lo < c < c_hi else None


FLOOR_BEND_M = 0.35  # the near floor bends up to this far off the plane in the depth net
FLOOR_TILT = 0.6  # ...and its surface tilts up to about 53 degrees; a wall or box face is 90


def _floor_reach(hgt: np.ndarray, tilt: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """The floor you stand on, traced up each image column from the bottom edge.

    The depth net bends the floor just ahead of the feet up off the plane (20 to 30 cm on the
    phone walk, its surface tilted 30 to 40 degrees), so the plane alone calls it an obstacle
    under a metre away in every frame. Pixels joined to the bottom edge by an unbroken run of
    such floor-like pixels are floor; the run stops at the first upright face (a wall, a box,
    a chair leg), and everything from there up is judged by the plane as before.
    """
    floorish = ok & (np.abs(hgt) < FLOOR_BEND_M) & (tilt > FLOOR_TILT)
    return np.logical_and.accumulate(floorish[::-1], axis=0)[::-1]


def analyse(
    depth_raw: np.ndarray,
    scale: float,
    prior: float,
    hfov_deg: float,
    eye_h: float,
    prev: tuple[np.ndarray, float] | None,
    hold: np.ndarray | None = None,
) -> tuple[Scene, tuple[np.ndarray, float] | None]:
    """Floor-aligned obstacle geometry for one frame.

    The depth is lifted to 3D, the floor plane is found, and every pixel gets a height above the
    floor and a horizontal range and bearing on it. Obstacles are what stands 5 cm or more above
    the floor and below head height; the floor itself never counts. Below the floor by more than
    25 cm is a drop. Ranges are horizontal: what you would actually walk into.
    """
    # The grid takes the frame's own orientation: a 9:16 stream squeezed into a 16:9 grid bends
    # every ray, the floor stops looking level, and the whole lower view reads as an obstacle.
    w, h = GRID if depth_raw.shape[1] >= depth_raw.shape[0] else GRID[::-1]
    d = cv2.resize(depth_raw.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    f = 0.5 * w / np.tan(np.radians(hfov_deg) / 2)
    u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    P = np.stack([u * d / f, v * d / f, d], -1)
    ok = np.isfinite(d) & (d > 0.05)
    # Plane tolerance in raw units: 4 cm at the expected scale.
    # The floor gate uses the lens prior, not the running scale, so a wrong plane cannot drag
    # the scale and the gate along with it.
    plane = _fit_floor(P, prev, 0.04 / scale, 0.75 * eye_h / prior, 1.3 * eye_h / prior)
    found = plane is not None
    if plane is None and hold is not None:
        # Floor out of view (a desk fills the bottom of the frame): keep the last floor
        # direction and put it one eye height below the camera. Head pitch changes slowly.
        plane = (hold, eye_h / scale)
    corridor = np.zeros((h, w), bool)
    if plane is None:
        # No floor in view (looking up, or pressed against something): plain depth, top cut.
        rng_m = d * scale
        obst = ok.copy()
        obst[: int(0.3 * h)] = False
        bearing = np.degrees(np.arctan2(u, f))
        corridor = obst & (np.abs(bearing) < 20)
        hgt = np.full((h, w), np.nan)
        cam_h = float("nan")
    else:
        n, c = plane
        cam_h = c
        hgt = (P @ n + c) * scale
        fwd = np.array([0.0, 0.0, 1.0]) - n[2] * n
        fwd /= np.linalg.norm(fwd)
        right = np.cross(n, fwd)
        right = -right if right[0] < 0 else right
        x, z = (P @ right) * scale, (P @ fwd) * scale
        rng_m = np.hypot(x, z)
        bearing = np.degrees(np.arctan2(x, z))
        # Surface normals: the ultra-wide floor comes out of the depth net slightly curved, so a
        # floor pixel can sit a few cm off the plane. If it still faces up, it is still floor.
        Ps = cv2.GaussianBlur(P.astype(np.float32), (5, 5), 0)
        nrm = np.cross(np.gradient(Ps, axis=1), np.gradient(Ps, axis=0))
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
        tilt = np.abs(nrm @ n)
        obst = ok & (hgt < eye_h + 0.3) & (((hgt > 0.05) & (tilt <= 0.9)) | (hgt > 0.3))
        obst &= ~_floor_reach(hgt, tilt, ok)
        corridor = (np.abs(x) < BODY_HALF_M) & (z > 0)
    sectors = []
    a = SECTOR_HALF_DEG
    for lo, hi in ((-67.5, -a), (-a, a), (a, 67.5)):
        sel = obst & (bearing >= lo) & (bearing < hi)
        sectors.append(float(np.percentile(rng_m[sel], 5)) if sel.sum() >= 12 else math.inf)

    def nearest(mask: np.ndarray, min_px: int = 12) -> float:
        return float(np.percentile(rng_m[mask], 5)) if mask.sum() >= min_px else math.inf

    path_m = nearest(obst & corridor)
    # Trip hazards: things in the path whose whole height is under 40 cm (shoes, a box, a cable
    # run), not the bottom edge of a bed.
    trip_m = math.inf
    if plane is not None:
        n_c, lab = cv2.connectedComponents((obst & corridor).astype(np.uint8), connectivity=8)
        for i in range(1, n_c):
            m = lab == i
            if m.sum() >= 12 and np.nanmax(hgt[m]) < 0.4:
                trip_m = min(trip_m, float(np.percentile(rng_m[m], 5)))
    drop_m = (
        nearest(ok & corridor & (hgt < -0.25) & (rng_m < 3.0), int(0.004 * w * h))
        if plane is not None
        else math.inf
    )
    # The outlines' grid. With no floor the top cut is for the cue only: drawn, it ran a straight
    # line across the frame at 0.3 through whatever stood there, so outlines get the whole view.
    shown = ok if plane is None else obst
    grid = np.full((h, w), 255, np.uint8)
    grid[shown] = np.clip(rng_m[shown] * 50, 0, 254).astype(np.uint8)
    return Scene(sectors, grid, corridor, path_m, trip_m, drop_m, found, cam_h), (
        plane if found else None
    )


def hazards(
    grid: np.ndarray, corridor: np.ndarray, near_cm2: int = 100, keep: int = 2
) -> list[dict]:
    """The obstacles that matter: near surfaces (under 2 m, floor already removed) that reach
    into the 0.9 m walking path, nearest first, each as a smooth outline and a distance.

    Walls off to the side are near too, but they are not in the way, so they stay quiet. Each
    region is cut to a 0.4 m depth band behind its nearest in-path point, so a chair against a
    wall outlines the chair, not chair and wall together.
    """
    h, w = grid.shape
    k3 = np.ones((3, 3), np.uint8)
    near = (grid < near_cm2).astype(np.uint8)
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, k3)
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(near, connectivity=8)
    found = []
    for i in range(1, n):
        m = lab == i
        inpath = m & corridor
        if inpath.sum() < 0.002 * h * w:
            continue
        dist = float(np.percentile(grid[inpath], 5)) / 50
        band = (m & (grid < (dist + 0.4) * 50)).astype(np.uint8)
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, k3)
        bn, blab = cv2.connectedComponents(band, connectivity=8)
        if bn < 2:
            continue
        # The piece of the band that is actually in the way.
        hits = np.bincount(blab[corridor], minlength=bn)
        hits[0] = 0
        m = blab == int(hits.argmax())
        if m.sum() < 0.004 * h * w:
            continue
        big = cv2.resize(m.astype(np.uint8) * 255, (w * 4, h * 4), interpolation=cv2.INTER_LINEAR)
        big = cv2.GaussianBlur(big, (0, 0), 4)
        cs, _ = cv2.findContours(
            (big > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.2, True)[:, 0, :].astype(np.float32)
        poly = (c / [w * 4, h * 4]).round(4).ravel().tolist()
        ys, xs = np.nonzero(m & corridor)
        top = (float(xs.mean()) / w, float(ys.min()) / h)
        found.append(
            {"poly": poly, "dist": round(dist, 2), "tag": [round(top[0], 3), round(top[1], 3)]}
        )
    found.sort(key=lambda f: f["dist"])
    return found[:keep]


class SafetyLoop:
    def __init__(self, s: Settings, device: str = "mps"):
        from transformers import AutoModelForDepthEstimation
        from ultralytics import YOLO

        self.s, self.device = s, torch.device(device)
        self._depth = (
            AutoModelForDepthEstimation.from_pretrained(DEPTH_REPO, dtype=torch.float16)
            .to(self.device)
            .eval()
        )
        self._ml = None
        if COREML_DEPTH.exists():
            import coremltools as ct

            # Neural Engine only: on a walk replay with the map running, p95 33 ms against 56 ms
            # when Core ML may also use the GPU.
            units = ct.ComputeUnit.CPU_AND_NE
            self._ml = ct.models.MLModel(str(COREML_DEPTH), compute_units=units)
        # The detector in Core ML when exported (scripts/coreml_detect.py): 2 ms against 6 ms
        # alone on MPS, and it stays off the GPU the map is using.
        ml_det = s.weights_dir / "yolo26n.mlpackage"
        self._det = (
            YOLO(str(ml_det), task="detect")
            if ml_det.exists()
            else YOLO(str(s.weights_dir / "yolo26n.pt"))
        )
        self._det_device = None if ml_det.exists() else "mps"
        # Core ML compiles on its first call (about 2 s): pay that here, not on the first frame.
        blank = np.zeros((896, 504, 3), np.uint8)
        self.estimate_depth(blank)
        self._det.predict(blank, imgsz=s.detector_imgsz, device=self._det_device, verbose=False)
        self._seg = None
        if s.segment:
            self._seg = YOLO(str(s.weights_dir / "yoloe-26s-seg.pt"))
            self._seg.set_classes(list(s.watchlist))
        self.scale = s.depth_scale
        self.plane: tuple[np.ndarray, float] | None = None
        self.plane_t = 0.0
        self.floor_t = 0.0
        self.path_hist: deque[tuple[float, float]] = deque()
        self.drop_hist: deque[float] = deque(maxlen=4)

    def segment(self, bgr: np.ndarray, depth: np.ndarray) -> list[SegObject]:
        """Open-vocabulary masks with a distance each (25th percentile of depth under the mask)."""
        r = self._seg.predict(
            bgr, imgsz=self.s.detector_imgsz, device="mps", verbose=False, conf=0.3
        )[0]
        if r.masks is None:
            return []
        dh, dw = depth.shape
        h, w = bgr.shape[:2]
        out = []
        for xyn, b, c, p in zip(
            r.masks.xyn,
            r.boxes.xyxyn.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            strict=True,
        ):
            if len(xyn) < 3:
                continue
            m = np.zeros((dh, dw), np.uint8)
            cv2.fillPoly(m, [(xyn * [dw, dh]).astype(np.int32)], 1)
            vals = depth[m > 0]
            dist = float(np.percentile(vals, 25)) if vals.size else float("inf")
            ap = cv2.approxPolyDP((xyn * [w, h]).astype(np.float32), 1.5, True)[:, 0, :]
            poly = (ap / [w, h]).round(4).ravel().tolist()
            out.append(
                SegObject(self._seg.names[int(c)], float(p), poly, tuple(map(float, b)), dist)
            )
        return out

    def apply_scale_hint(self, ratio: float, alpha: float = 0.4) -> None:
        """The map's scale, used only while no floor is in view (the floor is the better ruler)."""
        if time.time() - self.floor_t < 2.0:
            return
        ratio = float(np.clip(ratio, 0.3, 3.0))
        self.scale = float(np.clip((1 - alpha) * self.scale + alpha * ratio, 0.3, 3.0))

    def estimate_depth(self, bgr: np.ndarray) -> np.ndarray:
        """Raw metric depth from the network, before any scale correction."""
        h, w = bgr.shape[:2]
        iw, ih = map_resolution(w, h)
        rgb = cv2.cvtColor(
            cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
        )
        if self._ml is not None and (iw, ih) == (294, 518):
            from PIL import Image

            out = self._ml.predict({"image": Image.fromarray(rgb)})["depth"]
            return np.asarray(out, np.float32)[0]
        x = (rgb.astype(np.float32) / 255 - _MEAN) / _STD
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device, torch.float16)
        with torch.no_grad():
            return self._depth(pixel_values=t).predicted_depth[0].float().cpu().numpy()

    def _geometry(self, raw: np.ndarray) -> tuple[Scene, float | None, float]:
        """Floor-aligned scene, the floor's scale measurement, and time to contact."""
        now = time.time()
        prev = self.plane if now - self.plane_t < 1.5 else None
        prior = self.s.depth_scale
        hold = self.plane[0] if self.plane is not None and now - self.plane_t < 5.0 else None
        scene, plane = analyse(
            raw, self.scale, prior, self.s.hfov_deg, self.s.eye_height_m, prev, hold
        )
        meas = None
        if plane is not None:
            self.plane, self.plane_t = plane, now
            meas = self.s.eye_height_m / plane[1]
            # The floor is the ruler: the camera sits eye_height_m above it. The lens prior bounds
            # how far one frame can move it.
            meas = float(np.clip(meas, 0.8 * prior, 1.25 * prior))
            self.scale = 0.85 * self.scale + 0.15 * meas
            self.floor_t = now
            scene, _ = analyse(raw, self.scale, prior, self.s.hfov_deg, self.s.eye_height_m, plane)
        # A drop is only reported when it holds for 3 of the last 4 frames.
        self.drop_hist.append(scene.drop_m)
        if sum(math.isfinite(x) for x in self.drop_hist) < 3:
            scene.drop_m = math.inf
        # Closing speed on the path: least-squares slope over the last 0.6 s.
        self.path_hist.append((now, scene.path_m))
        while self.path_hist and now - self.path_hist[0][0] > 0.6:
            self.path_hist.popleft()
        pts = [(t, d) for t, d in self.path_hist if math.isfinite(d)]
        ttc = math.inf
        if len(pts) >= 5 and math.isfinite(scene.path_m):
            t, d = np.array(pts).T
            slope = np.polyfit(t - t[-1], d, 1)[0]
            if slope < -0.15:  # closing faster than a slow shuffle
                ttc = scene.path_m / -slope
        return scene, meas, ttc

    def process(self, bgr: np.ndarray) -> SafetyResult:
        return self.label(bgr, self.reflex(bgr))

    def reflex(self, bgr: np.ndarray) -> SafetyResult:
        """The stop layer: live depth and the floor fit, everything the belt's plan needs. No
        detector, so a buzz never waits for labels (`label` adds them after)."""
        t0 = time.perf_counter()
        raw = self.estimate_depth(bgr)
        scene, meas, ttc = self._geometry(raw)
        depth = raw * self.scale
        t1 = time.perf_counter()
        return SafetyResult(
            scene.sectors,
            [],
            depth.astype(np.float16),
            [],
            {"depth": 1000 * (t1 - t0)},
            scene=scene,
            ttc_s=ttc,
            floor_scale=meas,
            depth_m=depth,
        )

    def label(self, bgr: np.ndarray, res: SafetyResult) -> SafetyResult:
        """The label layer on the same frame: what the near things are, with a distance each."""
        t1 = time.perf_counter()
        depth = res.depth_m
        if self._seg is not None:
            # The masks already carry a label, a box and a distance: no second detector pass.
            res.objects = self.segment(bgr, depth)
            res.obstacles = [Obstacle(o.label, o.score, o.box, o.dist_m) for o in res.objects]
            t2 = t3 = time.perf_counter()
        else:
            r = self._det.predict(
                bgr,
                imgsz=self.s.detector_imgsz,
                device=self._det_device,
                verbose=False,
                conf=CONFIRM_CONF,
            )[0]
            t2 = time.perf_counter()
            h, w = bgr.shape[:2]
            dh, dw = depth.shape
            for b, c, p in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                strict=True,
            ):
                x0, y0, x1, y1 = b
                # Central half of the box: edges are mostly background and drag the distance out.
                cx0, cx1 = x0 + 0.25 * (x1 - x0), x1 - 0.25 * (x1 - x0)
                cy0, cy1 = y0 + 0.25 * (y1 - y0), y1 - 0.25 * (y1 - y0)
                patch = depth[
                    int(cy0 / h * dh) : int(cy1 / h * dh) + 1,
                    int(cx0 / w * dw) : int(cx1 / w * dw) + 1,
                ]
                dist = float(np.percentile(patch, 25)) if patch.size else float("inf")
                (res.obstacles if p >= LIVE_CONF else res.weak).append(
                    Obstacle(r.names[int(c)], float(p), (x0 / w, y0 / h, x1 / w, y1 / h), dist)
                )
            t3 = time.perf_counter()
        res.timings_ms.update(
            {
                "detect": 1000 * (t2 - t1),
                "post": 1000 * (t3 - t2),
                "total": res.timings_ms["depth"] + 1000 * (t3 - t1),
            }
        )
        return res
