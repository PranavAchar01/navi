"""The safety loop: one frame in, per-sector distances and labelled obstacles out.

Metric depth (Depth Anything V2 Metric-Indoor Small) gives metres; YOLO26n names what is there.
Monocular metric scale drifts, so the memory loop feeds back a scale correction measured against
the MapAnything map (`apply_scale_hint`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .align import map_resolution
from .config import Settings
from .geometry import CAMERA_SECTORS, sector_distances_from_depth

DEPTH_REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


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


def proximity_grid(
    depth: np.ndarray, size: tuple[int, int] = (192, 108), hfov_deg: float = 100.0
) -> np.ndarray:
    """Per-pixel distance in centimetres / 2 (uint8, 255 = far or floor) for the HUD glow.

    The floor right in front of the wearer is always close, so it is found and removed: the
    depth is lifted to 3D, a plane is fitted to the lower part of the frame (RANSAC, normal
    near vertical), and its inliers are marked far. Everything else keeps its distance.
    """
    w, h = size
    d = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    f = 0.5 * w / np.tan(np.radians(hfov_deg) / 2)
    u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    P = np.stack([u * d / f, v * d / f, d], -1)
    low = P[int(h * 0.55) :].reshape(-1, 3)
    low = low[np.isfinite(low).all(1) & (low[:, 2] > 0.1)]
    floor = np.zeros((h, w), bool)
    if len(low) > 50:
        rng = np.random.default_rng(0)
        best, best_n = None, 0
        for _ in range(48):
            a, b, c = low[rng.choice(len(low), 3, replace=False)]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n /= nn
            if abs(n[1]) < 0.75:  # camera y is down: the floor normal is mostly vertical
                continue
            cnt = int((np.abs((low - a) @ n) < 0.05).sum())
            if cnt > best_n:
                best, best_n = (a, n), cnt
        if best is not None and best_n > 0.2 * len(low):
            a, n = best
            floor = np.abs((P - a) @ n) < 0.09
    out = np.clip(d * 50, 0, 254).astype(np.uint8)
    out[floor | ~np.isfinite(d)] = 255
    return out


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
        self._det = YOLO(str(s.weights_dir / "yolo26n.pt"))
        self._seg = None
        if s.segment:
            self._seg = YOLO(str(s.weights_dir / "yoloe-26s-seg.pt"))
            self._seg.set_classes(list(s.watchlist))
        self.scale = s.depth_scale

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
        """Blend a measured map-to-depth ratio into the running scale, clamped to sane bounds."""
        ratio = float(np.clip(ratio, 0.3, 3.0))
        self.scale = float(np.clip((1 - alpha) * self.scale + alpha * ratio, 0.3, 3.0))

    def estimate_depth(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        iw, ih = map_resolution(w, h)
        rgb = cv2.cvtColor(
            cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
        )
        x = (rgb.astype(np.float32) / 255 - _MEAN) / _STD
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device, torch.float16)
        with torch.no_grad():
            d = self._depth(pixel_values=t).predicted_depth[0].float().cpu().numpy()
        return d * self.scale

    def process(self, bgr: np.ndarray) -> SafetyResult:
        t0 = time.perf_counter()
        depth = self.estimate_depth(bgr)
        t1 = time.perf_counter()
        sectors = sector_distances_from_depth(depth, len(CAMERA_SECTORS))
        if self._seg is not None:
            # The masks already carry a label, a box and a distance: no second detector pass.
            objects = self.segment(bgr, depth)
            t2 = t3 = time.perf_counter()
            obstacles = [Obstacle(o.label, o.score, o.box, o.dist_m) for o in objects]
            return SafetyResult(
                sectors,
                obstacles,
                depth.astype(np.float16),
                objects,
                {
                    "depth": 1000 * (t1 - t0),
                    "detect": 1000 * (t2 - t1),
                    "post": 0.0,
                    "total": 1000 * (t3 - t0),
                },
            )
        r = self._det.predict(
            bgr, imgsz=self.s.detector_imgsz, device="mps", verbose=False, conf=0.35
        )[0]
        t2 = time.perf_counter()
        h, w = bgr.shape[:2]
        dh, dw = depth.shape
        obstacles = []
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
                int(cy0 / h * dh) : int(cy1 / h * dh) + 1, int(cx0 / w * dw) : int(cx1 / w * dw) + 1
            ]
            dist = float(np.percentile(patch, 25)) if patch.size else float("inf")
            obstacles.append(
                Obstacle(r.names[int(c)], float(p), (x0 / w, y0 / h, x1 / w, y1 / h), dist)
            )
        objects = self.segment(bgr, depth) if self._seg is not None else []
        t3 = time.perf_counter()
        return SafetyResult(
            sectors,
            obstacles,
            depth.astype(np.float16),
            objects,
            {
                "depth": 1000 * (t1 - t0),
                "detect": 1000 * (t2 - t1),
                "post": 1000 * (t3 - t2),
                "total": 1000 * (t3 - t0),
            },
        )
