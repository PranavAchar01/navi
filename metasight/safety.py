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
class SafetyResult:
    sectors: list[float]  # front-left, front, front-right, metres
    obstacles: list[Obstacle]
    depth: np.ndarray  # float16 metres at network resolution, after scale correction
    timings_ms: dict[str, float] = field(default_factory=dict)


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
        self.scale = 1.0

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
        r = self._det.predict(
            bgr, imgsz=self.s.detector_imgsz, device="mps", verbose=False, conf=0.35
        )[0]
        t2 = time.perf_counter()
        sectors = sector_distances_from_depth(depth, len(CAMERA_SECTORS))
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
        t3 = time.perf_counter()
        return SafetyResult(
            sectors,
            obstacles,
            depth.astype(np.float16),
            {
                "depth": 1000 * (t1 - t0),
                "detect": 1000 * (t2 - t1),
                "post": 1000 * (t3 - t2),
                "total": 1000 * (t3 - t0),
            },
        )
