"""Proof of the fast obstacle path on the walkthrough clip: metric depth + YOLO26n -> per-sector
nearest distance -> what an 8-motor belt would do. Writes shots/proximity_demo.mp4 and a timing record.

The camera sees roughly the front 3 of 8 belt sectors (about 100 deg horizontal on the phone
ultrawide). Side and rear sectors need the 3D map, which is the reconstruction model's job.
"""

import subprocess
import time

import cv2
import numpy as np
import torch
from common import FRAMES, SHOTS, save_result
from transformers import AutoModelForDepthEstimation
from ultralytics import YOLO

DEV = torch.device("mps")
REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
depth_model = (
    AutoModelForDepthEstimation.from_pretrained(REPO, torch_dtype=torch.float16)
    .to(DEV)
    .eval()
)
det = YOLO("yolo26n.pt")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
W, H = 518, 294
SECTORS = ["front-left", "front", "front-right"]
NEAR, WARN = 1.0, 2.0  # metres: buzz hard / buzz soft

out_dir = SHOTS / "proximity_frames"
out_dir.mkdir(exist_ok=True)
frames = sorted(FRAMES.glob("*.jpg"))[::3]  # 10 fps
t_depth, t_det, t_total = [], [], []
for k, f in enumerate(frames):
    t0 = time.perf_counter()
    bgr = cv2.imread(str(f))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = (
        cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        - MEAN
    ) / STD
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(DEV, torch.float16)
    t1 = time.perf_counter()
    with torch.no_grad():
        d = depth_model(pixel_values=x).predicted_depth[0].float().cpu().numpy()
    t2 = time.perf_counter()
    r = det.predict(bgr, imgsz=320, device="mps", verbose=False, conf=0.35)[0]
    t3 = time.perf_counter()
    # Nearest obstacle per sector: 5th percentile depth in the lower 70% of the image (skip ceiling).
    band = d[int(0.3 * d.shape[0]) :]
    cols = np.array_split(np.arange(band.shape[1]), 3)
    near = [float(np.percentile(band[:, c], 5)) for c in cols]
    t_total.append(time.perf_counter() - t0)
    t_depth.append(t2 - t1)
    t_det.append(t3 - t2)

    vis = bgr.copy()
    dh, dw = d.shape
    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        x0, y0, x1, y1 = b.astype(int)
        sx, sy = dw / vis.shape[1], dh / vis.shape[0]
        patch = d[int(y0 * sy) : int(y1 * sy) + 1, int(x0 * sx) : int(x1 * sx) + 1]
        dist = float(np.median(patch)) if patch.size else float("nan")
        col = (
            (0, 0, 255)
            if dist < NEAR
            else (0, 165, 255)
            if dist < WARN
            else (80, 200, 80)
        )
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 3)
        cv2.putText(
            vis,
            f"{r.names[int(c)]} {dist:.1f} m",
            (x0 + 4, max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            col,
            2,
            cv2.LINE_AA,
        )
    # Belt strip: 3 front sectors, filled by intensity.
    bw = vis.shape[1] // 3
    for i, n in enumerate(near):
        inten = float(np.clip((WARN + 0.5 - n) / (WARN + 0.5 - 0.4), 0, 1))
        col = (0, 0, 255) if n < NEAR else (0, 165, 255) if n < WARN else (90, 90, 90)
        cv2.rectangle(
            vis,
            (i * bw + 6, vis.shape[0] - 70),
            (i * bw + 6 + int((bw - 12) * inten), vis.shape[0] - 20),
            col,
            -1,
        )
        cv2.rectangle(
            vis,
            (i * bw + 6, vis.shape[0] - 70),
            ((i + 1) * bw - 6, vis.shape[0] - 20),
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis,
            f"{SECTORS[i]} {n:.1f} m",
            (i * bw + 14, vis.shape[0] - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        f"depth {1000 * (t2 - t1):.0f} ms  det {1000 * (t3 - t2):.0f} ms  (M5 Pro MPS, background QoS)",
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_dir / f"{k:04d}.jpg"), vis)

subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-framerate",
        "10",
        "-i",
        str(out_dir / "%04d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(SHOTS / "proximity_demo.mp4"),
    ],
    check=True,
)
med = lambda a: round(1000 * float(np.median(a[3:])), 1)
save_result(
    "proximity_pipeline",
    {
        "frames": len(frames),
        "depth_ms": med(t_depth),
        "det_ms": med(t_det),
        "total_ms": med(t_total),
        "pipeline_fps": round(1000 / med(t_total), 1),
        "note": "serial depth+det per frame incl. jpeg decode, 518x294 depth, 320 det",
    },
)
