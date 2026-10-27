"""Ultralytics YOLO26 (closed-set), YOLOE-26 (open-vocabulary text prompts) and YOLO26-depth on MPS."""

import time

import numpy as np
from common import SHOTS, frame_paths, save_result
from ultralytics import YOLO

frames = frame_paths(fps=2)
probe = frames[20]  # kitchen
OPEN_VOCAB = [
    "keys",
    "door",
    "chair",
    "stairs",
    "table",
    "sofa",
    "refrigerator",
    "lamp",
    "plant",
]


def bench(model, name, imgsz, **kw):
    for f in frames[:3]:
        model.predict(f, imgsz=imgsz, device="mps", verbose=False, **kw)
    ts = []
    for f in frames[:40]:
        t0 = time.perf_counter()
        model.predict(f, imgsz=imgsz, device="mps", verbose=False, **kw)
        ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    save_result(
        f"yolo_{name}_{imgsz}",
        {
            "model": name,
            "imgsz": imgsz,
            "latency_ms": round(1000 * med, 1),
            "fps": round(1 / med, 1),
            "note": "end-to-end predict() incl. jpeg decode + NMS-free post",
        },
    )


for w in [w for w in ["yolo26n", "yolo26s"] if not (SHOTS / f"{w}_probe.jpg").exists()]:
    m = YOLO(f"{w}.pt")
    for sz in [640, 320]:
        bench(m, w, sz)
    m.predict(probe, imgsz=640, device="mps", verbose=False)[0].save(
        str(SHOTS / f"{w}_probe.jpg")
    )

ye = YOLO("yoloe-26s-seg.pt")
ye.set_classes(OPEN_VOCAB)
bench(ye, "yoloe-26s-seg-text", 640)
for i, f in enumerate([probe, frames[4], frames[40]]):
    ye.predict(f, imgsz=640, device="mps", conf=0.15, verbose=False)[0].save(
        str(SHOTS / f"yoloe_openvocab_{i}.jpg")
    )

yd = YOLO("yolo26n-depth.pt")
bench(yd, "yolo26n-depth", 640)
