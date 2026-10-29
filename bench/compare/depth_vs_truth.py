"""Distance to an object, three ways, scored against the Kinect depth that ships with TUM RGB-D.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python bench/compare/depth_vs_truth.py

For every YOLO26n box on sampled TUM frames, the true distance is the median sensor depth in
the box's central half. Each method then guesses that distance:

  yolo_size   YOLO alone: pinhole range from the box height (or width) and a typical real size
              for the class. This is the best a box-only detector can do.
  da2         YOLO + Depth Anything V2 Metric-Indoor Small, one frame at a time (the live loop's
              depth net, before the map re-anchors its scale).
  da2_anchor  The same, rescaled per window by the median ratio to MapAnything's depth: what the
              live loop does once the map exists.
  mapany      MapAnything (facebook/map-anything-apache) on windows of 8 views with the known lens,
              its own metric depth per view.

Direction (bearing) comes from the box centre and the lens in every method, so it is identical
and not scored. Writes bench/results/compare/depth_vs_truth.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metasight.memory import MapAnythingRunner  # noqa: E402

TUM = ROOT / "data" / "tum"
OUT = ROOT / "bench" / "results" / "compare"
OUT.mkdir(parents=True, exist_ok=True)
SEQS = {
    "rgbd_dataset_freiburg1_desk": 517.3,
    "rgbd_dataset_freiburg1_360": 517.3,
    "rgbd_dataset_freiburg2_desk": 520.9,
    "rgbd_dataset_freiburg2_pioneer_slam": 520.9,
}
N_FRAMES, WIN = 32, 8
# Typical real size in metres, ("h" or "w", size). Rough by design: YOLO alone has nothing better.
PRIOR = {
    "person": ("h", 1.70),
    "chair": ("h", 0.90),
    "tv": ("h", 0.40),
    "laptop": ("h", 0.24),
    "bottle": ("h", 0.25),
    "cup": ("h", 0.10),
    "potted plant": ("h", 0.40),
    "couch": ("h", 0.85),
    "dining table": ("h", 0.75),
    "book": ("h", 0.22),
    "keyboard": ("w", 0.45),
    "mouse": ("w", 0.065),
    "cell phone": ("h", 0.14),
    "teddy bear": ("h", 0.30),
    "clock": ("h", 0.30),
    "vase": ("h", 0.25),
    "refrigerator": ("h", 1.75),
    "bed": ("h", 0.60),
    "suitcase": ("h", 0.60),
    "backpack": ("h", 0.45),
    "remote": ("w", 0.18),
    "scissors": ("w", 0.18),
    "sink": ("w", 0.50),
    "toilet": ("h", 0.75),
    "bench": ("h", 0.45),
}
DEV = torch.device("mps")


def assoc(seq: Path) -> list[tuple[Path, Path]]:
    def read(name):
        rows = [ln.split() for ln in open(seq / name) if not ln.startswith("#")]
        return np.array([float(r[0]) for r in rows]), [seq / r[1] for r in rows]

    tr, rgb = read("rgb.txt")
    td, dep = read("depth.txt")
    pairs = []
    for t, p in zip(tr, rgb, strict=True):
        j = int(np.argmin(np.abs(td - t)))
        if abs(td[j] - t) < 0.02:
            pairs.append((p, dep[j]))
    return pairs


class DA2:
    def __init__(self):
        from transformers import AutoModelForDepthEstimation

        repo = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        self.m = AutoModelForDepthEstimation.from_pretrained(repo, dtype=torch.float16)
        self.m = self.m.to(DEV).eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (518, 392), interpolation=cv2.INTER_AREA), 4)
        x = (rgb.astype(np.float32) / 255 - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(DEV, torch.float16)
        with torch.no_grad():
            d = self.m(pixel_values=x).predicted_depth[0].float().cpu().numpy()
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)


def inner(d: np.ndarray, box, mask=None) -> float | None:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    sl = (
        slice(int(y1 + bh / 4), int(y2 - bh / 4) + 1),
        slice(int(x1 + bw / 4), int(x2 - bw / 4) + 1),
    )
    v = d[sl]
    ok = np.isfinite(v) & (v > 0.05)
    if mask is not None:
        ok &= mask[sl]
    return float(np.median(v[ok])) if ok.sum() >= 20 else None


def summarise(rows: list[dict], key: str) -> dict:
    pairs = [(r[key], r["gt"]) for r in rows if r.get(key)]
    if not pairs:
        return {"n": 0}
    p, g = np.array(pairs).T
    rel = np.abs(p - g) / g
    return {
        "n": len(pairs),
        "abs_rel_median": round(float(np.median(rel)), 3),
        "abs_rel_mean": round(float(np.mean(rel)), 3),
        "abs_err_median_m": round(float(np.median(np.abs(p - g))), 3),
        "within_10pct": round(float(np.mean(rel < 0.10)), 3),
        "within_25pct": round(float(np.mean(rel < 0.25)), 3),
        "ratio_median": round(float(np.median(p / g)), 3),
    }


def main() -> None:
    from ultralytics import YOLO

    det = YOLO(str(ROOT / "yolo26n.pt"))
    da2 = DA2()
    rows, pix, times = (
        [],
        {"da2": [], "da2_anchor": [], "mapany": []},
        {"yolo": [], "da2": [], "mapany_window": []},
    )
    runners: dict[float, MapAnythingRunner] = {}
    for name, fx in SEQS.items():
        pairs = assoc(TUM / name)
        idx = np.linspace(0, len(pairs) - 1, N_FRAMES).round().astype(int)
        hfov = math.degrees(2 * math.atan(320 / fx))
        if hfov not in runners:
            runners.clear()
            runners[hfov] = MapAnythingRunner("mps", hfov_deg=hfov)
        ma = runners[hfov]
        for w0 in range(0, N_FRAMES, WIN):
            sel = [pairs[i] for i in idx[w0 : w0 + WIN]]
            bgrs = [cv2.imread(str(p)) for p, _ in sel]
            gts = [
                cv2.imread(str(d), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000 for _, d in sel
            ]
            t = time.perf_counter()
            # 504x378 keeps 4:3 exactly and is a multiple of the 14 px patch.
            preds = ma.infer(
                [cv2.resize(b, (504, 378), interpolation=cv2.INTER_AREA) for b in bgrs]
            )
            times["mapany_window"].append(time.perf_counter() - t)
            for bgr, gt, pr in zip(bgrs, gts, preds, strict=True):
                h, w = bgr.shape[:2]
                md = cv2.resize(pr["depth_z"], (w, h), interpolation=cv2.INTER_LINEAR)
                mm = cv2.resize(pr["mask"].astype(np.uint8), (w, h), cv2.INTER_NEAREST) > 0
                t = time.perf_counter()
                dd = da2(bgr)
                times["da2"].append(time.perf_counter() - t)
                valid = (gt > 0.1) & mm
                s = float(np.median(md[valid] / dd[valid])) if valid.sum() > 500 else 1.0
                da = dd * s
                for key, est in (("da2", dd), ("da2_anchor", da), ("mapany", md)):
                    ok = valid & np.isfinite(est) & (est > 0)
                    pix[key].append(float(np.mean(np.abs(est[ok] - gt[ok]) / gt[ok])))
                t = time.perf_counter()
                r = det.predict(bgr, imgsz=640, device="mps", conf=0.35, verbose=False)[0]
                times["yolo"].append(time.perf_counter() - t)
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), strict=True):
                    label = r.names[int(c)]
                    g = inner(gt, b)
                    if g is None:
                        continue
                    row = {"seq": name, "label": label, "gt": g}
                    if label in PRIOR:
                        axis, size = PRIOR[label]
                        px = (b[3] - b[1]) if axis == "h" else (b[2] - b[0])
                        row["yolo_size"] = float(fx * size / max(px, 1.0))
                    row["da2"] = inner(dd, b)
                    row["da2_anchor"] = inner(da, b)
                    row["mapany"] = inner(md, b, mm)
                    rows.append(row)
        print(name, len(rows), flush=True)

    keys = ["yolo_size", "da2", "da2_anchor", "mapany"]
    same = [r for r in rows if all(r.get(k) for k in keys)]
    res = {
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "sequences": list(SEQS),
        "frames": N_FRAMES * len(SEQS),
        "boxes": len(rows),
        "all_boxes": {k: summarise(rows, k) for k in keys},
        "same_boxes": {k: summarise(same, k) for k in keys},
        "pixel_abs_rel_mean": {k: round(float(np.mean(v)), 3) for k, v in pix.items()},
        "latency_ms_median": {
            "yolo26n_640": round(1000 * float(np.median(times["yolo"][3:])), 1),
            "da2_518x392": round(1000 * float(np.median(times["da2"][3:])), 1),
            "mapanything_8_views": round(1000 * float(np.median(times["mapany_window"][1:])), 1),
        },
        "labels": {lb: sum(r["label"] == lb for r in rows) for lb in {r["label"] for r in rows}},
        "note": "machine shared with other agents' jobs; latencies are under load",
    }
    (OUT / "depth_vs_truth.json").write_text(json.dumps(res, indent=1))
    (OUT / "depth_vs_truth_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
