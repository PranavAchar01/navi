"""How often is the thing you ask for out of view? On Pranav's own ARKit walk.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python bench/compare/out_of_view.py

The walk's own pins (data/walks/20261028-135040-iphone/pins.json) give the object labels the
map remembered. YOLOE-26L (the detector that made those pins) is run on every 3rd frame with
the same labels. For each label, from its first sighting on, every sampled frame where the
detector does not see it is a moment when "where is my X" has no answer without memory:
a box-only detector can only say "not in view". Writes bench/results/compare/out_of_view.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WALK = ROOT / "data" / "walks" / "20261028-135040-iphone"
OUT = ROOT / "bench" / "results" / "compare"
OUT.mkdir(parents=True, exist_ok=True)
STEP, CONF, IMGSZ = 3, 0.25, 640


def main() -> None:
    from ultralytics import YOLO

    pins = json.loads((WALK / "pins.json").read_text())
    labels = sorted({p["label"] for p in pins if p["kind"] == "object" and p["label"] != "person"})
    frames = [json.loads(ln) for ln in open(WALK / "frames.jsonl")][::STEP]
    det = YOLO(str(ROOT / "yoloe-26l-seg.pt"))
    det.set_classes(labels)
    seen = np.zeros((len(frames), len(labels)), bool)
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        r = det.predict(str(WALK / "frames" / f["file"]), imgsz=IMGSZ, conf=CONF, verbose=False)
        for c in r[0].boxes.cls.cpu().numpy():
            seen[i, int(c)] = True
    took = time.perf_counter() - t0
    dur = (frames[-1]["ts"] - frames[0]["ts"]) / 1000
    per = {}
    for j, lb in enumerate(labels):
        hit = np.flatnonzero(seen[:, j])
        if not len(hit):
            per[lb] = {"sightings": 0}
            continue
        after = seen[hit[0] :, j]
        per[lb] = {
            "sightings": int(after.sum()),
            "first_s": round((frames[hit[0]]["ts"] - frames[0]["ts"]) / 1000, 1),
            "out_of_view_share": round(float(1 - after.mean()), 3),
        }
    shares = [v["out_of_view_share"] for v in per.values() if "out_of_view_share" in v]
    res = {
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "walk": WALK.name,
        "walk_s": round(dur, 1),
        "frames_scored": len(frames),
        "labels": len(labels),
        "labels_seen": len(shares),
        "mean_out_of_view_share": round(float(np.mean(shares)), 3),
        "median_out_of_view_share": round(float(np.median(shares)), 3),
        "labels_out_of_view_over_80pct": sum(s > 0.8 for s in shares),
        "detector": f"yoloe-26l-seg imgsz {IMGSZ} conf {CONF}",
        "detector_ms_per_frame": round(1000 * took / len(frames), 1),
        "per_label": per,
    }
    (OUT / "out_of_view.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_label"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
