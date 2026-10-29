"""Metric depth scale of a recorded ARKit walk, from the phone's own travel.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python \
        scripts/scale_check.py WALK [--min-fwd 0.2]

Runs under the heavy-job lock (it loads the depth model). For every pair of tracked frames where
ARKit says the phone moved straight ahead by at least --min-fwd metres (yaw change under 8
degrees, sideways drift under half the forward travel), the centre patch of the raw depth
(before any scale) changes by the same distance, so scale = forward travel / (raw depth before -
raw depth after) in metres per raw unit. Prints and saves the median, quartiles and the pair
count, for all pairs and for pairs with 0.5 m or more of travel (a straight run at a wall gives
many), to bench/results/scale-check.json. The live belt's scale is metasight.glasses.toml's
depth_scale, clipped to 0.8 to 1.25 times itself by safety._geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_sector_labels as msl  # noqa: E402

from metasight.config import Settings  # noqa: E402
from metasight.safety import SafetyLoop  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_YAW_DEG, MAX_GAP = 8.0, 200  # frame seqs between the two frames of a pair
OUT = ROOT / "bench" / "results" / "scale-check.json"


def centre_patch_raw(loop: SafetyLoop, walk: Path, rows: list[dict], poses: dict) -> tuple:
    """Raw depth of the middle 20% of the image, and ground-plane pose, per sampled frame."""
    hor, raw = {}, {}
    for r in rows[::3]:
        h = msl._horizontal(poses[r["seq"]]) if r["seq"] in poses else None
        if h is None:
            continue
        d = loop.estimate_depth(cv2.imread(str(walk / "frames" / r["file"])))
        hh, ww = d.shape
        hor[r["seq"]] = h
        raw[r["seq"]] = float(
            np.median(d[int(0.4 * hh) : int(0.6 * hh), int(0.4 * ww) : int(0.6 * ww)])
        )
    return hor, raw


def pair_scales(hor: dict, raw: dict, min_fwd: float) -> list[tuple[int, int, float, float]]:
    out = []
    qs = sorted(raw)
    for a in qs:
        for b in qs:
            if not 0 < b - a <= MAX_GAP:
                continue
            (pa, fa, _), (pb, fb, _) = hor[a], hor[b]
            yaw = math.degrees(math.acos(np.clip(fa @ fb, -1, 1)))
            dp = pb - pa
            fwd, lat = float(dp @ fa), abs(float(dp @ np.array([fa[1], -fa[0]])))
            dr = raw[a] - raw[b]
            if abs(fwd) < min_fwd or yaw > MAX_YAW_DEG or lat > 0.5 * abs(fwd) or abs(dr) < 1e-3:
                continue
            out.append((a, b, fwd, fwd / dr))
    return out


def summary(pairs: list, floor: float) -> dict:
    k = np.array([p[3] for p in pairs if abs(p[2]) >= floor])
    if not len(k):
        return {"pairs": 0}
    q1, med, q3 = np.percentile(k, [25, 50, 75])
    return {
        "pairs": int(len(k)),
        "negative_pairs": int((k < 0).sum()),
        "median": round(float(med), 2),
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("--min-fwd", type=float, default=0.2)
    a = ap.parse_args()
    walk = ROOT / "data" / "walks" / a.walk
    cam = json.loads((walk / "camera.json").read_text())
    s = Settings.load(ROOT / "metasight.glasses.toml")
    s.hfov_deg = cam.get("hfov", s.hfov_deg)
    rows = [json.loads(line) for line in (walk / "frames.jsonl").read_text().splitlines()]
    hor, raw = centre_patch_raw(SafetyLoop(s), walk, rows, msl._poses(walk))
    pairs = pair_scales(hor, raw, a.min_fwd)
    res = {
        "walk": a.walk,
        "unit": "metres per raw depth unit (the scale the live belt should use)",
        "live_depth_scale": s.depth_scale,
        "frames": len(raw),
        f"travel_ge_{a.min_fwd}m": summary(pairs, a.min_fwd),
        "travel_ge_0.5m": summary(pairs, 0.5),
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
