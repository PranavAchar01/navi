"""Per-stage timing of the live safety path on one recorded walk, in this process.

    .venv/bin/python bench/perf_stages.py data/walks/<id> --out bench/results/perf/x.json

Runs the same steps as Server._process and frame_loop, in the same order, on the walk's own
JPEGs, and times each one: decode, resizes, heading, sharpness, depth network, floor geometry
(`analyse`, called once or twice a frame), segmentation, people, plan, message build and the
JSON encode the hub does. Model warm-up frames are dropped. Prints p50/p95 per stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metasight import safety as safetymod  # noqa: E402
from metasight.align import map_resolution  # noqa: E402
from metasight.belt import plan  # noqa: E402
from metasight.config import Settings  # noqa: E402
from metasight.conversation import Talk  # noqa: E402
from metasight.geometry import CAMERA_SECTORS, SECTORS, merge_sectors  # noqa: E402
from metasight.heading import Heading  # noqa: E402
from metasight.server import FINDER_LONG_SIDE, _finite  # noqa: E402

T: dict[str, list[float]] = defaultdict(list)


def timed(name, fn):
    def run(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name].append(1000 * (time.perf_counter() - t))

    return run


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--config", default=str(ROOT / "metasight.glasses.toml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="save the raw depth maps here (.npz)")
    a = ap.parse_args()

    s = Settings.load(Path(a.config))
    walk = Path(a.walk)
    rows = [json.loads(x) for x in (walk / "frames.jsonl").read_text().splitlines() if x]
    rows = rows[: a.frames + a.warmup]

    safetymod.analyse = timed("geometry (analyse)", safetymod.analyse)
    loop = safetymod.SafetyLoop(s)
    loop.estimate_depth = timed("depth net", loop.estimate_depth)
    raws: list[np.ndarray] = []
    if a.dump:
        net = loop.estimate_depth

        def keep(bgr):
            raws.append(net(bgr).astype(np.float16))
            return raws[-1].astype(np.float32)

        loop.estimate_depth = keep
    if loop._seg is not None:
        loop.segment = timed("segment (YOLOE + masks)", loop.segment)
    heading, talk = Heading(s.hfov_deg), Talk(s.hfov_deg)

    frame_ms: list[float] = []
    for i, r in enumerate(rows):
        if i == a.warmup:
            T.clear()
            frame_ms.clear()
        jpeg = (walk / "frames" / r["file"]).read_bytes()
        t0 = time.perf_counter()
        bgr = timed("jpeg decode", cv2.imdecode)(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        w, h = map_resolution(bgr.shape[1], bgr.shape[0])
        small = timed("resize to 518", cv2.resize)(bgr, (w, h), interpolation=cv2.INTER_AREA)
        gray = timed("heading gray", heading.shift_gray)(small)
        timed("sharpness", lambda g: float(cv2.Laplacian(g, cv2.CV_32F).var()))(gray)
        k = FINDER_LONG_SIDE / max(bgr.shape[:2])
        if k < 1:
            timed("resize to 960", cv2.resize)(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        res = timed("safety.process (all)", loop.process)(small)
        now = time.time()
        timed("heading turn", heading.image_turn)(gray, now)

        t = time.perf_counter()
        merged = merge_sectors(res.sectors, [math.inf] * 8)
        sc = res.scene
        known = [n in CAMERA_SECTORS for n in SECTORS]
        cmd, flags = plan(merged, known, sc.path_m, res.ttc_s, sc.drop_m)
        T["plan + sectors"].append(1000 * (time.perf_counter() - t))
        people = timed("people (Talk.frame)", talk.frame)(now, res.obstacles, small, heading)
        t = time.perf_counter()
        msg = {
            "type": "safety",
            "sectors": [_finite(d) for d in merged],
            "hazards": safetymod.hazards(sc.grid, sc.corridor),
            "floor": sc.floor,
            "objects": [
                {"label": o.label, "poly": o.poly, "box": [round(v, 4) for v in o.box]}
                for o in res.objects
            ],
            "ms": {k: round(v, 1) for k, v in res.timings_ms.items()},
        }
        T["message build (hazards)"].append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        text = json.dumps(msg, separators=(",", ":"), default=lambda o: o.tolist())
        json.dumps(people, separators=(",", ":"), default=lambda o: o.tolist())
        T["json encode"].append(1000 * (time.perf_counter() - t))
        T["_bytes safety msg"].append(len(text))
        frame_ms.append(1000 * (time.perf_counter() - t0))

    out = {}
    print(f"{len(frame_ms)} frames from {walk.name}, pid {os.getpid()}")
    print(f"{'stage':32s} {'p50 ms':>8s} {'p95 ms':>8s} {'mean':>8s}")
    for name, xs in [*T.items(), ("frame total (server side)", frame_ms)]:
        p50, p95, mean = pct(xs, 0.5), pct(xs, 0.95), sum(xs) / len(xs)
        out[name] = {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "mean": round(mean, 3),
            "n": len(xs),
        }
        print(f"{name:32s} {p50:8.2f} {p95:8.2f} {mean:8.2f}  (n={len(xs)})")
    if a.dump:
        np.savez_compressed(a.dump, depth=np.stack(raws), hfov=s.hfov_deg, eye_h=s.eye_height_m)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"walk": walk.name, "stages": out}, indent=1))


if __name__ == "__main__":
    main()
