"""The CPU-only stages of the live path, at normal priority, with no models loaded.

    .venv/bin/python bench/perf_cpu.py data/walks/<id> depth.npz [--profile]

`depth.npz` comes from `bench/perf_stages.py --dump`: the depth network's raw output on the
same walk, so the floor geometry and hazard outlines run on real frames. The live service runs
as an interactive process, so this is the number it sees; perf_stages runs in the background
band (E-cores) and reads 2 to 4x slower on the same code.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metasight.align import map_resolution  # noqa: E402
from metasight.heading import Heading  # noqa: E402
from metasight.safety import analyse, hazards  # noqa: E402


def bench(fn, items, reps=1):
    ts = []
    for _ in range(reps):
        for x in items:
            t = time.perf_counter()
            fn(x)
            ts.append(1000 * (time.perf_counter() - t))
    ts.sort()
    return ts[len(ts) // 2], ts[int(0.95 * len(ts))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("depth")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    z = np.load(a.depth)
    depths = [d.astype(np.float32) for d in z["depth"]]
    hfov, eye_h = float(z["hfov"]), float(z["eye_h"])
    walk = Path(a.walk)
    rows = [json.loads(x) for x in (walk / "frames.jsonl").read_text().splitlines() if x]
    jpegs = [(walk / "frames" / r["file"]).read_bytes() for r in rows[: len(depths)]]
    bgrs = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in jpegs]
    w, h = map_resolution(bgrs[0].shape[1], bgrs[0].shape[0])
    smalls = [cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA) for b in bgrs]

    # The plane each frame would have carried from the frame before, as the live loop does.
    scenes, planes, prev = [], [], None
    for d in depths:
        sc, pl = analyse(d, 1.0, 1.0, hfov, eye_h, prev)
        prev = pl or prev
        scenes.append(sc)
        planes.append(prev)
    with_prev = list(zip(depths, [None, *planes[:-1]], strict=True))
    found = sum(s.floor for s in scenes)

    rows_out = {
        "jpeg decode": bench(lambda j: cv2.imdecode(np.frombuffer(j, np.uint8), 1), jpegs),
        "resize to 518": bench(lambda b: cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA), bgrs),
        "heading gray": bench(Heading(hfov).shift_gray, smalls),
        "geometry (analyse)": bench(
            lambda dp: analyse(dp[0], 1.0, 1.0, hfov, eye_h, dp[1]), with_prev
        ),
        "hazards": bench(lambda s: hazards(s.grid, s.corridor), scenes),
    }
    print(f"{len(depths)} frames, floor found on {found}")
    print(f"{'stage':24s} {'p50 ms':>8s} {'p95 ms':>8s}")
    for k, (p50, p95) in rows_out.items():
        print(f"{k:24s} {p50:8.2f} {p95:8.2f}")
    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
        for dp in with_prev:
            analyse(dp[0], 1.0, 1.0, hfov, eye_h, dp[1])
        for s in scenes:
            hazards(s.grid, s.corridor)
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(18)


if __name__ == "__main__":
    main()
