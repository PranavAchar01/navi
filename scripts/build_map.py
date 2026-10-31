"""Build a house map offline from a recorded walkthrough, with the same engine the server runs.

python scripts/build_map.py data/walk30_720p.mp4 --out sessions/walk.npz [--seconds 30]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metasight.align import map_resolution  # noqa: E402
from metasight.config import Settings  # noqa: E402
from metasight.loops import Describer  # noqa: E402
from metasight.memory import Keyframe, MapAnythingRunner, MemoryEngine  # noqa: E402
from metasight.providers import make_finder  # noqa: E402
from metasight.safety import SafetyLoop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="sessions/offline.npz")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--fps", type=float, default=None)
    # Offline nobody waits, so windows can be longer than live: more shared views per stitch.
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    # Keep MapAnything's outputs per window on disk, so a stitching change rebuilds in minutes.
    ap.add_argument("--cache", action="store_true")
    # Keyframe geometry and window transforms, for trying placement changes without a rebuild.
    ap.add_argument("--dump", default=None)
    # Known lens for MapAnything (degrees across the frame); otherwise it guesses per window.
    ap.add_argument("--hfov", type=float, default=None)
    a = ap.parse_args()
    s = Settings.load()
    s.window, s.overlap = a.window or s.window, a.overlap or s.overlap
    fps = a.fps or s.keyframe_hz
    cap = cv2.VideoCapture(a.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / fps))
    safety = SafetyLoop(s)
    from ultralytics import YOLO

    runner = MapAnythingRunner(hfov_deg=a.hfov)
    if a.cache:
        from bench_map import CachedRunner

        runner = CachedRunner(Path(a.video).name, ROOT / "data" / "cache", hfov_deg=a.hfov)
    engine = MemoryEngine(
        s,
        runner,
        make_finder(s),
        YOLO(str(s.weights_dir / "yolo26n.pt")),
        describer=Describer(),  # without it no revisit is found and loops never close
    )
    kfs, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or i / src_fps > a.seconds:
            break
        if i % step == 0:
            w, h = map_resolution(frame.shape[1], frame.shape[0])
            small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            res = safety.process(small)
            kfs.append(
                Keyframe(len(kfs), time.time() + i / src_fps, small, res.depth, safety.scale)
            )
        i += 1
    print(f"{len(kfs)} keyframes", flush=True)
    per = s.window - s.overlap
    t0 = time.perf_counter()
    for k in range(0, len(kfs), per):
        chunk = kfs[k : k + per]
        if len(chunk) < 2:
            break
        ev = engine.process_window(chunk)
        st = next((e for e in ev if e["type"] == "memory_stats"), None)
        if st is None:  # every frame too blank to place: the engine skipped the window
            continue
        hint = next((e["ratio"] for e in ev if e["type"] == "scale_hint"), None)
        print(
            f"window {st['window']}: {st['views']} views {st['infer_ms']:.0f} ms infer, "
            f"{st['map_points']} pts, stitch scale {st['stitch_scale']:.3f}, scale hint {hint}",
            flush=True,
        )
        if hint:
            safety.apply_scale_hint(hint)
    print(f"total {time.perf_counter() - t0:.1f}s")
    for r in engine.loops:
        if "shift_m" in r:
            print("  loop", r)
    print(f"{sum(r.get('result') == 'closed' for r in engine.loops)} loops closed; pins:")
    for p in engine.pins.pins.values():
        print(" ", p.kind, p.label, [round(v, 2) for v in p.pos], p.near, f"seen {p.seen}")
    Path(a.out).parent.mkdir(exist_ok=True)
    engine.save(a.out)
    if a.dump:
        import pickle

        kfs_out = {q: (st.w, st.local, st.mask, st.cols, st.pose) for q, st in engine._kfs.items()}
        state = (kfs_out, engine._sims, engine._seq_rel, engine._seq_w, engine._loop_edges)
        Path(a.dump).write_bytes(pickle.dumps(state))
    pts = engine._map_pts
    print("extent", np.round(np.percentile(pts, 98, 0) - np.percentile(pts, 2, 0), 2))


if __name__ == "__main__":
    main()
