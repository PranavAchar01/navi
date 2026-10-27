"""Shared timing, memory sampling and point-cloud rendering for the benchmarks."""

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "f720_all"
RESULTS = ROOT / "bench" / "results"
SHOTS = ROOT / "shots"
RESULTS.mkdir(parents=True, exist_ok=True)
SHOTS.mkdir(parents=True, exist_ok=True)


def frame_paths(fps: float, start_s: float = 0.0, dur_s: float = 30.0) -> list[str]:
    """Subsample the 30 fps frame folder at `fps` keyframes per second."""
    allf = sorted(FRAMES.glob("*.jpg"))
    step = 30.0 / fps
    idx = [round(start_s * 30 + i * step) for i in range(int(dur_s * fps))]
    return [str(allf[i]) for i in idx if i < len(allf)]


class MemSampler:
    """Polls MPS driver memory and process RSS; keeps the peaks."""

    def __init__(self, period: float = 0.02):
        self.period = period
        self.peak_mps = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._proc = psutil.Process(os.getpid())

    def _run(self):
        while not self._stop.is_set():
            self.peak_mps = max(self.peak_mps, torch.mps.driver_allocated_memory())
            self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
            time.sleep(self.period)

    def __enter__(self):
        torch.mps.empty_cache()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def timed(fn, warmup: int = 1, reps: int = 3):
    """Run fn warmup+reps times, synchronising MPS; return (median_s, all_s, last_out, mem)."""
    out = None
    for _ in range(warmup):
        out = fn()
        torch.mps.synchronize()
    times = []
    with MemSampler() as mem:
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.median(times)), times, out, mem


def save_result(name: str, rec: dict):
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec), flush=True)


def render_cloud(
    pts: np.ndarray,
    cols: np.ndarray,
    title: str,
    out: Path,
    cams: np.ndarray | None = None,
    max_pts: int = 250_000,
):
    """Two views of a coloured point cloud: oblique 3D and top-down, OpenCV camera frame
    (x right, y down, z forward)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = pts.reshape(-1, 3)
    cols = cols.reshape(-1, 3)
    ok = np.isfinite(pts).all(1)
    pts, cols = pts[ok], cols[ok]
    # Trim outliers so the room fills the frame.
    lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
    keep = ((pts >= lo) & (pts <= hi)).all(1)
    pts, cols = pts[keep], cols[keep]
    if len(pts) > max_pts:
        sel = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts, cols = pts[sel], cols[sel]
    cols = np.clip(cols, 0, 1)
    fig = plt.figure(figsize=(16, 7), dpi=110)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols, s=0.3, linewidths=0)
    if cams is not None:
        ax.plot(cams[:, 0], cams[:, 2], -cams[:, 1], "r-", lw=2)
    ax.view_init(elev=28, azim=-60)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("up (m)")
    ax.set_title("oblique")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(pts[:, 0], pts[:, 2], c=cols, s=0.2, linewidths=0)
    if cams is not None:
        ax2.plot(cams[:, 0], cams[:, 2], "r-", lw=2, label="camera path")
        ax2.legend(loc="upper right")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_title("top-down")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
