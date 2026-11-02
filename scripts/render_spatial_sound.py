"""Render the spatial sound for a scripted find-the-chair walk to WAV, to hear it on headphones.

The wearer stands facing away from the chair, turns toward it (the beacon moves from behind to
the front and becomes the two-note figure), walks past a desk corner (ticks on the left) and
arrives (the chime). One file per method: HRTF and the plain pan fallback.

Run: PYTHONPATH=. .venv/bin/python scripts/render_spatial_sound.py
"""

import math

import numpy as np

from metasight import spatial_sound as ss
from metasight.config import ROOT

OUT = ROOT / "bench" / "results"
UP = np.array([0.0, 1.0, 0.0])  # ARKit world: y up


def track(dt: float = 0.02):
    """Times, positions and forward vectors: 3 s turning 180 degrees, then 5 s walking."""
    t, pos, fwd = [], [], []
    start = np.array([0.0, 1.6, 0.0])
    for k in range(int(3 / dt)):  # facing +z (away), turning left to face -z
        yaw = math.pi * min(1.0, k * dt / 2.5)
        t.append(k * dt)
        pos.append(start)
        fwd.append([-math.sin(yaw), 0.0, math.cos(yaw)])
    for k in range(int(5 / dt)):
        s = min(1.0, k * dt / 5)
        t.append(3 + k * dt)
        pos.append(start + np.array([0.0, 0.0, -3.4 * s]))
        fwd.append([0.0, 0.0, -1.0])
    return np.array(t), np.array(pos), np.array(fwd)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t, pos, fwd = track()
    up = np.tile(UP, (len(t), 1))
    chair = np.array([0.3, 0.45, -4.0])
    desk = ss.Hazard((-1.6, 0.0, -2.4), (-0.45, 0.75, -1.6))
    for method in ("hrtf", "pan"):
        st = ss.render(t, pos, fwd, up, target=chair, hazards=[desk], method=method)
        path = OUT / f"spatial-sound-find-chair-{method}.wav"
        ss.write_wav(path, st)
        print(path, f"{len(st) / ss.SAMPLE_RATE:.1f} s, peak {np.abs(st).max():.2f}")
    for e in ss.schedule(t, pos, fwd, up, chair, [desk]):
        print(f"{e.t:5.2f} s  {e.kind:15s} az {e.where.az_deg:7.1f}  {e.where.ground_m:.2f} m")


if __name__ == "__main__":
    main()
