"""Level a map built without gravity (glasses walks): the walker's head stays at one height, so
the plane through the walk trail is horizontal. Rotates the map so that plane's normal is +y,
with the floor (the dense side) below, and reports eye height and ceiling as a scale check.

python scripts/level_map.py sessions/house.npz --out sessions/house-level.npz
"""

import argparse

import numpy as np


def level_rotation(trail: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Rotation taking the trail plane's normal to +y, signed so most points are below it."""
    m = trail.mean(0)
    n = np.linalg.svd(trail - m)[2][2]
    if np.median((pts - m) @ n) > 0:
        n = -n
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(n, y), float(n @ y)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = dict(np.load(a.map))
    trail = d["trail"][:, :3]
    rot = level_rotation(trail, d["pts"])
    tilt = np.degrees(np.arccos(np.clip(rot[1, 1], -1, 1)))
    d["pts"] = (d["pts"] @ rot.T).astype(np.float32)
    d["trail"][:, :3] = trail @ rot.T
    pose = d["pose"].copy()
    pose[:3, :3], pose[:3, 3] = rot @ pose[:3, :3], rot @ pose[:3, 3]
    d["pose"] = pose
    head = float(np.median(d["trail"][:, 1]))
    hist, edges = np.histogram(d["pts"][:, 1] - head, bins=np.arange(-3, 3.01, 0.1))
    # Floor and ceiling bound the points: the outermost bins still at half the peak density.
    dense = np.where(hist >= 0.5 * hist.max())[0]
    floor, ceiling = float(edges[dense[0]]), float(edges[dense[-1] + 1])
    print(f"tilt removed {tilt:.1f} deg; head height spread {np.std(d['trail'][:, 1]):.2f} m")
    print(f"eye to floor {-floor:.2f} m, eye to ceiling ~{ceiling:.2f} m")
    np.savez_compressed(a.out, **d)


if __name__ == "__main__":
    main()
