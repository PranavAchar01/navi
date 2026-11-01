"""Score a leveled map without ground truth, so map changes are compared by number, not by eye.

python scripts/map_quality.py sessions/a-level.npz [more.npz ...]

head_cm   spread of the walker's head height (one floor: a flat path; tilt or drift raises it)
wall_ent  entropy (bits) of the top-down wall band at 5 cm: a wall seen twice at two angles
          spreads over more cells, so ghosting raises it
area_m2   top-down area the wall band covers (cells with 3+ points); ghosts and drift inflate it
floor_m   eye to floor, the outermost dense height bin below the path (glasses: about 1.5-1.7)
path_m    length of the walker's path; a map that shrinks the walk scores a small area too, so
          compare area only between maps whose floor_m and path_m agree
wall_ent and area_m2 are measured after scaling the map to put the floor EYE_M below the path,
so a map that shrank or grew does not win on them
"""

import sys

import numpy as np

CELL = 0.05
EYE_M = 1.6  # only to compare maps at one scale; not used to build anything


def score(path: str) -> dict:
    d = np.load(path)
    p, t = d["pts"], d["trail"][:, :3]
    head = float(np.median(t[:, 1]))
    hist, edges = np.histogram(p[:, 1] - head, bins=np.arange(-4, 4.01, 0.1))
    dense = np.where(hist >= 0.5 * hist.max())[0]
    floor = -float(edges[dense[0]])
    k = EYE_M / max(floor, 0.1)
    q = p[(p[:, 1] > head - 0.85 * floor) & (p[:, 1] < head + 0.5 * floor)] * k
    cells = np.floor(q[:, [0, 2]] / CELL).astype(np.int64)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    f = counts / counts.sum()
    return {
        "head_cm": round(100 * float(np.std(t[:, 1])), 1),
        "wall_ent": round(float(-(f * np.log2(f)).sum()), 2),
        "area_m2": round(float((counts >= 3).sum()) * CELL * CELL, 1),
        "floor_m": round(floor, 2),
        "path_m": round(float(np.linalg.norm(np.diff(t[:, [0, 2]], axis=0), axis=1).sum()), 1),
        "points": int(len(p)),
    }


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print(a, score(a))
