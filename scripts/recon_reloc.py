"""Re-register frames the dense run misplaced, and frames it never saw, against the room map.

The trusted frames (recon.trusted: camera within 15 cm of both time neighbours) are the map's
references. Each of them gives SIFT keypoints lifted onto the mesh through its own camera. A
frame to place finds its three nearest references by DINO image vector, matches SIFT against
them, and is placed by PnP-RANSAC on the lifted points. Placed frames are then checked against
their own walk: a pose more than 30 cm from the median of its time neighbours is dropped. The
result is a cams file like the dense run's, with "src": "dense" or "pnp".

    PYTHONPATH=. .venv/bin/python scripts/recon_reloc.py --mesh data/rooms/bedroom/v002 --tag s1
        --out data/rooms/bedroom/recon-v2/v2.1/cams.jsonl [--walks W ...] [--every 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from recon_score import embed, feats, k_full, lift, match  # noqa: E402
from recon_v2 import WALKS, load_cams, scene_of, ts_of  # noqa: E402

NEW_WALKS = [
    "20261029-182525-glasses",
    "20261102-023733-glasses",
    "20261102-025349-glasses",
    "20261102-030055-glasses",
]
MIN_INLIERS, MIN_SHARE = 40, 0.25
NEIGHBOUR_M = 0.30


class Refs:
    """Trusted frames with their DINO vectors and, on first use, their lifted keypoints."""

    def __init__(self, rows: list[dict], scene, cache: Path) -> None:
        self.rows, self.scene, self.lifted = rows, scene, {}
        paths = [WALKS / r["walk"] / "frames" / r["file"] for r in rows]
        if cache.exists() and len(np.load(cache)["e"]) == len(rows):
            self.e = np.load(cache)["e"]
        else:
            self.e = embed(paths)
            np.savez(cache, e=self.e)
        self.paths = paths

    def points(self, j: int):
        if j not in self.lifted:
            kp, d = feats(self.paths[j])
            uv = np.array([k.pt for k in kp]) if kp else np.zeros((0, 2))
            p = lift(self.scene, self.rows[j], uv) if len(uv) else np.zeros((0, 3))
            ok = np.isfinite(p).all(1)
            self.lifted[j] = (d[ok] if d is not None else None, p[ok])
            if len(self.lifted) > 4000:
                self.lifted.pop(next(iter(self.lifted)))
        return self.lifted[j]


def place(path: Path, e: np.ndarray, k: np.ndarray, refs: Refs) -> dict | None:
    kq, dq = feats(path)
    if dq is None:
        return None
    P, U = [], []
    for j in np.argsort(-(refs.e @ e))[:3]:
        dr, pr = refs.points(int(j))
        mm = match(dr, dq)
        if len(mm) < 12:
            continue
        P.append(pr[[a for a, _ in mm]])
        U.append(np.array([kq[b].pt for _, b in mm]))
    if not P or sum(len(p) for p in P) < MIN_INLIERS:
        return None
    P, U = np.concatenate(P).astype(np.float64), np.concatenate(U).astype(np.float64)
    ok, rv, tv, idx = cv2.solvePnPRansac(
        P, U, k, None, reprojectionError=6.0, iterationsCount=2000, confidence=0.999
    )
    if not ok or idx is None or len(idx) < MIN_INLIERS or len(idx) / len(P) < MIN_SHARE:
        return None
    idx = idx[:, 0]
    ok, rv, tv = cv2.solvePnP(P[idx], U[idx], k, None, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(P[idx], rv, tv, k, None)
    rms = float(np.sqrt(np.mean(np.sum((proj[:, 0] - U[idx]) ** 2, 1))))
    r, _ = cv2.Rodrigues(rv)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = r.T, (-r.T @ tv).ravel()
    return {"pose": pose, "inliers": int(len(idx)), "rms": rms}


def consistent(pos: np.ndarray, win: int = 5) -> np.ndarray:
    """Poses within NEIGHBOUR_M of the median of their time neighbours (itself left out)."""
    n = len(pos)
    ok = np.zeros(n, bool)
    for i in range(n):
        nb = np.r_[pos[max(0, i - win) : i], pos[i + 1 : i + 1 + win]]
        ok[i] = len(nb) >= 2 and np.linalg.norm(pos[i] - np.median(nb, 0)) < NEIGHBOUR_M
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--walks", nargs="*", help="only these walks (default: all broken + new)")
    ap.add_argument("--every", type=int, default=2, help="place every n-th frame of a walk")
    a = ap.parse_args()
    import open3d as o3d

    mesh_dir = Path(a.mesh)
    cams = load_cams(mesh_dir, a.tag)
    scene = scene_of(o3d.io.read_triangle_mesh(str(mesh_dir / f"mesh-{a.tag}.ply")))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trusted = [r for rows in cams.values() for r in rows if r["trusted"]]
    refs = Refs(trusted, scene, out.parent / "ref_dino.npz")
    k_glasses = np.median([r["k"] for r in trusted if "glasses" in r["walk"]], 0)

    jobs: dict[str, list[dict]] = {}
    for w, rows in cams.items():
        bad = [r for r in rows if not r["trusted"]][:: a.every]
        if bad:
            jobs[w] = [{"walk": w, "file": r["file"], "k": r["k"]} for r in bad]
    for w in NEW_WALKS:
        fs = sorted((WALKS / w / "frames").glob("*.jpg"), key=lambda p: ts_of(p.name))[:: a.every]
        jobs[w] = [{"walk": w, "file": f.name, "k": k_glasses} for f in fs]
    if a.walks:
        jobs = {w: v for w, v in jobs.items() if w in a.walks}

    rows_out = [
        {
            "walk": r["walk"],
            "file": r["file"],
            "pose": r["pose"].tolist(),
            "k": r["k"].tolist(),
            "src": "dense",
        }
        for r in trusted
    ]
    stats = {}
    for w, todo in jobs.items():
        t0, placed = time.time(), []
        for i in range(0, len(todo), 64):
            chunk = todo[i : i + 64]
            paths = [WALKS / w / "frames" / q["file"] for q in chunk]
            for q, p, e in zip(chunk, paths, embed(paths), strict=True):
                r = place(p, e, k_full(np.asarray(q["k"])), refs)
                if r:
                    placed.append({**q, **r})
            print(
                f"{w} {i + len(chunk)}/{len(todo)} placed {len(placed)} {time.time() - t0:.0f}s",
                flush=True,
            )
        placed.sort(key=lambda r: ts_of(r["file"]))
        ok = consistent(np.array([r["pose"][:3, 3] for r in placed])) if placed else []
        kept = [r for r, g in zip(placed, ok, strict=True) if g]
        stats[w] = {
            "tried": len(todo),
            "placed": len(placed),
            "kept": len(kept),
            "rms_median": round(float(np.median([r["rms"] for r in kept])), 2) if kept else None,
        }
        print(w, stats[w], flush=True)
        # The pose is in full-size pixels' camera; k is stored at the dense run's size.
        rows_out += [
            {
                "walk": r["walk"],
                "file": r["file"],
                "pose": r["pose"].tolist(),
                "k": np.asarray(r["k"]).tolist(),
                "src": "pnp",
                "inliers": r["inliers"],
                "rms": round(r["rms"], 2),
            }
            for r in kept
        ]
        with out.open("w") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
        (out.parent / "reloc.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    import shutil

    from room_watch import take_slot

    slot = take_slot()
    try:
        main()
    finally:
        shutil.rmtree(slot, ignore_errors=True)
