"""Ground-truth cue labels for a recorded walk: the nearest obstacle per belt sector, per frame.

    .venv/bin/python scripts/make_sector_labels.py [walk ...]

For a walk with ARKit poses (feed "arkit"), the cached depth strips (bench/results/walk-strips.npz,
written by replay_false_buzz.py --extract) are placed in the world with each frame's pose and
collected on a 0.15 m grid. A cell is kept only when at least 4 frames saw it and the sightings
span at least 1 s: one-frame depth noise and flicker are not obstacles. Each frame's label is
then the nearest kept cell per sector from that frame's own pose, so the label reads the map the
whole walk agreed on, not the single frame the live scheme sees.

Labels are a consensus of the same depth model, not an independent measurement: they remove
frame noise, not the model's scale error or ARKit drift (docs/haptics-sim.md, "Ground-truth
labels"). Writes bench/labels/<walk>-sectors.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from metasight import pan
from metasight.server import arkit_pose

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "bench" / "results" / "walk-strips.npz"
OUT = ROOT / "bench" / "labels"
STEP = 2  # the cache holds every 2nd frame
CELL = 0.15  # m
MIN_SEEN, MIN_SPAN_S = 4, 1.0
RANGE_M = 3.0  # the live depth's trusted range; labels past it read "none"
MIN_RANGE_M = 0.15
SECTORS = 8


def _poses(walk: Path) -> dict[int, np.ndarray]:
    """Camera-to-world matrix per frame seq, tracking frames only."""
    out = {}
    for line in (walk / "poses.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "arpose" and r.get("track") == "tracking":
            out[r["seq"]] = arkit_pose(r["m"])
    return out


def _horizontal(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Position and unit forward / right on the ground plane (x, z), world y up."""
    fwd, right = pose[[0, 2], 2], pose[[0, 2], 0]
    if np.linalg.norm(fwd) < 0.3:  # looking at the floor or ceiling: no heading
        return None
    return pose[[0, 2], 3], fwd / np.linalg.norm(fwd), right / np.linalg.norm(right)


def _nearest(dist: np.ndarray, near: np.ndarray, mask: np.ndarray) -> float | None:
    sel = dist[near & mask]
    return round(float(sel.min()), 2) if sel.size else None


def kept_cells(walk: Path, d: np.lib.npyio.NpzFile) -> np.ndarray:
    """Centres (x, z) of the grid cells the walk agreed on, in the ARKit world frame."""
    name = walk.name
    ts, strips, hfov = d[f"{name}/ts"], d[f"{name}/strips"], float(d[f"{name}/hfov"][0])
    poses = _poses(walk)
    rows = [json.loads(line) for line in (walk / "frames.jsonl").read_text().splitlines()]
    seqs = [r["seq"] for r in rows[::STEP]][: len(ts)]
    az = np.radians(pan.strip_bearings(strips.shape[1], hfov))
    cells: dict[tuple[int, int], list[float]] = {}  # cell -> sighting times
    for t, seq, st in zip(ts, seqs, strips, strict=True):
        pose = poses.get(seq)
        h = _horizontal(pose) if pose is not None else None
        if h is None:
            continue
        pos, fwd, right = h
        for a, dist in zip(az, st, strict=True):
            if not (MIN_RANGE_M <= dist <= RANGE_M):
                continue
            p = pos + dist * (fwd + math.tan(a) * right)
            cells.setdefault((int(p[0] // CELL), int(p[1] // CELL)), []).append(float(t))
    return np.array(
        [
            ((i + 0.5) * CELL, (k + 0.5) * CELL)
            for (i, k), seen in cells.items()
            if len(seen) >= MIN_SEEN and max(seen) - min(seen) >= MIN_SPAN_S
        ]
    )


def label_walk(walk: Path, d: np.lib.npyio.NpzFile) -> dict:
    name = walk.name
    ts, hfov = d[f"{name}/ts"], float(d[f"{name}/hfov"][0])
    poses = _poses(walk)
    rows = [json.loads(line) for line in (walk / "frames.jsonl").read_text().splitlines()]
    seqs = [r["seq"] for r in rows[::STEP]][: len(ts)]
    kept = kept_cells(walk, d)

    half = math.radians(hfov) / 2
    frames = []
    for t, seq in zip(ts, seqs, strict=True):
        pose = poses.get(seq)
        h = _horizontal(pose) if pose is not None else None
        if h is None or not len(kept):
            frames.append({"t": round(float(t - ts[0]), 2), "seq": seq, "tracked": False})
            continue
        pos, fwd, right = h
        rel = kept - pos
        dist = np.hypot(rel[:, 0], rel[:, 1])
        bearing = np.degrees(np.arctan2(rel @ right, rel @ fwd))
        near = dist <= RANGE_M

        sec = [
            _nearest(dist, near, abs((bearing - 45 * i + 180) % 360 - 180) <= 22.5)
            for i in range(SECTORS)
        ]
        b = np.radians(bearing)
        edges = [-half, -half / 3, half / 3, half]  # the camera's left, centre and right thirds
        fov = [
            _nearest(dist, near, (b >= lo) & (b < hi))
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        frames.append(
            {
                "t": round(float(t - ts[0]), 2),
                "seq": seq,
                "tracked": True,
                "sectors": sec,
                "fov_thirds": fov,
            }  # fmt: skip
        )
    tracked = [f for f in frames if f["tracked"]]
    return {
        "walk": name,
        "feed": "arkit",
        "hfov_deg": hfov,
        "range_m": RANGE_M,
        "cell_m": CELL,
        "kept_cells": int(len(kept)),
        "frames_total": len(frames),
        "frames_tracked": len(tracked),
        "sector_order": "0 ahead, 1 front-right ... 7 front-left, 45 deg each",
        "fov_thirds_order": "camera left, centre, right (what the 3 live front cues read)",
        "null": "no kept obstacle within range_m",
        "frames": frames,
    }


def main(names: list[str] | None = None) -> None:
    d = np.load(CACHE)
    names = names or sorted({k.split("/")[0] for k in d.files})
    OUT.mkdir(exist_ok=True)
    for name in names:
        walk = ROOT / "data" / "walks" / name
        cam = json.loads((walk / "camera.json").read_text())
        if cam.get("feed") != "arkit":
            print(name, f"feed {cam.get('feed')!r}: no phone poses, skipped")
            continue
        lab = label_walk(walk, d)
        (OUT / f"{name}-sectors.json").write_text(json.dumps(lab, separators=(",", ":")))
        print(
            name, lab["frames_tracked"], "tracked of", lab["frames_total"],
            "frames;", lab["kept_cells"], "cells kept",
        )  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1:])
