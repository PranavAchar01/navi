"""Render the binaural track for a simulated run: the walker's head pose track plus the scene
(the item being found and the obstacles) to a stereo WAV, and a timing note for the cut.

The pose track is a .jsonl or .csv with one row per frame:
    t                    seconds; the first row is WAV time 0 (put the video's frame 0 there)
    x, y, z              head position, metres, y up
    fx, fy, fz           forward vector, or yaw_deg (0 = -z, positive turns right) and
                         optional pitch_deg (positive looks up)
A 2D log (x, y, heading on the floor plane, no z) maps with --floor2d: floor x -> x, floor
y -> -z, head height --eye metres, heading radians counter-clockwise from +x.

The scene is a JSON file: {"target": [x, y, z], "hazards": [{"lo": [..], "hi": [..]}, ..]};
a hazard without "hi" is a point.

A simwalk sidecar (<clip>.pose.jsonl from the sim montage) needs no scene: its first line is a
header with the target and every object's oriented box, the rows carry x, y, z (z up, floor
frame) and a unit forward vector, and t = frame / fps from video frame 0. The target sounds as
the beacon; every other object ticks from the nearest face of its box (its yaw folded into an
axis-aligned box).

    PYTHONPATH=. .venv/bin/python scripts/render_sim_sound.py POSES [SCENE] --out OUT.wav
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from metasight import spatial_sound as ss

UP = np.array([0.0, 1.0, 0.0])


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open() as f:
            return [{k: float(v) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _pick(d: dict, *keys: str):
    return next(d[k] for k in keys if k in d)


def sidecar(path: Path, ticks_on_target: bool = False):
    """A simwalk <clip>.pose.jsonl: (t, pos, fwd, up, target, hazards, header)."""
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    head, rows = lines[0], lines[1:]
    t = np.array([r["t"] for r in rows], float)
    pos = np.array([[r["x"], r["y"], r["z"]] for r in rows], float)
    fwd = np.array([r["forward"] for r in rows], float)
    up = np.tile([0.0, 0.0, 1.0], (len(t), 1))
    objects = head.get("objects", [])
    tgt = head.get("target")
    if isinstance(tgt, dict):
        tgt = tgt.get("name", tgt.get("centre", tgt.get("center")))
    target, hazards = None, []
    for o in objects:
        c = np.asarray(_pick(o, "centre", "center"), float)
        hx, hy, hz = _pick(o, "half", "half_sizes", "half_size", "halfsize")
        yaw = math.radians(o.get("yaw_deg", 0.0))
        ex = abs(math.cos(yaw)) * hx + abs(math.sin(yaw)) * hy
        ey = abs(math.sin(yaw)) * hx + abs(math.cos(yaw)) * hy
        is_target = tgt is not None and tgt in (o.get("name"), o.get("label"))
        if is_target and target is None:
            target = c
            if not ticks_on_target:
                continue
        lo, hi = c - [ex, ey, hz], c + [ex, ey, hz]
        hazards.append(ss.Hazard(tuple(lo), tuple(hi)))
    if target is None and isinstance(tgt, list):
        target = np.asarray(tgt, float)
    return t, pos, fwd, up, target, hazards, head


def track(rows: list[dict], floor2d: bool, eye: float):
    t = np.array([r["t"] for r in rows], float)
    if floor2d:
        pos = np.array([[r["x"], eye, -r["y"]] for r in rows], float)
        h = np.array([r["heading"] for r in rows], float)
        fwd = np.stack([np.cos(h), np.zeros_like(h), -np.sin(h)], axis=1)
        return t, pos, fwd
    pos = np.array([[r["x"], r["y"], r["z"]] for r in rows], float)
    if "fx" in rows[0]:
        fwd = np.array([[r["fx"], r["fy"], r["fz"]] for r in rows], float)
    else:
        yaw = np.radians([r["yaw_deg"] for r in rows])
        pitch = np.radians([r.get("pitch_deg", 0.0) for r in rows])
        c = np.cos(pitch)
        fwd = np.stack([np.sin(yaw) * c, np.sin(pitch), -np.cos(yaw) * c], axis=1)
    return t, pos, fwd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("poses", type=Path)
    ap.add_argument("scene", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=int, default=48_000)
    ap.add_argument("--floor2d", action="store_true")
    ap.add_argument("--eye", type=float, default=1.6)
    ap.add_argument("--no-ticks", action="store_true", help="the beacon only")
    ap.add_argument("--pan", action="store_true", help="stereo pan instead of the KEMAR HRTF")
    a = ap.parse_args()

    rows = load_rows(a.poses)
    if "scenario" in rows[0] or a.scene is None:
        t, pos, fwd, up, target, hazards, _ = sidecar(a.poses)
    else:
        t, pos, fwd = track(rows, a.floor2d, a.eye)
        up = np.tile(UP, (len(t), 1))
        scene = json.loads(a.scene.read_text())
        target = np.asarray(scene["target"], float) if scene.get("target") else None
        hazards = [
            ss.Hazard(tuple(h["lo"]), tuple(h["hi"]) if h.get("hi") else None)
            for h in scene.get("hazards", [])
        ]
    if a.no_ticks:
        hazards = []
    method = "pan" if a.pan else "hrtf"
    x = ss.render(t, pos, fwd, up, target=target, hazards=hazards, method=method)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ss.write_wav(a.out, x, a.rate)

    events = ss.schedule(t, pos, fwd, up, target, hazards)
    first: dict[str, float] = {}
    counts: dict[str, int] = {}
    for e in events:
        first.setdefault(e.kind, round(float(e.t - t[0]), 3))
        counts[e.kind] = counts.get(e.kind, 0) + 1
    note = {
        "wav": str(a.out),
        "rate_hz": a.rate,
        "method": method,
        "duration_s": round(len(x) / ss.SAMPLE_RATE, 3),
        "t0": f"WAV time 0 = the first pose row (t = {t[0]:.3f} in the pose file)",
        "pose_rows": len(t),
        "pose_rate_hz": round((len(t) - 1) / (t[-1] - t[0]), 2) if len(t) > 1 else None,
        "first_s": first,
        "counts": counts,
        "events": [
            {
                "t": round(float(e.t - t[0]), 3),
                "kind": e.kind,
                "az_deg": round(e.where.az_deg, 1),
                "ground_m": None if math.isnan(e.where.ground_m) else round(e.where.ground_m, 2),
            }
            for e in events
        ],
    }
    timing = a.out.with_suffix(".timing.json")
    timing.write_text(json.dumps(note, indent=1))
    print(f"{a.out}: {note['duration_s']} s at {a.rate} Hz, peak {np.abs(x).max():.2f}")
    print(f"  first: {first}\n  counts: {counts}\n  {timing}")


if __name__ == "__main__":
    main()
