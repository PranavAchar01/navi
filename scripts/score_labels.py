"""Score the belt schemes on a recorded walk against its ground-truth cue labels.

    PYTHONPATH=. .venv/bin/python scripts/score_labels.py

Labels come from make_sector_labels.py (bench/labels/<walk>-sectors.json). A frame is
"cue on" when any motor is felt (intensity 26/255 or more, the skin model's floor). It is
"obstacle near" when the nearest labelled obstacle in the camera's view is under a threshold.
False buzz = cue on, nothing near (events: runs of such frames, per minute). Missed buzz =
obstacle near, no cue on. The cue only sees the camera's view, so only the label's view-thirds
are used. Labels are not exact (docs/haptics-sim.md), so every number is also given with the
labels scaled 0.84x and 1.16x, the depth model's 16% scale error after the map's correction.
Missed frames are split by whether the frame's own depth shows something inside 2 m: if not,
only the map (an earlier look) knows the obstacle, and the reflex cannot buzz for it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from replay_false_buzz import CACHE, commands  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OFF_M, SLOW_M = 2.0, 1.0  # a cue is due inside 2 m; inside 1 m it should be urgent
SCALES = (0.84, 1.0, 1.16)
LIVE_SCALE, TRUE_SCALE_GUESS = 0.42, 0.75  # live depth_scale median on 135040; ARKit estimate
SCHEMES = ("sector", "panned", "panned+hold", "panned+track", "panned+track+hold")


def _nearest(frame: dict) -> float:
    """Nearest labelled obstacle in the camera's view; 9 m when none or untracked."""
    if not frame["tracked"]:
        return 9.0
    return min((v for v in frame["fov_thirds"] if v is not None), default=9.0)


def runs(flag: np.ndarray) -> int:
    return int(np.sum(flag[1:] & ~flag[:-1]) + (1 if len(flag) and flag[0] else 0))


def score(on: np.ndarray, near_m: np.ndarray, thr: float, minutes: float, kind: str) -> dict:
    near = near_m < thr
    bad = (on & ~near) if kind == "false" else (~on & near)
    base = on if kind == "false" else near
    return {
        "events_per_min": round(runs(bad) / minutes, 1),
        "frames_pct": round(100 * float(bad.sum()) / max(1, int(base.sum())), 1),
    }


def score_one(d, lab: dict) -> dict:
    w = lab["walk"]
    ts = d[f"{w}/ts"]
    keep = np.array([f["tracked"] for f in lab["frames"]])
    near_m = np.array([_nearest(f) for f in lab["frames"]])
    minutes = float((ts[keep][-1] - ts[keep][0]) / 60)  # tracked span; gaps are rare
    strips = d[f"{w}/strips"]
    seen_now = strips.min(axis=1) < OFF_M  # the frame's own depth shows something inside 2 m
    res = {"minutes": round(minutes, 2), "frames": int(keep.sum())}
    for scheme in SCHEMES:
        on = commands(
            d[f"{w}/strips"], d[f"{w}/path"], d[f"{w}/ttc"], float(d[f"{w}/hfov"][0]),
            scheme, 2, 26,
        ).any(axis=1)  # fmt: skip
        row = {}
        for s in SCALES:
            nm = near_m * s
            row[f"x{s}"] = {
                "false_buzz": score(on[keep], nm[keep], OFF_M, minutes, "false"),
                "missed_inside_2m": score(on[keep], nm[keep], OFF_M, minutes, "miss"),
                "missed_inside_1m": score(on[keep], nm[keep], SLOW_M, minutes, "miss"),
            }  # fmt: skip
        miss = keep & (near_m < OFF_M) & ~on
        row["missed_inside_2m_frames"] = {
            "total": int(miss.sum()),
            "camera_sees_it_now": int((miss & seen_now).sum()),
            "not_in_current_frame": int((miss & ~seen_now).sum()),
        }
        row["cue_on_pct"] = round(100 * float(on[keep].mean()), 1)
        res[scheme] = row
    res["sensitivity_if_true_scale_0.75"] = _sensitivity(d, w, keep, near_m, minutes)
    res["near_pct"] = {
        "inside_2m": round(100 * float((near_m[keep] < OFF_M).mean()), 1),
        "inside_1m": round(100 * float((near_m[keep] < SLOW_M).mean()), 1),
    }
    return res


def _sensitivity(d, w: str, keep: np.ndarray, near_m: np.ndarray, minutes: float) -> dict:
    """The sector scheme if the labels' distances were really TRUE_SCALE_GUESS / LIVE_SCALE times
    larger (the live scale is not metric, docs/haptics-sim.md): false front cue, missed near,
    and over-urgent (the cue reads under FAST_M = 0.5 m but the obstacle is not)."""
    f = TRUE_SCALE_GUESS / LIVE_SCALE
    on = commands(
        d[f"{w}/strips"], d[f"{w}/path"], d[f"{w}/ttc"], float(d[f"{w}/hfov"][0]), "sector", 2, 26
    ).any(axis=1)  # fmt: skip
    nm = near_m * f
    strips = d[f"{w}/strips"]
    front = strips[:, 5:10].min(axis=1)  # the centre third of the camera, live-scale metres
    urgent = keep & (front < 0.5)
    return {
        "label_distance_factor": round(f, 2),
        "a_false_buzz": score(on[keep], nm[keep], OFF_M, minutes, "false"),
        "c_missed_inside_2m": score(on[keep], nm[keep], OFF_M, minutes, "miss"),
        "c_missed_inside_1m": score(on[keep], nm[keep], SLOW_M, minutes, "miss"),
        "b_over_urgent_pct": round(
            100 * float((urgent & (nm >= 0.5)).sum() / max(1, urgent.sum())), 1
        ),
        "near_pct_inside_2m": round(100 * float((nm[keep] < OFF_M).mean()), 1),
        "near_pct_inside_1m": round(100 * float((nm[keep] < SLOW_M).mean()), 1),
    }


def show(w: str, res: dict) -> None:
    print(w, res["minutes"], "min,", res["frames"], "tracked frames; near", res["near_pct"])
    for scheme in SCHEMES:
        r, lo, hi = res[scheme]["x1.0"], res[scheme]["x0.84"], res[scheme]["x1.16"]
        ev, fb = "events_per_min", r["false_buzz"]
        print(
            f"  {scheme:18s} cue on {res[scheme]['cue_on_pct']:5.1f}%  "
            f"false {fb[ev]}/min ({lo['false_buzz'][ev]}..{hi['false_buzz'][ev]})  "
            f"missed<2m {r['missed_inside_2m'][ev]}/min  "
            f"missed<1m {r['missed_inside_1m'][ev]}/min  "
            f"{res[scheme]['missed_inside_2m_frames']}"
        )


def main(names: list[str] | None = None) -> None:
    """Score the named walks' labels (all when none), merged into label-scores.json."""
    d = np.load(CACHE)
    path = ROOT / "bench" / "results" / "label-scores.json"
    out = json.loads(path.read_text()) if path.exists() and names else {}
    for lab_path in sorted((ROOT / "bench" / "labels").glob("*-sectors.json")):
        lab = json.loads(lab_path.read_text())
        if names and lab["walk"] not in names:
            continue
        out[lab["walk"]] = score_one(d, lab)
        show(lab["walk"], out[lab["walk"]])
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
