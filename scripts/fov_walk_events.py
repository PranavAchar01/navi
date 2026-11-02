"""Field-of-view events in recorded glasses walks: when a hazard the camera just saw leaves the
view while it is still in the way.

For each walk: the glasses IMU (motion.jsonl) gives head pitch from gravity; its clock is matched
to the frames by correlating gyro turn speed with the image's own frame-to-frame shift. When a
cue log from the live server is given (data/demo-takes/*-cues.jsonl), its safety records give the
live path distance per frame, and the events are the frames where the path read under 1.2 m and
then jumped at least JUMP_M further (or went empty) within 2 s; each is
labelled by what the head did in between (lift, turn, or neither). Writes
bench/results/fov-walk-events.json.

    PYTHONPATH=. .venv/bin/python scripts/fov_walk_events.py WALK [--cues CUES.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from metasight.fovmem import pitch_from_gravity

ROOT = Path(__file__).resolve().parent.parent
JUMP_M = 0.35  # the path reading moved this much further away: what was nearest left the view


def load_motion(walk: Path) -> np.ndarray:
    rows = []
    for line in open(walk / "motion.jsonl"):
        rows += [r for r in json.loads(line)["s"] if len(r) >= 7]
    return np.array(rows)


def image_shift(walk: Path, frames: list[dict]) -> np.ndarray:
    """Per frame, the pixel shift (dx, dy) from the frame before, at 1/8 size."""
    out = np.zeros((len(frames), 2))
    prev = None
    win = None
    for k, f in enumerate(frames):
        img = cv2.imread(str(walk / "frames" / f["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            prev = None
            continue
        g = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8)).astype(np.float32)
        if win is None:
            win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g, win)
            out[k] = dx, dy
        prev = g
    return out


def clock_offset(t_img: np.ndarray, speed_img: np.ndarray, mot: np.ndarray) -> float:
    """Offset to add to frame time (s) to get IMU time, from turn speed agreement."""
    w = np.linalg.norm(mot[:, 4:7], axis=1)
    base = mot[0, 0] - t_img[0]
    best, best_c = base, -2.0
    for off in base + np.arange(-3.0, 3.0, 0.01):
        g = np.interp(t_img + off, mot[:, 0], w)
        c = np.corrcoef(g, speed_img)[0, 1]
        if c > best_c:
            best, best_c = off, c
    return float(best), float(best_c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("--cues")
    a = ap.parse_args()
    walk = Path(a.walk)
    frames = [json.loads(line) for line in open(walk / "frames.jsonl")]
    frames.sort(key=lambda f: f["ts"])
    t_img = np.array([f["ts"] for f in frames]) / 1000.0
    shift = image_shift(walk, frames)
    dt = np.diff(t_img, prepend=t_img[0] - 0.07).clip(0.02, 0.5)
    speed_img = np.linalg.norm(shift, axis=1) / dt
    mot = load_motion(walk)
    off, corr = clock_offset(t_img, speed_img, mot)
    # Gravity, low passed at about 2 Hz, then pitch per frame.
    acc = mot[:, 1:4]
    k = 7
    acc_s = np.stack([np.convolve(acc[:, i], np.ones(k) / k, mode="same") for i in range(3)], 1)
    g_at = np.stack([np.interp(t_img + off, mot[:, 0], acc_s[:, i]) for i in range(3)], 1)
    pitch = np.array([pitch_from_gravity(g) for g in g_at])
    w_at = np.stack([np.interp(t_img + off, mot[:, 0], mot[:, 4 + i]) for i in range(3)], 1)
    rec = {
        "walk": walk.name,
        "clock_offset_s": round(off, 3),
        "turn_speed_corr": round(corr, 3),
        "pitch_deg_pct": [round(float(x), 1) for x in np.percentile(pitch, [5, 25, 50, 75, 95])],
        "frames": len(frames),
    }
    # Vertical image shift should follow pitch rate (image moves down when the head lifts).
    dp = np.gradient(pitch, t_img)
    rec["dy_vs_pitch_rate_corr"] = round(float(np.corrcoef(shift[:, 1] / dt, dp)[0, 1]), 3)
    seq_of = {f["ts"]: f["seq"] for f in frames}
    per = {}
    if a.cues:
        for line in open(a.cues):
            d = json.loads(line)
            if d.get("type") != "safety":
                continue
            ts = float(d["ts"]) / 1000.0
            i = int(np.clip(np.searchsorted(t_img, ts), 0, len(t_img) - 1))
            if abs(t_img[i] - ts) > 0.03 and i > 0 and abs(t_img[i - 1] - ts) < abs(t_img[i] - ts):
                i -= 1
            path = d.get("path")
            per[i] = {
                "t": ts,
                "path": math.inf if path is None else float(path),
                "floor": d.get("floor"),
                "yaw": d.get("yaw"),
            }
        idx = sorted(per)
        events = []
        last_end = -1.0
        for n, i in enumerate(idx):
            p0 = per[i]
            if p0["path"] > 1.2 or p0["t"] < last_end:
                continue
            for j in idx[n + 1 :]:
                p1 = per[j]
                if p1["t"] - p0["t"] > 2.0:
                    break
                if p1["path"] < p0["path"] + JUMP_M:
                    continue
                # Head lift (pitch toward level or up) or turn between the two frames.
                lift = pitch[i] - pitch[j]
                turn = 0.0
                if p0["yaw"] is not None and p1["yaw"] is not None:
                    turn = math.degrees(
                        abs((p1["yaw"] - p0["yaw"] + math.pi) % (2 * math.pi) - math.pi)
                    )
                yaw_gyro = float(
                    np.degrees(
                        np.abs(np.trapezoid(w_at[i : j + 1], t_img[i : j + 1], axis=0)).max()
                    )
                )
                kind = "lift" if lift >= 6 else "turn" if max(turn, yaw_gyro) >= 15 else "other"
                events.append(
                    {
                        "seen_seq": seq_of[frames[i]["ts"]],
                        "seen_file": frames[i]["file"],
                        "lost_seq": seq_of[frames[j]["ts"]],
                        "lost_file": frames[j]["file"],
                        "path_m": p0["path"],
                        "gap_s": round(p1["t"] - p0["t"], 2),
                        "pitch_seen": round(float(pitch[i]), 1),
                        "pitch_lost": round(float(pitch[j]), 1),
                        "turn_deg": round(max(turn, yaw_gyro), 1),
                        "kind": kind,
                    }
                )
                last_end = p1["t"] + 1.0
                break
        rec["events"] = events
        rec["event_kinds"] = {
            k: sum(e["kind"] == k for e in events) for k in ("lift", "turn", "other")
        }
        rec["safety_frames"] = len(per)
    out = ROOT / "bench" / "results" / "fov-walk-events.json"
    allr = json.loads(out.read_text()) if out.exists() else {}
    allr[walk.name] = rec
    out.write_text(json.dumps(allr, indent=1))
    np.savez(
        ROOT / "bench" / "results" / f"fov-pitch-{walk.name}.npz",
        t=t_img,
        pitch=pitch,
        seq=np.array([f["seq"] for f in frames]),
        shift=shift,
        offset=off,
    )
    print(json.dumps({k: v for k, v in rec.items() if k != "events"}, indent=1))
    for e in rec.get("events", [])[:40]:
        print(e)


if __name__ == "__main__":
    main()
