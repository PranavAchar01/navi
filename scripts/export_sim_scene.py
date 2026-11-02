"""Write one simulated walk as a scene file for the Blender renderer (scripts/blender_sim_room.py).

The file holds the room (wall boxes, the doors and whiteboard on them), every object with its
boxes and oriented outline, and the walker per video frame at 30 fps: position, heading, head
pitch, how far it has sat down, and the eight felt motor levels. Everything is in the simwalk
floor frame (x, y on the floor, z up, metres; heading counter-clockwise from +x).

Next to it go the head pose sidecar (<out>.pose.jsonl, the input of scripts/render_sim_sound.py)
and the belt cues (<out>.belt.json, the format web/simbelt.html replays), both on the same
video clock, so the sound, the belt panel and the render line up frame for frame.

    PYTHONPATH=. python scripts/export_sim_scene.py find_chair --seed 1000 --arm I5 \
        --out data/demo-takes/sim/scene.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sim_walk as E  # noqa: E402

from metasight import fovrooms  # noqa: E402
from metasight import hapsim as h  # noqa: E402
from metasight import simwalk as S  # noqa: E402

FPS = 30
SIT_S = 1.6
HOLD_S = 1.2
SIT_DROP_M = 0.5  # how far the eyes come down when sitting
FELT_ON = 0.05  # a motor below this level is off on the panel


def flush_parts(sc: S.Scenario) -> list[dict]:
    """Doors and the whiteboard: flat things on the walls the walk never feels, kept for the set."""
    if sc.world.size[0] < 4.5:  # catalogue rooms have none
        return []
    data = fovrooms.load_objects()
    floor_y = float(data["floor_y"])
    ox, oy = S.OFFSET
    out = []
    for o in data["objects"]:
        if o["class"] not in fovrooms.FLUSH:
            continue
        (px, py, pz), (sx, sy, sz) = o["obb"]["center"], o["obb"]["size"]
        out.append(
            {
                "name": o["id"],
                "cls": o["class"],
                "centre": [px + ox, -pz + oy, py - floor_y],
                "size": [sx, sz, sy],
                "open": o["id"] == "door_03",
            }
        )
    return out


def frames(sc: S.Scenario, tr: S.Trace) -> list[dict]:
    """The walk resampled to video frames, then the sit-down and a short hold."""
    st = tr.steps
    ts = np.array([s[0] for s in st])
    out = []
    n = int(ts[-1] * FPS) + 1
    for k in range(n):
        t = k / FPS
        i = min(int(np.searchsorted(ts, t)), len(st) - 1)
        s = st[i]
        out.append(
            {
                "t": round(t, 4),
                "x": round(s[1], 4),
                "y": round(s[2], 4),
                "heading": round(s[3], 4),
                "pitch_deg": round(sc.pitch_at(t), 2),
                "sit": 0.0,
                "felt": [round(float(v), 3) for v in s[4]],
                "collisions": int(s[6]),
                "min_clear": round(max(0.0, float(s[7])), 3),  # running minimum, metres
                "false": int(s[8]),  # false onsets so far
                "walking": True,
            }
        )
    last = dict(out[-1], walking=False)
    res = tr.result
    target = sc.world.obj(sc.target) if sc.target else None
    if sc.sit and res is not None and res.success and target is not None:
        cx, cy = target.obb[0], target.obb[1]
        x0, y0, hd0 = last["x"], last["y"], last["heading"]
        face = math.atan2(y0 - cy, x0 - cx)
        m = int(SIT_S * FPS)
        for k in range(m):
            a = (k + 1) / m
            b = min(1.0, a * 1.6)
            turn = (face - hd0 + math.pi) % (2 * math.pi) - math.pi
            out.append(
                dict(
                    last,
                    t=round(last["t"] + a * SIT_S, 4),
                    x=round(x0 + (cx - x0) * b, 4),
                    y=round(y0 + (cy - y0) * b, 4),
                    heading=round(hd0 + turn * b, 4),
                    sit=round(a, 3),
                    felt=[round(v * math.exp(-(k + 1) / 4), 3) for v in last["felt"]],
                )
            )
        last = out[-1]
    for k in range(int(HOLD_S * FPS)):
        out.append(dict(last, t=round(last["t"] + (k + 1) / FPS, 4), felt=[0.0] * 8))
    return out


def pose_rows(doc: dict) -> list[dict]:
    """The sidecar render_sim_sound.py reads: header, then one head pose per video frame."""
    head = {
        "scenario": doc["scenario"],
        "layout": doc["layout"],
        "fps": FPS,
        "first_frame_t": 0.0,
        "frame": doc["frame"],
        "target": doc["target"],
        "hazard": doc["hazard"],
        "objects": [
            {
                "name": o["name"],
                "label": o["label"],
                "centre": [round(v, 3) for v in o["obb"][:3]],
                "half": [round(v, 3) for v in o["obb"][3:6]],
                "yaw_deg": round(math.degrees(o["obb"][6]), 1),
            }
            for o in doc["objects"]
        ],
    }
    rows = [head]
    for k, f in enumerate(doc["frames"]):
        hd, p = f["heading"], math.radians(f["pitch_deg"])
        rows.append(
            {
                "frame": k,
                "t": round(k / FPS, 3),
                "x": f["x"],
                "y": f["y"],
                "z": round(S.EYE_H - SIT_DROP_M * f["sit"], 4),
                "heading_deg": round(math.degrees(hd), 2),
                "pitch_down_deg": f["pitch_deg"],
                "forward": [
                    round(math.cos(hd) * math.cos(p), 4),
                    round(math.sin(hd) * math.cos(p), 4),
                    round(-math.sin(p), 4),
                ],
            }
        )
    return rows


def mode_of(level: float) -> str:
    """The app's bands for a felt level: caution, then fast, then the solid stop."""
    if level < FELT_ON:
        return "off"
    return "slow" if level < 0.5 else "fast" if level < 0.85 else "solid"


def belt_cues(doc: dict) -> dict:
    """The felt levels as belt messages: duty 0-255 and the app's mode per motor."""
    cues = []
    for k, f in enumerate(doc["frames"]):
        felt = f["felt"]
        if len(felt) == 4:  # diag4: the four modules sit on the diagonal sectors
            felt = [0.0, felt[0], 0.0, felt[1], 0.0, felt[2], 0.0, felt[3]]
        duty = [int(round(255 * min(1.0, v))) for v in felt]
        modes = [mode_of(v) for v in felt]
        cues.append({"t": round(k / FPS, 4), "duty": duty, "modes": modes, "lost": False})
    return {"t0": 0.0, "layout": doc["layout"], "cues": cues}


def scene_file(sc: S.Scenario, tr: S.Trace, layout: str, seed: int) -> dict:
    m = tr.metrics()
    m["success"] = bool(m["success"] and m["collisions"] == 0)  # a bumped walk is not "reached"
    return {
        "scenario": sc.name,
        "title": sc.title,
        "layout": layout,
        "seed": seed,
        "fps": FPS,
        "frame": "simwalk floor frame: x, y on the floor, z up, metres; heading ccw from +x",
        "motor_deg": list(h.MOTOR_DEG[layout]),
        "walls": [list(b) for b in sc.world.walls],
        "flush": flush_parts(sc),
        "objects": [
            {
                "name": o.name,
                "label": o.label,
                "obb": list(o.obb),
                "boxes": [list(b) for b in o.boxes],
            }
            for o in sc.world.objs
        ],
        "start": list(sc.start),
        "goal": list(sc.goal),
        "target": sc.target,
        "hazard": sc.hazard,
        "metrics": m,
        "frames": frames(sc, tr),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=S.SCENARIOS)
    ap.add_argument("--layout", default="eight", choices=("eight", "diag4"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--arm", default="I5", choices=E.ARMS, help="walker iteration (run_sim_walk)")
    ap.add_argument("--max-s", type=float, help="keep only the first seconds of a long walk")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sc = S.scenario(a.scenario)
    tr = S.walk(sc, E.belt_for(a.arm, a.layout), seed=a.seed, **E.ARMS[a.arm][2])
    doc = scene_file(sc, tr, a.layout, a.seed)
    if a.max_s and len(doc["frames"]) > a.max_s * FPS:  # a walk that ran to the time limit
        doc["frames"] = doc["frames"][: int(a.max_s * FPS)]
        doc["cut_s"] = a.max_s
    doc["arm"] = a.arm
    doc["arm_note"] = E.ARMS[a.arm][3]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc))
    a.out.with_suffix(".pose.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in pose_rows(doc))
    )
    a.out.with_suffix(".belt.json").write_text(json.dumps(belt_cues(doc)))
    print(a.scenario, a.layout, a.arm, doc["metrics"], len(doc["frames"]), "frames")


if __name__ == "__main__":
    main()
