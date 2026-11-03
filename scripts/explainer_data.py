"""Data for the technical explainer video (data/demo-takes/explainer), all from recorded runs.

Walk 20261102-030055 is the one take with a full server cue log (20261102-030143-cues.jsonl):
its safety records carry what the server detected, the hazard outlines, the eight sector
distances and the cues, and its belt records what went to the belt. The belt3d replay
(data/demo-takes/cut/belt3d/replay.mp4) plays the same belt records, so every panel shares one
clock, `tau` = seconds since the first belt record.

What the log does not keep is recomputed with the server's own code on the same frames: depth
and the floor-aligned obstacle grid (`SafetyLoop.reflex`), and the field-of-view memory's cells
(`fovmem.FovLoop`, fed the logged detections and heading, as scripts/replay_fov.py does).

The room is the reconstruction the object twin was built on (recon-v2/v2.1: mesh, objects,
twin.json, relocalised cameras), in its y-up room frame.

Stage 1 takes a ~/helloworld/.heavy-lock-N slot (depth on ~600 frames):

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/explainer_data.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
WALK = ROOT / "data/walks/20261102-030055-glasses"
CUES = ROOT / "data/demo-takes/20261102-030143-cues.jsonl"
RECON = ROOT / "data/rooms/bedroom/recon-v2/v2.1"
OUT = ROOT / "data/demo-takes/explainer/hf/assets"
FPS = 30
TAU0, SPAN = 15.5, 24.0  # the stretch of the log the film plays, seconds of tau
WARM = 1000  # memory starts this frame, so it holds cells by the time the film starts
CUT_M = 2.2  # room points above floor + this are ceiling


def log_records() -> tuple[dict[int, dict], list[dict], float]:
    safety, belt = {}, []
    for line in open(CUES):
        d = json.loads(line)
        if d.get("type") == "safety":
            safety[d["seq"]] = d
        elif d.get("type") == "belt":
            belt.append(d)
    return safety, belt, belt[0]["wall"]


def frame_tau(frames: list[dict], safety: dict[int, dict], t0: float) -> np.ndarray:
    """tau of every frame: the server's receive clock where it logged the frame, else the
    capture clock shifted by the median receive delay."""
    ts = np.array([f["ts"] / 1000.0 for f in frames])
    seqs = [s for s in safety if s < len(frames)]
    lag = float(np.median([safety[s]["wall"] - ts[s] for s in seqs]))
    tau = ts + lag - t0
    for s in seqs:
        tau[s] = safety[s]["wall"] - t0
    return tau


def ramp(x: np.ndarray) -> np.ndarray:
    """0..1 to BGR: near is bright cyan-white, far fades to the background navy."""
    stops = np.array([[11, 8, 6], [70, 40, 18], [170, 120, 40], [235, 210, 120], [255, 250, 235]])
    p = np.clip(x, 0, 1) * (len(stops) - 1)
    i = np.minimum(p.astype(int), len(stops) - 2)
    f = (p - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def depth_frame(depth_m: np.ndarray, grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    d = cv2.resize(np.nan_to_num(depth_m.astype(np.float32), nan=9.0), (w, h), cv2.INTER_CUBIC)
    img = ramp(1.0 - (np.clip(d, 0.3, 4.0) - 0.3) / 3.7).astype(np.float32)
    g = cv2.resize(grid, (w, h), interpolation=cv2.INTER_NEAREST)
    obst = (g < 255).astype(np.float32)
    obst = cv2.GaussianBlur(obst, (0, 0), 1.2)[..., None]
    amber = np.array([40, 150, 255], np.float32)
    img = img * (1 - 0.55 * obst) + amber * 0.55 * obst
    return img.astype(np.uint8)


def run_pipeline(frames, safety, tau, want: set[int]) -> tuple[dict[int, dict], dict]:
    """Depth, obstacle grid and memory cells for the frames in `want`."""
    from replay_fov import clock, load_motion

    from metasight.config import Settings
    from metasight.fovmem import FovLoop
    from metasight.heading import Heading
    from metasight.safety import Obstacle, SafetyLoop

    cam = json.loads((WALK / "camera.json").read_text())
    s = Settings.load(ROOT / "metasight.glasses.toml")
    s.hfov_deg = cam.get("hfov", s.hfov_deg)
    loop = SafetyLoop(s)
    fov = FovLoop(s.hfov_deg, s.eye_height_m, low_slope=0.02)
    head = Heading(s.hfov_deg)
    mot = load_motion(WALK)
    off, _ = clock(WALK, frames, mot)
    accel: list[np.ndarray] = []
    m, t_prev = 0, mot[0, 0]
    out: dict[int, dict] = {}
    depth_dir = OUT.parent / "work" / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    timing = []
    for f in frames[WARM : max(want) + 1]:
        t = f["ts"] / 1000.0 + off
        while m < len(mot) and mot[m, 0] <= t:
            row = mot[m]
            accel.append(row[1:4])
            accel[:] = accel[-40:]
            up = np.mean(accel, axis=0)
            up = up / (np.linalg.norm(up) or 1.0)
            up = up if up[1] >= 0 else -up
            head.gyro(row[4:7], up, row[0] - t_prev, t)
            t_prev = row[0]
            m += 1
        bgr = cv2.imread(str(WALK / "frames" / f["file"]))
        res = loop.reflex(bgr)
        sc = res.scene
        lg = safety.get(f["seq"])
        obs = [
            Obstacle(o["label"], o["score"], tuple(o["box"]), o["dist"])
            for o in (lg or {}).get("obstacles", [])
        ]
        if lg is not None and lg.get("yaw") is not None:
            hd = SimpleNamespace(yaw=lg["yaw"], turn=head.turn, pos=np.array(lg["pos"]))
        else:
            hd = SimpleNamespace(yaw=None, turn=head.turn, pos=None)
        r = fov.step(f["ts"] / 1000.0, res.depth_m, sc.grid, sc.floor, obs, accel, hd)
        timing.append(res.timings_ms.get("depth", math.nan))
        if f["seq"] not in want:
            continue
        cv2.imwrite(
            str(depth_dir / f"{f['seq']}.jpg"),
            depth_frame(res.depth_m, sc.grid, (bgr.shape[1], bgr.shape[0])),
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
        mem = fov.mem
        yaw = hd.yaw if hd.yaw is not None else hd.turn
        px, pz = (0.0, 0.0) if hd.pos is None else (float(hd.pos[0]), float(hd.pos[2]))
        sel = mem.hits >= mem.min_hits
        cells = mem.pts[sel]
        fw = (math.sin(yaw), math.cos(yaw))
        rt = (-math.cos(yaw), math.sin(yaw))
        dx, dz = cells[:, 0] - px, cells[:, 1] - pz
        cr, cf = dx * rt[0] + dz * rt[1], dx * fw[0] + dz * fw[1]
        near = np.hypot(cr, cf) < 3.2
        cr, cf, hh = cr[near], cf[near], cells[near, 2]
        inview = np.abs(np.degrees(np.arctan2(cr, cf))) < s.hfov_deg / 2
        keep = np.linspace(0, len(cr) - 1, min(len(cr), 600)).astype(int) if len(cr) else []
        out[f["seq"]] = {
            "cells": [
                [
                    round(float(cr[i]), 2),
                    round(float(cf[i]), 2),
                    round(float(hh[i]), 2),
                    int(inview[i]),
                ]
                for i in keep
            ],
            "hidden": r.get("hidden"),
            "pitch": r.get("pitch"),
            "floor": bool(sc.floor),
        }
    return out, {"depth_ms_median": round(float(np.nanmedian(timing)), 1)}


def write_video(paths: list[Path], dst: Path, size: tuple[int, int]) -> None:
    w, h = size
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "14",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "15",
            str(dst),
        ],
        stdin=subprocess.PIPE,
    )
    cache: dict[Path, np.ndarray] = {}
    for p in paths:
        if p not in cache:
            cache.clear()
            cache[p] = cv2.resize(cv2.imread(str(p)), (w, h), interpolation=cv2.INTER_AREA)
        ff.stdin.write(cache[p].tobytes())
    ff.stdin.close()
    ff.wait()


def room() -> dict:
    """Room points (ceiling cut, every other vertex), when each one is first fused (the first
    relocalised camera, in walk time order, that has it within 3 m and 35 degrees of its view
    axis), the camera path, the objects and the twin's history."""
    doc = json.loads((RECON / "objects.json").read_text())
    fy = doc["floor_y"]
    raw = (RECON / "mesh.ply").resolve().read_bytes()
    head_end = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(raw[:head_end].split(b"element vertex ")[1].split(b"\n")[0])
    vt = np.dtype([("p", "<f8", 3), ("n", "<f8", 3), ("c", "u1", 3)])
    v = np.frombuffer(raw, vt, count=n, offset=head_end)
    p, c = v["p"].astype(np.float32), v["c"]
    lo, hi = np.percentile(p[::50], [1.5, 98.5], axis=0)
    ok = (p[:, 1] < fy + CUT_M) & (p[:, 1] > fy - 0.1)
    ok &= (p[:, 0] > lo[0] - 0.1) & (p[:, 0] < hi[0] + 0.1)
    ok &= (p[:, 2] > lo[2] - 0.1) & (p[:, 2] < hi[2] + 0.1)
    idx = np.flatnonzero(ok)[::2]
    p, c = p[idx], c[idx]

    cams = []
    for line in open(RECON / "cams.jsonl"):
        d = json.loads(line)
        cams.append((d["walk"], d["file"], np.array(d["pose"], np.float32)))
    cams.sort(key=lambda x: x[0])  # walks in time order; within a walk the file is in order
    sub = cams[:: max(1, len(cams) // 480)]
    pos = np.array([x[2][:3, 3] for x in sub])
    fwd = np.array([x[2][:3, 2] for x in sub])
    first = np.full(len(p), len(sub), np.int32)
    for i in range(len(sub)):
        d = p - pos[i]
        r = np.linalg.norm(d, axis=1)
        cosang = (d @ fwd[i]) / np.maximum(r, 1e-6)
        hit = (r < 3.0) & (cosang > math.cos(math.radians(35))) & (first == len(sub))
        first[hit] = i
    # Points no sampled camera faces still belong to the map: they arrive with the last ones.
    first[first == len(sub)] = len(sub) - 1
    order = np.argsort(first, kind="stable")
    p, c, first = p[order], c[order], first[order]
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    np.concatenate([p.astype(np.float32).ravel().view(np.uint8), c.ravel()]).tofile(
        OUT / "data/room.bin"
    )
    np.asarray(first, np.uint16).tofile(OUT / "data/room-first.bin")

    objs = []
    for o in doc["objects"]:
        top = np.array(o["obb"]["corners_top"])
        objs.append(
            {
                "label": o["label"],
                "class": o["class"],
                "hazard": bool(o.get("hazard")),
                "top": np.round(top, 3).tolist(),
                "bottom_y": round(o["obb"]["center"][1] - o["obb"]["size"][1] / 2, 3),
                "top_m": o.get("top_height_m"),
                "clear_m": o.get("floor_clearance_m"),
                "size": o.get("size_m"),
            }
        )
    twin = json.loads((RECON / "twin.json").read_text())
    tm = json.loads((RECON / "twin-metrics.json").read_text())
    walks = sorted({x[0] for x in cams})
    return {
        "n": int(len(p)),
        "floor_y": fy,
        "cams": len(sub),
        "cam_pos": np.round(pos, 3).tolist(),
        "cam_fwd": np.round(fwd, 3).tolist(),
        "frames_fused": json.loads((RECON / "metrics.json").read_text())["summary"]["frames_fused"],
        "walks": len(walks),
        "objects": objs,
        "events": [e for e in twin["events"] if e["type"] in ("moved", "added", "removed")],
        "twin_counts": tm["events"],
        "tracks": tm["tracks"],
        "out_of_view_share": tm["hazards"]["out_of_view_share"],
    }


def sim() -> dict:
    """The find-the-chair walks of the parallel-walks montage (data/demo-takes/sim/grid), eight
    motors, seeds 1000-1029, per arm: path, contact count and seated flag at 15 Hz."""
    grid = ROOT / "data/demo-takes/sim/grid/scenes"
    first = json.loads((grid / "I0-eight-1000.json").read_text())
    arms = {}
    for arm in ("I0", "I3", "I5", "I6"):
        walks = []
        for seed in range(1000, 1030):
            d = json.loads((grid / f"{arm}-eight-{seed}.json").read_text())
            fr = d["frames"][::2]
            walks.append(
                {
                    "seed": seed,
                    "xy": [[f["x"], f["y"]] for f in fr],
                    "c": [f["collisions"] for f in fr],
                    "sit": [round(f["sit"], 2) for f in fr],
                    "clean": bool(d["metrics"]["success"]) and d["metrics"]["collisions"] == 0,
                    "contacts": d["metrics"]["collisions"],
                }
            )
        arms[arm] = {"note": d["arm_note"], "walks": walks}
    return {
        "hz": first["fps"] / 2,
        "walls": first["walls"],
        "objects": first["objects"],
        "start": first["start"],
        "goal": first["goal"],
        "arms": arms,
    }


def belt_dark() -> None:
    """The belt3d replay (light studio background) cut to the film's stretch of the log and laid
    on the film's dark background: alpha from each pixel's distance to the studio colour, so
    the soft floor shadow fades out and the belt keeps its own colours."""
    src = ROOT / "data/demo-takes/cut/belt3d/replay.mp4"
    off = json.loads((src.parent / "replay.json").read_text())["offset"]
    w, h = 1920, 672
    rd = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{TAU0 + off:.3f}",
            "-i",
            str(src),
            "-t",
            str(SPAN),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    x0, x1, y0, y1 = 100, 1820, 40, 600
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{x1 - x0}x{y1 - y0}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            str(OUT / "media/belt.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    studio = np.array([246, 245, 248], np.float32)
    dark = np.array([12, 9, 7], np.float32)
    while True:
        buf = rd.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        f = np.frombuffer(buf, np.uint8).reshape(h, w, 3)[y0:y1, x0:x1].astype(np.float32)
        luma = f.mean(-1)
        sat = f.max(-1) - f.min(-1)
        # The floor shadow is a light neutral grey: keep what is darker than it or coloured.
        dark_a = np.clip((studio.mean() - luma - 70) / 50, 0, 1)
        a = np.maximum(dark_a, np.clip((sat - 18) / 30, 0, 1))
        a = cv2.GaussianBlur(cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((5, 5))), (0, 0), 0.8)
        a = a[..., None]
        ff.stdin.write((f * a + dark * (1 - a)).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()


def to_js() -> None:
    """The composition loads its data as plain scripts (a render fetches nothing): each JSON
    becomes `window.EX_<name> = ...`, each binary a base64 string."""
    import base64

    d = OUT / "data"
    for name in ("live", "room", "sim"):
        if (d / f"{name}.json").exists():
            body = (d / f"{name}.json").read_text()
            (d / f"{name}.js").write_text(f"window.EX_{name}={body};\n")
    for name in ("room", "room-first"):
        b64 = base64.b64encode((d / f"{name}.bin").read_bytes()).decode()
        (d / f"{name}-bin.js").write_text(f"window.EX_{name.replace('-', '_')}_b64='{b64}';\n")


def main() -> None:
    if "--js" in sys.argv:
        to_js()
        return
    if "--belt" in sys.argv:
        belt_dark()
        return
    if "--sim" in sys.argv:
        (OUT / "data/sim.json").write_text(json.dumps(sim(), separators=(",", ":")))
        return
    frames = sorted((json.loads(x) for x in open(WALK / "frames.jsonl")), key=lambda f: f["seq"])
    safety, belt, t0 = log_records()
    tau = frame_tau(frames, safety, t0)
    n_out = int(round(SPAN * FPS))
    picks = []
    for k in range(n_out):
        t = TAU0 + k / FPS
        i = int(np.searchsorted(tau, t, side="right") - 1)
        picks.append(max(i, 0))
    want = set(picks)

    live, stats = run_pipeline(frames, safety, tau, want)
    media = OUT / "media"
    media.mkdir(parents=True, exist_ok=True)
    write_video([WALK / "frames" / frames[i]["file"] for i in picks], media / "cam.mp4", (504, 896))
    write_video(
        [OUT.parent / "work/depth" / f"{frames[i]['seq']}.jpg" for i in picks],
        media / "depth.mp4",
        (504, 896),
    )

    bt = np.array([b["wall"] - t0 for b in belt])
    rows = []
    for k, i in enumerate(picks):
        t = TAU0 + k / FPS
        seq = frames[i]["seq"]
        lg = safety.get(seq) or safety[max(s for s in safety if s <= seq)]
        b = belt[max(int(np.searchsorted(bt, t, side="right") - 1), 0)]
        mem = live[seq]
        rows.append(
            {
                "seq": seq,
                "tau": round(t, 3),
                "obstacles": [
                    {
                        "label": o["label"],
                        "score": o["score"],
                        "box": [round(x, 4) for x in o["box"]],
                        "dist": o["dist"],
                    }
                    for o in lg["obstacles"]
                ],
                "hazards": [{"poly": h["poly"], "dist": h["dist"]} for h in lg["hazards"]],
                "sectors": lg["sectors"],
                "known": lg["known"],
                "path": lg["path"],
                "cues": [[c["mode"], round(c["intensity"], 3)] for c in lg["cues"]],
                "duty": b["duty"],
                "modes": b["modes"],
                "ms": lg["ms"],
                "fps": lg["fps"],
                "cells": mem["cells"],
                "hidden": mem["hidden"],
                "pitch": mem["pitch"],
            }
        )
    meta = {
        "walk": WALK.name,
        "cues": CUES.name,
        "tau0": TAU0,
        "span": SPAN,
        "fps": FPS,
        "belt_t0": t0,
        "seq_first": picks[0],
        "seq_last": picks[-1],
        **stats,
        "in_fps_median": float(np.median([s["in_fps"] for s in safety.values()])),
        "fps_median": float(np.median([s["fps"] for s in safety.values()])),
        "ms_median": {
            k: float(np.median([s["ms"][k] for s in safety.values()]))
            for k in ("depth", "detect", "total")
        },
    }
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data/live.json").write_text(json.dumps({"meta": meta, "rows": rows}))
    (OUT / "data/room.json").write_text(json.dumps(room()))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
