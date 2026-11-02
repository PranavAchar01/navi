"""Replay the hazard memory (metasight.fovmem.FovLoop) on a recorded glasses walk.

Stage 1 (heavy, takes a ~/helloworld/.heavy-lock-N slot): every frame goes through the same
depth + safety analysis the server runs (`SafetyLoop.process`), then `FovLoop.step` with the
recorded accelerometer (glasses IMU, clock matched to the frames the way scripts/fov_walk_events.py
does) and a heading. The heading is the recorded map yaw and pose when a cue log covers the frame,
else `Heading.gyro` turn integrated from the recorded gyro (as the server does without a map
pose). One row per frame goes to bench/results/fov-walk-replay-<walk>.jsonl.

Stage 2 (--score) reads those rows and writes bench/results/fov-walk-replay.json: per event,
whether the memory holds the hazard through the gap; memory-only cues a minute (memory nearer
than live by more than 0.5 m) overall and in still stretches (head gyro under 0.3 rad/s over a
second). No ground truth: these are upper bounds on false cues, not a false-alarm rate.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/replay_fov.py WALK [--cues CUES]
    PYTHONPATH=. .venv/bin/python scripts/replay_fov.py --score WALK [WALK ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
LOCKS = ROOT.parents[3] if ROOT.parent.name == "worktrees" else ROOT.parent
RES = ROOT / "bench" / "results"
STILL_W = 0.3  # rad/s, mean gyro speed over a second (the seated head scans: median 0.48)
MEM_GAP_M = 0.5  # memory nearer than live by this much is a memory-only cue
HOLD_TOL_M = 0.3
# Looked at the seen / middle / lost frames of each event of walk 20261102-030055 (2026-11-02).
BY_EYE = {
    859: "real turn: laptop and monitor 0.5 m ahead leave the view, room and chair appear",
    951: "real turn: chair back at about 0.7 m leaves, a door and a desk edge come in",
    1036: "real turn: table of robot arms leaves, bare wall",
    1164: "real turn: door and hanging bag at arm's length leave, bed and chair come in",
    1215: "not a loss: small pan, the desk edge stays in view at the right (depth change)",
    1343: "real turn away from the desk with the edge still beside him (earlier check)",
    1418: "not real: desk still in view, depth noise (earlier check)",
    1452: "real turn: window sill and wall leave, jersey passes, room comes in",
    1508: "partial turn: table stays in view at the right, chair edge stays at the bottom",
    1563: "not real: desk still in view, depth noise (earlier check)",
}


def take_slot() -> Path:
    n = 2
    while True:
        slot = LOCKS / f".heavy-lock-{n}"
        try:
            slot.mkdir()
        except FileExistsError:
            n += 1
            continue
        (slot / "owner").write_text(f"{os.getpid()} fov-replay\n")
        return slot


def release(slot: Path) -> None:
    (slot / "owner").unlink(missing_ok=True)
    slot.rmdir()


def load_motion(walk: Path) -> np.ndarray:
    rows = []
    for line in open(walk / "motion.jsonl"):
        rows += [r for r in json.loads(line)["s"] if len(r) >= 7]
    return np.array(rows)


def clock(walk: Path, frames: list[dict], mot: np.ndarray) -> tuple[float, float]:
    import fov_walk_events as fw

    t_img = np.array([f["ts"] for f in frames]) / 1000.0
    shift = fw.image_shift(walk, frames)
    dt = np.diff(t_img, prepend=t_img[0] - 0.07).clip(0.02, 0.5)
    return fw.clock_offset(t_img, np.linalg.norm(shift, axis=1) / dt, mot)


def replay(
    walk: Path, cues: Path | None, band_lo: float | None = None, prior: Path | None = None
) -> None:
    import cv2

    from metasight.config import Settings
    from metasight.fovmem import FovLoop
    from metasight.heading import Heading
    from metasight.safety import SafetyLoop

    cam = json.loads((walk / "camera.json").read_text())
    s = Settings.load(ROOT / "metasight.glasses.toml")
    s.hfov_deg = cam.get("hfov", s.hfov_deg)
    loop = SafetyLoop(s)
    band = None if band_lo is None else (band_lo, s.eye_height_m + 0.15)
    fov = FovLoop(s.hfov_deg, s.eye_height_m, low_slope=0.02, band=band)
    head = Heading(s.hfov_deg)
    frames = sorted((json.loads(x) for x in open(walk / "frames.jsonl")), key=lambda f: f["ts"])
    mot = load_motion(walk)
    off, corr = clock(walk, frames, mot)
    log: dict[int, dict] = {}
    if cues:
        for line in open(cues):
            d = json.loads(line)
            if d.get("type") == "safety":
                log[d["seq"]] = d
    tag = "" if band_lo is None else f"-lo{band_lo:g}"  # a variant run, kept apart from the base
    ptree, kept = None, []
    if prior is not None:  # L29: how well the live cells line up with a prior map's surfaces
        from scipy.spatial import cKDTree

        from metasight import recon
        from metasight.fovmem import prior_cells

        doc = recon.load_prior(prior)
        ptree = cKDTree(prior_cells(doc["objects"], None, (0.08, 1.75), doc["floor_y"])[0])
        tag += "-prior"
    out = RES / f"fov-walk-replay-{walk.name}{tag}.jsonl"
    accel: list[np.ndarray] = []
    m = 0
    t_prev = mot[0, 0]
    with open(out, "w") as fh:
        for f in frames:
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
            bgr = cv2.imread(str(walk / "frames" / f["file"]))
            if bgr is None:
                continue
            res = loop.process(bgr)
            sc = res.scene
            lg = log.get(f["seq"])
            if lg is not None and lg.get("yaw") is not None:
                hd = SimpleNamespace(
                    yaw=lg["yaw"],
                    turn=head.turn,
                    pos=np.array(lg["pos"]) if lg.get("pos") else None,
                )
            else:
                hd = SimpleNamespace(yaw=None, turn=head.turn, pos=None)
            r = fov.step(f["ts"] / 1000.0, res.depth_m, sc.grid, sc.floor, res.obstacles, accel, hd)
            row = {
                "seq": f["seq"],
                "t": round(f["ts"] / 1000.0, 3),
                "live": [None if not math.isfinite(v) else round(float(v), 2) for v in sc.sectors],
                "path_live": None if not math.isfinite(sc.path_m) else round(float(sc.path_m), 2),
                "scene_floor": bool(sc.floor),
                "gyro": round(float(np.linalg.norm(mot[max(m - 1, 0), 4:7])), 3),
                "yaw_src": "log" if lg is not None and lg.get("yaw") is not None else "gyro",
            }
            row.update(r)
            if fov.last_cues is not None and r.get("pitch") is not None:
                xz = fov.last_cues.hidden_xz
                row["xz"] = [
                    None if not math.isfinite(v[0]) else [round(v[0], 3), round(v[1], 3)]
                    for v in xz
                ]
                row["hpos"] = (
                    None
                    if hd.pos is None
                    else [round(float(hd.pos[0]), 3), round(float(hd.pos[2]), 3)]
                )
                row["hh"] = [
                    None if not math.isfinite(v) else round(v, 3) for v in fov.last_cues.hidden_h
                ]
                row["hyaw"] = round(float(hd.yaw if hd.yaw is not None else hd.turn), 4)
            if ptree is not None and row.get("hpos") and row["yaw_src"] == "log":
                mem = fov.mem
                sel = ~mem.prior & (mem.hits >= mem.min_hits)
                cells = mem.pts[sel]
                near = np.hypot(cells[:, 0] - row["hpos"][0], cells[:, 1] - row["hpos"][1]) < 2.0
                cells = cells[near]
                if len(cells):
                    d, _ = ptree.query(cells)
                    row["al_n"], row["al_med"] = len(cells), round(float(np.median(d)), 3)
                    row["al_in15"] = round(float(np.mean(d <= 0.15)), 3)
                    if f["seq"] % 10 == 0:
                        kept.append(np.column_stack([cells, np.full(len(cells), f["seq"])]))
            fh.write(json.dumps(row) + "\n")
    if kept:
        np.save(RES / f"fov-prior-cells-{walk.name}.npy", np.vstack(kept))
    print(f"{walk.name}: {len(frames)} frames, clock corr {corr:.3f} -> {out}")


def runs(flag: np.ndarray) -> int:
    return int(np.sum(flag[1:] & ~flag[:-1]) + (1 if flag[0] else 0)) if len(flag) else 0


def score_walk(name: str, events: list[dict], yaw_of: dict[int, float]) -> dict:
    rows = [json.loads(x) for x in open(RES / f"fov-walk-replay-{name}.jsonl")]
    rows = [r for r in rows if r.get("pitch") is not None]
    t = np.array([r["t"] for r in rows])
    inf = math.inf

    def hid(r):
        return [inf if v is None else v for v in (r.get("hidden") or [None] * 8)]

    def liv(r):
        v = [inf if x is None else x for x in r["live"]]  # front-left, front, front-right
        out = [inf] * 8
        out[7], out[0], out[1] = v
        return out

    onsets = np.zeros((len(rows), 8), bool)  # a sector's held cell appears inside 1.1 m
    mem_only = np.zeros(len(rows), bool)
    mem_only_near = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        h, lv = hid(r), liv(r)
        mem_only[i] = any(h[k] < inf and lv[k] - h[k] > MEM_GAP_M for k in range(8))
        onsets[i] = [h[k] < 1.1 for k in range(8)]
        mem_only_near[i] = any(h[k] < 1.1 and lv[k] - h[k] > MEM_GAP_M for k in range(8))
    g = np.array([r["gyro"] for r in rows])
    step = float(np.median(np.diff(t)))
    w = max(1, int(round(1.0 / max(step, 1e-3))))
    gm = np.convolve(g, np.ones(w) / w, mode="same")
    still = gm < STILL_W
    out = {
        "frames": len(rows),
        "minutes": round(float(t[-1] - t[0]) / 60, 2),
        "with_floor_fit": round(float(np.mean([bool(r["floor"]) for r in rows])), 3),
        "yaw_from_log": round(float(np.mean([r["yaw_src"] == "log" for r in rows])), 3),
        "still_share": round(float(still.mean()), 3),
        "mem_only_frames_share": round(float(mem_only.mean()), 3),
    }
    for tag, flag in (("any", mem_only), ("under_1.1m", mem_only_near)):
        for part, sel in (("all", np.ones(len(rows), bool)), ("still", still)):
            mins = max(float(np.sum(sel)) * step / 60, 1e-6)
            out[f"mem_only_cues_per_min_{tag}_{part}"] = round(runs(flag & sel) / mins, 1)
    # L27 cue gate (fovmem.gate_memory_cues) on the same rows: memory-only cues (any distance)
    # per minute, overall, in gyro-still stretches and in step-still stretches (no step for 1 s).
    if rows and "close" in rows[0]:
        from metasight.fovmem import gate_memory_cues

        nostep = np.array([(r.get("still_s") or 0.0) > 1.0 for r in rows])
        out["step_still_share"] = round(float(nostep.mean()), 3)
        mins = max(float(nostep.sum()) * step / 60, 1e-6)
        out["mem_only_cues_per_min_any_step_still"] = round(runs(mem_only & nostep) / mins, 1)
        for g in ("s", "a", "r", "sar"):
            flag = np.zeros(len(rows), bool)
            for i, r in enumerate(rows):
                gh, _, _ = gate_memory_cues(
                    r["hidden"],
                    r["band"],
                    r["close"],
                    r.get("still_s"),
                    mute_still="s" in g,
                    approach="a" in g,
                    mute_rear="r" in g,
                )
                lv = liv(r)
                flag[i] = any(gh[k] is not None and lv[k] - gh[k] > MEM_GAP_M for k in range(8))
            for part, sel in (
                ("all", np.ones(len(rows), bool)),
                ("gyro_still", still),
                ("step_still", nostep),
            ):
                mins = max(float(np.sum(sel)) * step / 60, 1e-6)
                out[f"gate_{g}_mem_only_cues_per_min_any_{part}"] = round(
                    runs(flag & sel) / mins, 1
                )
    for part, sel in (("all", np.ones(len(rows), bool)), ("still", still)):
        mins = max(float(np.sum(sel)) * step / 60, 1e-6)
        n = sum(runs(onsets[:, k] & sel) for k in range(8))
        out[f"sector_onsets_per_min_under_1.1m_{part}"] = round(n / mins, 1)
    # Directional hold test: a hazard seen in live sector b0 (degrees, clockwise from ahead) moves
    # to b0 + (yaw change) after the head turns (fovmem: relative bearing = yaw - world bearing);
    # it is held if the hidden ring has a cell within HOLD_TOL_M of the live reading in that
    # sector or either neighbour. Chance level: the same test on every other frame pair 30 frames
    # apart (about 2 s) that has yaw from the cue log and a live reading under 1.2 m.
    centre = {7: -45.0, 0: 0.0, 1: 45.0}

    def directional(i0: int, i1: int) -> tuple[bool | None, float]:
        y0, y1 = yaw_of.get(rows[i0]["seq"]), yaw_of.get(rows[i1]["seq"])
        lv = liv(rows[i0])
        k0 = min(centre, key=lambda k: lv[k])
        if y0 is None or y1 is None or lv[k0] == inf:
            return None, inf
        dyaw = math.degrees((y1 - y0 + math.pi) % (2 * math.pi) - math.pi)
        k1 = int(round(((centre[k0] + dyaw) % 360) / 45)) % 8
        h = hid(rows[i1])
        ok = any(abs(h[(k1 + d) % 8] - lv[k0]) <= HOLD_TOL_M for d in (-1, 0, 1))
        return ok, lv[k0]

    hits = []
    for i in range(len(rows) - 30):
        ok, pre0 = directional(i, i + 30)
        if ok is not None and pre0 < 1.2:
            hits.append(ok)
    out["directional_chance_hold_rate_30_frames"] = round(float(np.mean(hits)), 2) if hits else None
    out["directional_chance_pairs"] = len(hits)
    by_seq = {r["seq"]: i for i, r in enumerate(rows)}
    evs = []
    for e in events:
        i0 = by_seq.get(e["seen_seq"])
        i1 = by_seq.get(e["lost_seq"])
        if i0 is None or i1 is None:
            continue
        pre = min(liv(rows[i0]))
        gap = range(i0, i1 + 1)
        mins = [min(hid(rows[i])) for i in gap]
        held = [any(abs(v - pre) <= HOLD_TOL_M for v in hid(rows[i])) for i in gap]
        evs.append(
            {
                "seen": e["seen_seq"],
                "lost": e["lost_seq"],
                "detector_label": e["kind"],
                "by_eye": BY_EYE.get(e["seen_seq"]),
                "pre_gap_live_m": None if pre == inf else round(pre, 2),
                "hidden_at_lost_m": None if mins[-1] == inf else round(mins[-1], 2),
                "held_at_lost": bool(held[-1]),
                "directional_held_at_lost": directional(i0, i1)[0],
                "directional_held_any_next_10": any(
                    directional(i0, j)[0] for j in range(i1, min(i1 + 10, len(rows)))
                ),
                "held_frame_share": round(float(np.mean(held)), 2),
                "held_any_next_10_frames": any(
                    abs(v - pre) <= HOLD_TOL_M
                    for i in range(i1, min(i1 + 10, len(rows)))
                    for v in hid(rows[i])
                ),
                "floor_fit_share": round(float(np.mean([bool(rows[i]["floor"]) for i in gap])), 2),
            }
        )
    out["events"] = evs
    return out


def align(name: str, prior: Path) -> None:
    """Offset between the live map and a prior map (L29), from the cells `replay(prior=...)` kept
    on frames with the map pose: the translation (and the turn about the prior's origin) that
    puts the live cells nearest to the prior's surfaces, over the whole take and its thirds."""
    from scipy.spatial import cKDTree

    from metasight import recon
    from metasight.fovmem import prior_cells

    doc = recon.load_prior(prior)
    tree = cKDTree(prior_cells(doc["objects"], None, (0.08, 1.75), doc["floor_y"])[0][:, :2])
    cells = np.load(RES / f"fov-prior-cells-{name}.npy")
    rng = np.random.default_rng(0)

    def fit(c: np.ndarray) -> dict:
        c = c[rng.choice(len(c), min(len(c), 3000), replace=False)]
        base = float(np.median(tree.query(c[:, :2])[0]))
        best = (base, 0.0, 0.0, 0.0)

        def search(turns, dxs, dzs):
            nonlocal best
            for turn in turns:
                yaw = math.radians(turn)
                cs, sn = math.cos(yaw), math.sin(yaw)
                rot = np.column_stack([cs * c[:, 0] - sn * c[:, 1], sn * c[:, 0] + cs * c[:, 1]])
                for dx in dxs:
                    for dz in dzs:
                        d = float(np.median(tree.query(rot + [dx, dz])[0]))
                        if d < best[0]:
                            best = (d, dx, dz, turn)

        coarse = np.arange(-1.6, 1.61, 0.1)
        search(np.arange(-10, 10.1, 5.0), coarse, coarse)
        _, bx, bz, bt = best
        search(
            np.arange(bt - 2.5, bt + 2.6, 1.25),
            np.arange(bx - 0.1, bx + 0.11, 0.025),
            np.arange(bz - 0.1, bz + 0.11, 0.025),
        )
        return {
            "cells": len(c),
            "median_nn_m_as_is": round(base, 3),
            "median_nn_m_best": round(best[0], 3),
            "shift_m": [round(best[1], 2), round(best[2], 2)],
            "shift_len_m": round(math.hypot(best[1], best[2]), 2),
            "turn_deg": best[3],
        }

    seqs = np.unique(cells[:, 3])
    cut = np.quantile(seqs, [1 / 3, 2 / 3])
    out = {"all": fit(cells)}
    for i, (lo, hi) in enumerate(zip([-math.inf, *cut], [*cut, math.inf], strict=True)):
        out[f"third{i + 1}"] = fit(cells[(cells[:, 3] > lo) & (cells[:, 3] <= hi)])
    (RES / f"fov-prior-align-{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk", nargs="+")
    ap.add_argument("--cues")
    ap.add_argument("--band-lo", type=float, help="lowest height (m) a cell is held at (L27)")
    ap.add_argument("--prior", help="objects.json of a prior map: measure how live cells line up")
    ap.add_argument("--align", action="store_true", help="fit the offset from the --prior cells")
    ap.add_argument("--score", action="store_true")
    ap.add_argument(
        "--yaw-cues", help="cue log with the map yaw; applies to the first walk only (--score)"
    )
    a = ap.parse_args()
    if a.align:
        align(Path(a.walk[0]).name, Path(a.prior))
        return
    if a.score:
        ev = json.loads((RES / "fov-walk-events.json").read_text())
        yaw_of = {}
        if a.yaw_cues:
            for line in open(a.yaw_cues):
                d = json.loads(line)
                if d.get("type") == "safety" and d.get("yaw") is not None:
                    yaw_of[d["seq"]] = d["yaw"]
        res = {}
        for w in a.walk:
            name = Path(w).name
            res[name] = score_walk(
                name, ev.get(name, {}).get("events", []), yaw_of if w == a.walk[0] else {}
            )
        (RES / "fov-walk-replay.json").write_text(json.dumps(res, indent=1))
        for k, v in res.items():
            print(k, {x: y for x, y in v.items() if x != "events"})
            for e in v["events"]:
                print(" ", e)
        return
    slot = take_slot()
    try:
        replay(
            Path(a.walk[0]),
            Path(a.cues) if a.cues else None,
            a.band_lo,
            Path(a.prior) if a.prior else None,
        )
    finally:
        release(slot)


if __name__ == "__main__":
    main()
