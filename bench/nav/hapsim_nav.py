"""Map-guided routes and the map's reflex priors, against the simulated blind walker.
SIMULATED USER MODEL, NOT A HUMAN STUDY (metasight/hapsim.py has the perception model).

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python bench/nav/hapsim_nav.py [--rooms 100]

The hapsim flat (two rooms, a door, 4 to 7 pieces of furniture) gets heights: about a third of
the pieces are low (a coffee table, 0.35 to 0.5 m), the rest 0.6 to 1.0 m. The camera sees a
piece only when its top is above the bottom edge of the frame, which looks 35 degrees below level
from 1.6 m, so a 0.45 m table vanishes inside about 1.6 m. The map is a little stale: one piece
is new since it was made (the map lacks it) and one mapped piece is gone (a ghost).

Belt conditions (the walker, its steering and its skin are identical in all of them):
  today        camera strips for the three front sectors, the map's side and back sectors from a
               pose that is 4 to 8 s old (a map window), as the server does now
  priors       today + nav.priors / nav.apply every frame with a fresh pose (the phone's ARKit)
  priors_stale today + priors from the 4 to 8 s old position (glasses today: no live position)
  gated        priors, but strong map cues only on the walking path (nav.CORRIDOR_M either side,
               nav.PATH_AHEAD_M ahead, or a predicted contact), off-path ones a slow cue at most,
               and ghost clearing (nav.Ghosts: cells the camera keeps seeing through are dropped)
  gated_none   gated, off-path hidden things get no cue at all
  gated_noghost  gated without ghost clearing
  gated_new    gated, and a new reading on the path is raised a step (nav.apply raise_new)
  *_seen       (gated_seen, gated_none_seen) strong cues only for map cells live depth matched
               earlier in the walk (nav.Ghosts(need_seen=True)): a ghost under the frame is never
               seen first
  *-no<rule>   ablation: -nohidden, -nocontact, -nonew, -nocalm drop that nav.apply rule;
               gated_none_seen-nohidden (contact ahead only) is what the server shadows now
Routes:
  walls        hapsim's walker: a remembered route through the doors that ignores furniture
  nav          nav.Navigator on the map's floor plan, its bearing every frame

Counted per minute of walking: false buzzes (hapsim's definition: a motor turns on with nothing
real within 2 m of its span), missed buzzes (something real comes within 0.7 m of a sector and
that motor and both neighbours stay quiet for 0.3 s), and alarms alongside (a fast or solid onset
on something the body is not heading into within 2 s). Writes bench/results/nav/hapsim_nav.json.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metasight import hapsim as h
from metasight import nav, pan
from metasight.belt import MODES, plan

OUT = ROOT / "bench" / "results" / "nav"
EYE_H, BELOW_DEG = 1.6, 35.0
MAP_PERIOD_S, MAP_AGE_S = 4.0, 4.0  # a map pose every window, already this old when it lands
MISS_M, MISS_HOLD_S = 0.7, 0.3


@dataclass
class World:
    room: h.Room
    tops: np.ndarray  # per rect, metres
    map_rects: np.ndarray  # what the map has: walls, furniture minus the new piece, plus a ghost
    map_tops: np.ndarray
    plan: nav.FloorPlan
    truth: nav.FloorPlan  # the room as it is now, for the shortest route
    mapped: np.ndarray  # per real rect: on the map or not (new)


def world(seed: int, door_m: float = 0.9) -> World:
    room = h.make_room(seed)
    rng = np.random.default_rng(seed + 555)
    rects = room.rects.copy()
    if door_m < 0.9:  # narrow the door from its top side
        rects[4, 3] += 0.9 - door_m
        room = h.Room(rects, room.n_walls, room.start, room.goal, room.size, room.seed)
    nf = len(rects) - room.n_walls
    tops = np.full(len(rects), 2.5)
    low = rng.random(nf) < 0.35
    tops[room.n_walls :] = np.where(low, rng.uniform(0.35, 0.5, nf), rng.uniform(0.6, 1.0, nf))
    keep = np.ones(len(rects), bool)
    keep[room.n_walls + int(rng.integers(nf))] = False  # new since the map was made
    m_rects, m_tops = rects[keep], tops[keep]
    for _ in range(50):  # a ghost: mapped, gone now; not on the start, goal or door
        fw, fh = rng.uniform(0.4, 1.0, 2)
        x, y = rng.uniform(0.5, room.size[0] - fw - 0.5), rng.uniform(0.5, room.size[1] - fh - 0.5)
        box = (x, y, x + fw, y + fh)
        wx = rects[4, 0] + 0.05
        if (
            h._clear(box, room.start[:2], 1.0)
            and h._clear(box, room.goal, 0.8)
            and not (box[0] - 0.8 < wx < box[2] + 0.8)
            and all(
                h._clear(box, ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2), 0.9)
                for r in rects[room.n_walls :]
            )
        ):
            m_rects = np.vstack([m_rects, box])
            m_tops = np.append(m_tops, rng.uniform(0.4, 1.0))
            break
    w, hh = room.size
    boxes = [(r[1], r[0], r[3], r[2], t) for r, t in zip(m_rects, m_tops, strict=True)]
    fp = nav.plan_from_rects(boxes, -0.3, -0.3, hh + 0.3, w + 0.3)
    real = [(r[1], r[0], r[3], r[2], t) for r, t in zip(rects, tops, strict=True)]
    truth = nav.plan_from_rects(real, -0.3, -0.3, hh + 0.3, w + 0.3)
    return World(room, tops, m_rects, m_tops, fp, truth, keep)


def cast_each(rects: np.ndarray, ox: float, oy: float, angles: np.ndarray) -> np.ndarray:
    """(rays, rects) range from the origin along each ray to each rectangle; slab method."""
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx0, tx1 = (rects[:, 0] - ox) / dx, (rects[:, 2] - ox) / dx
        ty0, ty1 = (rects[:, 1] - oy) / dy, (rects[:, 3] - oy) / dy
    tmin = np.nan_to_num(np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1)), nan=np.inf)
    tmax = np.nan_to_num(np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1)), nan=-np.inf)
    hit = (tmax >= np.maximum(tmin, 0)) & (tmax > 0)
    return np.where(hit, np.maximum(tmin, 0), np.inf)


def camera_strips(wd: World, x, y, heading, rng) -> list[float]:
    """What the depth camera reads per strip: only pieces whose top clears the frame's bottom."""
    b = np.radians(pan.strip_bearings(h.STRIPS, h.HFOV))
    sub = np.array([-0.5, 0.0, 0.5]) * math.radians(h.HFOV / h.STRIPS)
    r = cast_each(wd.room.rects, x, y, (heading - (b[:, None] + sub[None, :])).ravel())
    under = np.arctan2(EYE_H - wd.tops[None, :], np.maximum(r, 1e-3)) > math.radians(BELOW_DEG)
    r = np.where(under, np.inf, r).min(axis=1).reshape(h.STRIPS, 3).min(axis=1)
    r = np.minimum(r, h.MAX_RANGE) * rng.lognormal(0, 0.04, h.STRIPS)
    return [float(d) if d < h.MAX_RANGE - 1e-6 else math.inf for d in r]


def map_sectors(wd: World, x, y, heading) -> list[float]:
    sec = heading - np.radians(np.arange(8)[:, None] * 45 + np.linspace(-22, 22, 5)[None, :])
    r = h.cast(wd.map_rects, x, y, sec.ravel()).reshape(8, 5).min(axis=1)
    return [float(d) if d < 3.0 else math.inf for d in r]


def heading_into(rects, x, y, heading, v) -> float:
    """Seconds until the body, carrying straight on, touches a real rectangle (inf past 2 s)."""
    if v < 0.1:
        return math.inf
    for k in range(1, 21):
        px, py = x + v * 0.1 * k * math.cos(heading), y + v * 0.1 * k * math.sin(heading)
        if h._clearance(rects, px, py) < 0.05:
            return 0.1 * k
    return math.inf


def run(wd: World, cond: str, route: str, seed: int = 0, max_s: float = 90.0) -> dict:
    room, belt, skin = wd.room, h.Belt(scheme="sector"), h.DEFAULT_SKIN
    rng = np.random.default_rng(seed * 7919 + room.seed)
    x, y, heading = room.start
    navi = None
    if route == "nav":
        navi = nav.Navigator(wd.plan)
        if not navi.set_goal([room.goal[1], 0.0, room.goal[0]]):
            return {"planned": False, "success": False}
        wps = [room.goal]
    else:
        if not h.route(room, False):
            return {"planned": False, "success": False}
        wps = h.waypoints(room)
    best = nav.Navigator(wd.truth)
    best.set_goal([room.goal[1], 0.0, room.goal[0]])
    g0 = best.step([y, EYE_H, x], heading)
    shortest = g0.route_m if g0 is not None else math.nan
    felt = h._Felt()
    pending: deque = deque()
    live = [(0, 0)] * 8
    t, wi, dist, coll, in_contact = 0.0, 0, 0.0, 0, False
    want, speed, last_v, last_sense, side, side_until = 0.0, 0.0, 0.0, -1.0, 0, 0.0
    k_rise = 1 - math.exp(-h.DT / belt.spinup_s)
    lag = int(h.REACT_S / h.DT)
    hist: deque = deque(maxlen=int(12 / h.DT))  # (t, x, y, heading) for the map's old poses
    map_pose = (t, x, y, heading)
    prev_on, prev_strong = [False] * 8, [False] * 8
    danger_since = [None] * 8
    missed_now = [False] * 8
    fb = miss = along = fb_prior = 0
    fb_rule = {"contact": 0, "hidden": 0, "ghost_near": 0}  # what made the priors' false onsets
    ghost = wd.map_rects[-1] if len(wd.map_rects) > wd.mapped.sum() else None
    pri = None
    miss_kind = {"wall": 0, "mapped": 0, "new": 0}
    did_tot = {"hidden": 0, "calm": 0, "new": 0, "contact": 0}
    pri_ms: list[float] = []
    base = cond.split("-")[0].removesuffix("_seen")
    need = cond.split("-")[0].endswith("_seen")
    ghosts = nav.Ghosts(need) if base in ("gated", "gated_none", "gated_new") else None
    ghost_n = 0
    last_heading = heading
    n = 0
    while t < max_s:
        hist.append((t, x, y, heading))
        if t - map_pose[0] >= MAP_PERIOD_S + MAP_AGE_S:  # a window lands: its pose is MAP_AGE_S old
            map_pose = next((p for p in hist if p[0] >= t - MAP_AGE_S), hist[-1])
        if t - last_sense >= h.SENSE_DT - 1e-9:
            dt_s = t - last_sense if last_sense >= 0 else h.SENSE_DT
            last_sense = t
            truth = h.sense(room, x, y, heading, rng)
            strips = camera_strips(wd, x, y, heading, rng)
            n3 = len(strips)
            thirds = [min(strips[i * n3 // 3 : (i + 1) * n3 // 3]) for i in range(3)]
            path = min(strips[n3 // 2 - 1 : n3 // 2 + 2])
            ttc = path / last_v if last_v > 0.05 else math.inf
            _, mx, my, mh = map_pose
            ms = map_sectors(wd, mx, my, mh)
            k = round(h.wrap_deg(math.degrees(heading - mh)) / 45.0) % 8
            ms = [ms[(i - k) % 8] for i in range(8)]  # turned left k sectors: ahead is now right
            merged = list(ms)
            merged[7], merged[0], merged[1] = thirds
            cmd, _ = plan(merged, [True] * 8, path, ttc, math.inf)
            before = list(cmd)
            if cond != "today":
                if cond != "priors_stale":
                    px, py, err = x, y, 0.05
                else:
                    px, py = mx, my
                    err = last_v * (t - map_pose[0])
                yaw_rate = h.wrap_deg(math.degrees(heading - last_heading)) / max(dt_s, 1e-3)
                pri = nav.priors(
                    wd.plan,
                    [py, EYE_H, px],
                    heading,
                    thirds,
                    h.HFOV,
                    speed_mps=last_v,
                    yaw_rate=math.radians(yaw_rate),
                    pos_err_m=err,
                    eye_h_m=EYE_H,
                    below_deg=BELOW_DEG,
                    ghosts=ghosts,
                )
                pri_ms.append(pri.ms)
                ghost_n += pri.ghosts
                if not cond.startswith("gated"):  # as first built: no path gate
                    pri.path_hidden, pri.on_path = list(pri.hidden), [True] * 8
                off = "none" if cond.startswith("gated_none") else "slow"
                drop = cond.split("-no")[1] if "-no" in cond else ""  # ablation: gated-nocontact
                if drop == "hidden":
                    pri.hidden = pri.path_hidden = [math.inf] * 8
                elif drop == "new":
                    pri.known = [None] * 8
                elif drop == "contact":
                    pri.contact_s = math.inf
                new = not cond.startswith("gated") or cond.startswith("gated_new")
                did = nav.apply(cmd, pri, merged, off_path=off, raise_new=new)
                for key, val in did.items():
                    did_tot[key] += val
                if drop == "calm":
                    cmd = [b if a[0] < b[0] else a for a, b in zip(cmd, before, strict=True)]
            last_heading = heading
            on = [i > 0 and m not in (MODES["off"], MODES["unknown"]) for m, i in cmd]
            strong = [m in (MODES["fast"], MODES["solid"]) for m, _ in cmd]
            into = heading_into(room.rects, x, y, heading, last_v)
            for m in range(8):
                if on[m] and not prev_on[m] and not h._real_near(truth, m, 2.0):
                    fb += 1
                    if cmd[m] != before[m]:  # the priors made or changed this cue
                        fb_prior += 1
                        alone = list(before)
                        nav.apply(alone, replace(pri, contact_s=math.inf), merged, off, new)
                        fb_rule["hidden" if alone[m] == cmd[m] else "contact"] += 1
                        if ghost is not None:
                            gx = max(ghost[0] - x, 0, x - ghost[2])
                            gy = max(ghost[1] - y, 0, y - ghost[3])
                            fb_rule["ghost_near"] += math.hypot(gx, gy) < 1.5
                if strong[m] and not prev_strong[m] and math.isinf(into):
                    along += 1
                danger = truth[1][m] < MISS_M
                quiet = not (on[m] or on[(m - 1) % 8] or on[(m + 1) % 8])
                if danger and quiet:
                    danger_since[m] = t if danger_since[m] is None else danger_since[m]
                    if t - danger_since[m] >= MISS_HOLD_S and not missed_now[m]:
                        miss += 1
                        missed_now[m] = True
                        far = [
                            max(r[0] - x, 0, x - r[2]) ** 2 + max(r[1] - y, 0, y - r[3]) ** 2
                            for r in room.rects
                        ]
                        i = int(np.argmin(far))
                        kind = "wall" if i < room.n_walls else "mapped" if wd.mapped[i] else "new"
                        miss_kind[kind] += 1
                else:
                    danger_since[m], missed_now[m] = None, False
            prev_on, prev_strong = on, strong
            pending.append((t + belt.latency_s, cmd))
        while pending and pending[0][0] <= t:
            live = pending.popleft()[1]
        drive = np.array([h.duty(m, i, t, belt) for m, i in live])
        felt.amp += (drive - felt.amp) * k_rise
        felt.adapt += np.where(felt.amp > skin.floor, h.DT / skin.adapt_s, -2 * h.DT / skin.adapt_s)
        felt.adapt = np.clip(felt.adapt, 0, 1)
        felt.hist.append(felt.amp * (1 - skin.adapt_max * felt.adapt))
        if n % 5 == 0:
            hh = list(felt.hist)
            win = np.array(hh[: max(1, len(hh) - lag)][-int(h.WINDOW_S / h.DT) :])
            if navi is not None:
                g = navi.step([y, EYE_H, x], heading)
                goal_b = g.bearing_deg if g is not None else 0.0
            else:
                gx, gy = wps[wi]
                goal_b = h.wrap_deg(-math.degrees(math.atan2(gy - y, gx - x) - heading))
            was = side
            want, speed, side = h.steer(h.perceive(win, belt, skin, rng), goal_b, belt, side)
            if side and not was:
                side_until = t + 2.5
            elif side and t > side_until:
                side = 0
        n += 1
        dpsi = max(-h.TURN_MAX * h.DT, min(h.TURN_MAX * h.DT, want))
        heading -= dpsi
        want -= dpsi
        v = speed * (0.5 if abs(want) > math.radians(60) else 1.0)
        nx_, ny_ = x + v * h.DT * math.cos(heading), y + v * h.DT * math.sin(heading)
        hit = h._push_out(room.rects, nx_, ny_)
        if hit is not None:
            nx_, ny_ = hit
            coll += 0 if in_contact else 1
            in_contact, last_v = True, 0.0
        else:
            in_contact, last_v = False, v
        dist += math.dist((x, y), (nx_, ny_))
        x, y = nx_, ny_
        t += h.DT
        if math.dist((x, y), wps[wi]) < 0.45:
            wi += 1
            if wi >= len(wps):
                break
    ok = wi >= len(wps)
    wrong = 0  # cleared cells that are a real thing (the ghost clearing's own false alarms)
    for key in ghosts.stale if ghosts is not None else ():
        gi = round(key / 1_000_003)
        cz, cx = (gi + 0.5) * nav.CELL, (key - gi * 1_000_003 + 0.5) * nav.CELL
        e = nav.CELL
        wrong += any(r[0] - e < cx < r[2] + e and r[1] - e < cz < r[3] + e for r in room.rects)
    return {
        "planned": True,
        "success": ok,
        "time_s": t,
        "path_m": dist,
        "shortest_m": shortest,
        "collisions": coll,
        "false": fb,
        "false_prior": fb_prior,
        "false_prior_rule": fb_rule,
        "missed": miss,
        "missed_kind": miss_kind,
        "along": along,
        "did": did_tot,
        "ghost_cells": ghost_n,
        "ghost_cells_real": wrong,
        "prior_ms": float(np.median(pri_ms)) if pri_ms else None,
        "prior_ms_p95": float(np.percentile(pri_ms, 95)) if pri_ms else None,
    }


def summarize(rs: list[dict]) -> dict:
    planned = [r for r in rs if r["planned"]]
    ok = [r for r in planned if r["success"]]
    mins = sum(r["time_s"] for r in planned) / 60 or 1e-9
    out = {
        "runs": len(rs),
        "planned": len(planned) / len(rs),
        "success": len(ok) / len(rs),
        "collisions_per_run": float(np.mean([r["collisions"] for r in planned]))
        if planned
        else None,
        "time_s": float(np.mean([r["time_s"] for r in ok])) if ok else None,
        "efficiency": float(np.mean([r["shortest_m"] / r["path_m"] for r in ok])) if ok else None,
        "false_per_min": sum(r["false"] for r in planned) / mins,
        "false_from_priors_per_min": sum(r["false_prior"] for r in planned) / mins,
        "false_from_priors_by_rule_per_min": {
            k: sum(r["false_prior_rule"][k] for r in planned) / mins
            for k in ("contact", "hidden", "ghost_near")
        },
        "missed_per_min": sum(r["missed"] for r in planned) / mins,
        "runs_with_a_miss": float(np.mean([r["missed"] > 0 for r in planned])) if planned else None,
        "missed_per_min_by_kind": {
            k: sum(r["missed_kind"][k] for r in planned) / mins for k in ("wall", "mapped", "new")
        },
        "along_per_min": sum(r["along"] for r in planned) / mins,
        "ghost_cells_per_run": float(np.mean([r["ghost_cells"] for r in planned] or [0])),
        "ghost_cells_real_per_run": float(np.mean([r["ghost_cells_real"] for r in planned] or [0])),
        "prior_changes_per_min": {
            k: sum(r["did"][k] for r in planned) / mins
            for k in ("hidden", "calm", "new", "contact")
        },
    }
    ms = [r["prior_ms"] for r in planned if r["prior_ms"] is not None]
    if ms:
        out["prior_ms_median"] = float(np.median(ms))
        out["prior_ms_p95"] = float(
            np.median([r["prior_ms_p95"] for r in planned if r["prior_ms_p95"]])
        )
    return out


def job(args):
    seed, door, cond, route = args
    return (door, cond, route), run(world(seed, door), cond, route)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=100)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--first", type=int, default=2000)
    ap.add_argument("--out", default="hapsim_nav.json")
    ap.add_argument("--conds", help="only these conditions, door 0.9 / nav, e.g. today,gated")
    a = ap.parse_args()
    seeds = range(a.first, a.first + a.rooms)  # 3000+ to tune, 2000+ held out
    cases = [
        (0.9, "today", "walls"),
        (0.9, "today", "nav"),
        (0.9, "priors", "walls"),
        (0.9, "priors", "nav"),
        (0.9, "priors_stale", "walls"),
        (0.9, "priors_stale", "nav"),
        (0.65, "today", "walls"),
        (0.65, "today", "nav"),
        (0.65, "priors", "nav"),
        (0.9, "gated", "nav"),
        (0.9, "gated", "walls"),
        (0.65, "gated", "nav"),
    ]
    if a.conds:
        cases = [(0.9, c, "nav") for c in a.conds.split(",")]
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(job, [(s, *c) for c in cases for s in seeds], chunksize=4)
    by: dict = {}
    for key, r in res:
        by.setdefault(key, []).append(r)
    # Fixed inflation at the door: hapsim's own 0.32 m, the planner `walls` uses, vs nav's.
    fixed = {}
    for door in () if a.conds else (0.9, 0.75, 0.65):
        n_fixed = n_nav = 0
        for s in seeds:
            wd = world(s, door)
            n_fixed += bool(h.route(wd.room, False))
            n_nav += nav.Navigator(wd.plan).set_goal([wd.room.goal[1], 0, wd.room.goal[0]])
        fixed[f"{door:.2f}"] = {"fixed_0.32": n_fixed / a.rooms, "nav": n_nav / a.rooms}
    out = {
        "label": "simulated user model, not a human study",
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "rooms": a.rooms,
        "seeds": [seeds.start, seeds.stop],
        "camera": {"hfov": h.HFOV, "eye_h": EYE_H, "below_deg": BELOW_DEG},
        "map": {"period_s": MAP_PERIOD_S, "age_s": MAP_AGE_S, "new": 1, "ghost": 1},
        "route_found": fixed,
        "results": {f"door {k[0]:.2f} / {k[1]} / {k[2]}": summarize(v) for k, v in by.items()},
        # per room, in seed order, for paired comparisons: false, minutes, collisions, mapped misses
        "per_room": {
            f"door {k[0]:.2f} / {k[1]} / {k[2]}": [
                [
                    r["false"],
                    round(r["time_s"] / 60, 3),
                    r["collisions"],
                    r["missed_kind"]["mapped"],
                ]
                if r["planned"]
                else None
                for r in v
            ]
            for k, v in by.items()
        },
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
