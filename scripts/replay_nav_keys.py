"""Replay "take me to my keys" on a recorded ARKit walk with nav.Navigator.

    PYTHONPATH=. .venv/bin/python scripts/replay_nav_keys.py [--walk ID]

The floor plan is the house map's occupancy (data/home, MapAnything) and the goal is its "keys"
pin. The walker is the ARKit pose at each cached frame, so the headings are his. Per frame the
belt gets the live scheme's obstacle command (sectors with Persist(2)) and the Guide overlay
(pan.with_guide) for the route's bearing.

The walk's poses are in ARKit's frame and the house map is in the map solve's, and no map poses
were saved for the walk. Two fits join them, and both are loose (their numbers are in the JSON):
`align_pins` puts the walk's pins where the camera saw them in ARKit's frame, and `fit_home`
matches same-label pins between the walk and the house map. A map built from the walk's own
depth strips was tried first and rejected: it filled the area he stood in with walls.
Writes bench/results/nav-replay-keys.{json,png}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from make_sector_labels import CACHE, STEP, _poses  # noqa: E402

from metasight import nav, pan  # noqa: E402
from metasight.belt import MODES, plan  # noqa: E402
from metasight.server import arkit_pose  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OCC = nav.CELL  # 0.1 m


def align_pins(walk: Path, pins: list[dict]) -> tuple[np.ndarray, dict]:
    """Fit y-rotation + translation putting the map-frame pins where the camera saw them.

    Each pin was last seen at `ts` and first at `first_ts`; at those times it was somewhere in
    the camera's view, 0.3 to 4 m out. The cost is zero anywhere in view and grows toward the
    edges, so it fixes the transform loosely; `info` says how loosely."""
    poses = [json.loads(line) for line in (walk / "poses.jsonl").read_text().splitlines()]
    ts = np.array([p["ts"] / 1000 for p in poses])
    ok = np.array([p["track"] == "tracking" for p in poses])
    mats = [arkit_pose(p["m"]) for p in poses]
    obs = []
    for pin in pins:
        for t in {pin["ts"], pin["first_ts"]}:
            c = mats[int(np.argmin(np.abs(ts - t) + 1e3 * ~ok))]
            obs.append((np.array(pin["pos"]), c[:3, 3], c[:3, 2], c[:3, 0], c[:3, 1]))

    def cost(x: np.ndarray) -> float:
        r, tot = rot(x[0]), 0.0
        for p, pos, f, right, down in obs:
            v = r @ p + x[1:] - pos
            z = v @ f
            if z < 0.3:
                e = 1.0 + (0.3 - z)
            else:
                ha, va = math.atan2(v @ right, z), math.atan2(v @ down, z)
                e = 0.5 * (ha / 0.35) ** 2 + 0.5 * (va / 0.57) ** 2 + max(0.0, z - 4) ** 2
            tot += min(e, 4.0)
        return tot / len(obs)

    fits = []
    for th in np.radians(np.arange(0, 360, 20)):
        for ty in (-1.6, -1.2, -0.8):
            r = minimize(cost, [th, 0, ty, 0], method="Nelder-Mead", options={"maxiter": 600})
            fits.append((float(r.fun), r.x))
    fits.sort(key=lambda f: f[0])
    best = fits[0][1]
    distinct = [f for f in fits if abs(math.remainder(f[1][0] - best[0], 2 * math.pi)) > 0.5]
    info = {
        "yaw_deg": round(math.degrees(math.remainder(best[0], 2 * math.pi)), 1),
        "translation": [round(float(v), 2) for v in best[1:]],
        "cost": round(fits[0][0], 3),
        "best_cost_at_a_different_yaw": round(distinct[0][0], 3) if distinct else None,
    }
    return best, info


def to_ar(x: np.ndarray, p) -> np.ndarray:
    c, s = math.cos(x[0]), math.sin(x[0])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.asarray(p, float) + x[1:]


def rot(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def fit_home(walk_pins: list[dict], home_pins: list[dict]) -> tuple[float, np.ndarray, dict]:
    """Yaw and translation taking the walk's map frame to the house map's: two same-label pin
    pairs propose it, the one most walk pins agree with (a same-label home pin within 0.35 m)
    wins."""
    wp = [(np.array(p["pos"]), p["label"].lower()) for p in walk_pins]
    hp = [(np.array(p["pos"]), p["label"].lower()) for p in home_pins]
    pairs = [(i, j) for i, w in enumerate(wp) for j, h in enumerate(hp) if w[1] == h[1]]

    def inliers(r: np.ndarray, t: np.ndarray) -> int:
        n = 0
        for pos, lab in wp:
            q = r @ pos + t
            ds = [math.hypot(q[0] - h[0][0], q[2] - h[0][2]) for h in hp if h[1] == lab]
            n += bool(ds) and min(ds) < 0.35
        return n

    best = (0, 0.0, np.zeros(3))
    for (i1, j1), (i2, j2) in itertools.combinations(pairs, 2):
        if i1 == i2 or j1 == j2:
            continue
        a, b = wp[i1][0] - wp[i2][0], hp[j1][0] - hp[j2][0]
        if (
            math.hypot(a[0], a[2]) < 0.8
            or abs(math.hypot(a[0], a[2]) - math.hypot(b[0], b[2])) > 0.3
        ):
            continue
        for th in (
            math.atan2(b[0], b[2]) - math.atan2(a[0], a[2]),
            math.atan2(a[0], a[2]) - math.atan2(b[0], b[2]),
        ):
            r = rot(th)
            v = r @ a
            if math.hypot(v[0] - b[0], v[2] - b[2]) > 0.3:
                continue
            t = hp[j1][0] - r @ wp[i1][0]
            n = inliers(r, t)
            if n > best[0]:
                best = (n, th, t)
    n, th, t = best
    return (
        th,
        t,
        {"pins_agreeing": int(n), "of": len(wp), "yaw_deg": round(math.degrees(th) % 360, 1)},
    )


def home_frame(x: np.ndarray, th2: float, t2: np.ndarray):
    """ARKit world -> house map: invert the pin placement, then apply the home fit."""
    r1, r2 = rot(x[0]), rot(th2)

    def point(p):
        return r2 @ (r1.T @ (np.asarray(p, float) - x[1:])) + t2

    def vector(v):
        return r2 @ (r1.T @ np.asarray(v, float))

    return point, vector


def obstacle_cmds(d, name: str) -> list[list[tuple[int, int]]]:
    """The live scheme per cached frame: sectors, Persist(2)."""
    filt, out = pan.Persist(2), []
    inf = math.inf
    for st, p, tt in zip(d[f"{name}/strips"], d[f"{name}/path"], d[f"{name}/ttc"], strict=True):
        st = filt(list(st))
        n = len(st)
        f = [min(st[i * n // 3 : (i + 1) * n // 3]) for i in range(3)]
        merged = [f[1], f[2], inf, inf, inf, inf, inf, f[0]]
        known = [True, True, False, False, False, False, False, True]
        out.append(plan(merged, known, p, tt, inf)[0])
    return out


def replay(walk: Path, d, plan_: nav.FloorPlan, goal, point, vector) -> dict:
    name = walk.name
    nv = nav.Navigator(plan_)
    reached = nv.set_goal(goal, "keys")
    cmds = obstacle_cmds(d, name)
    rows = (walk / "frames.jsonl").read_text().splitlines()
    seqs = [json.loads(r)["seq"] for r in rows][::STEP]
    ts, poses = d[f"{name}/ts"], _poses(walk)
    frames, walker = [], []
    viol = solid = shown_n = 0
    for t, seq, obs in zip(ts, seqs, cmds, strict=False):
        pose = poses.get(seq)
        if pose is None:
            continue
        f = vector(pose[:3, 2])
        if math.hypot(f[0], f[2]) < 0.3:  # looking at the floor or ceiling: no heading
            continue
        pos, yaw = point(pose[:3, 3]), math.atan2(f[0], f[2])
        walker.append({"t": float(t - ts[0]), "pos": pos, "yaw": yaw})
        g = nv.step(pos, yaw) if reached else None
        row = {
            "t": round(float(t - ts[0]), 2), "pos": [round(float(v), 2) for v in pos],
            "yaw_deg": round(math.degrees(yaw), 1), "obstacle": [list(c) for c in obs],
        }  # fmt: skip
        has_solid = any(m == MODES["solid"] for m, _ in obs)
        solid += has_solid
        if g is None:
            row["guidance"] = None
        else:
            final, shown = pan.with_guide(obs, g.bearing_deg)
            shown_n += shown
            if has_solid and final != obs:  # STOP must leave the command untouched
                viol += 1
            row["guidance"] = {
                "bearing_deg": round(g.bearing_deg, 1), "route_m": round(g.route_m, 2),
                "arrived": g.arrived, "cmd": [list(c) for c in final], "guide_shown": shown,
            }  # fmt: skip
        frames.append(row)
    return {
        "reached": reached, "frames": frames, "nv": nv, "walker": walker,
        "guide_shown": shown_n, "solid_frames": solid, "violations": viol,
    }  # fmt: skip


def route_polyline(nv: nav.Navigator, pos) -> list[tuple[float, float]]:
    p = nv.plan
    c = p.ij(pos[0], pos[2])
    if not p.free(*c):
        c = p.snap(*c)
    r, out, k = p.blocked.shape[1], [], c[0] * p.blocked.shape[1] + c[1]
    while k >= 0:
        out.append((float(p.xs[k // r]), float(p.zs[k % r])))
        k = int(nv.next[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", default="20261028-135040-iphone")
    a = ap.parse_args()
    walk = ROOT / "data" / "walks" / a.walk
    d = np.load(CACHE)
    walk_pins = json.loads((walk / "pins.json").read_text())
    home = json.loads((ROOT / "data" / "home" / "home.json").read_text())
    keys = max((p for p in home["pins"] if p["label"] == "keys"), key=lambda p: p["seen"])
    fp = nav.floor_plan(np.load(ROOT / "data" / "home" / "occ.npy"))
    x, info = align_pins(walk, walk_pins)
    th2, t2, home_info = fit_home(walk_pins, home["pins"])
    point, vector = home_frame(x, th2, t2)
    walk_keys = max((p for p in walk_pins if p["label"] == "keys"), key=lambda p: p["seen"])
    wk = rot(th2) @ np.array(walk_keys["pos"]) + t2
    home_info["walk_keys_pin_vs_home_keys_pin_m"] = round(
        math.hypot(wk[0] - keys["pos"][0], wk[2] - keys["pos"][2]), 2
    )
    goal = np.array(keys["pos"])
    res = replay(walk, d, fp, goal, point, vector)
    nv, fr = res["nv"], res["frames"]
    g = [f["guidance"] for f in fr if f["guidance"]]
    start = np.array(fr[0]["pos"])
    route = route_polyline(nv, start) if res["reached"] else []
    cells = [fp.ij(px, pz) for px, pz in route]
    clear = [float(fp.clear[c]) for c in cells if fp.inside(*c)]
    bear = np.array([abs(v["bearing_deg"]) for v in g])
    sc = fp.ij(start[0], start[2])
    route_m = None
    if res["reached"] and fp.inside(*sc) and np.isfinite(nv.togo[sc]):
        route_m = round(float(nv.togo[sc]), 2)
    bearing_stats = None
    if len(bear):
        bearing_stats = {
            "median": round(float(np.median(bear)), 1),
            "p90": round(float(np.percentile(bear, 90)), 1),
        }

    def med(gs):
        return round(float(np.median([abs(v["bearing_deg"]) for v in gs])), 1) if gs else None

    out = {
        "walk": a.walk, "goal_pin": {"id": keys["id"], "label": "keys", "map": "data/home"},
        "pin_alignment_arkit": info, "walk_to_home_fit": home_info,
        "route_found": res["reached"],
        "route_m_from_start": route_m,
        "start_in_free_floor": bool(fp.free(*sc)),
        "route_min_clearance_m": round(min(clear), 2) if clear else None,
        "route_cells_blocked": int(sum(not fp.free(*c) for c in cells)),
        "frames": len(fr), "frames_with_guidance": len(g),
        "arrived_frames": sum(v["arrived"] for v in g),
        "bearing_abs_deg": bearing_stats,
        "stop_frames": res["solid_frames"], "stop_over_guide_violations": res["violations"],
        "guide_shown_frames": res["guide_shown"],
    }  # fmt: skip
    # How the answer moves if the walk's placement in the house is off by a few degrees of yaw.
    sens = {}
    for dth in (-20, -10, 10, 20):
        pt, vc = home_frame(x, th2 + math.radians(dth), t2)
        r2 = replay(walk, d, fp, goal, pt, vc)
        g2 = [f["guidance"] for f in r2["frames"] if f["guidance"]]
        sens[f"yaw{dth:+d}"] = {
            "route_found": r2["reached"],
            "median_abs_bearing": med(g2),
            "violations": r2["violations"],
        }  # fmt: skip
    out["alignment_sensitivity"] = sens
    (ROOT / "bench" / "results" / "nav-replay-keys.json").write_text(
        json.dumps({**out, "frames_log": fr})
    )
    print(json.dumps(out, indent=1))
    plot(res, route, fp, goal, wk, ROOT / "bench" / "results" / "nav-replay-keys.png")


def plot(res, route, fp, goal, walk_keys, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    ext = [fp.xs[0] - OCC / 2, fp.xs[-1] + OCC / 2, fp.zs[0] - OCC / 2, fp.zs[-1] + OCC / 2]
    img = np.full(fp.blocked.shape + (3,), 0.93)
    img[fp.seen] = 0.82  # the map saw floor here
    img[~np.isfinite(fp.cost) & ~fp.blocked] = (0.95, 0.85, 0.85)  # too close to a wall
    img[fp.blocked] = (0.15, 0.15, 0.2)
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=ext, interpolation="nearest")
    w = res["walker"]
    wp = np.array([[v["pos"][0], v["pos"][2]] for v in w])
    ax.scatter(wp[:, 0], wp[:, 1], c=[v["t"] for v in w], cmap="viridis", s=8, label="his pose")
    for v in w[::25]:
        ax.arrow(v["pos"][0], v["pos"][2], 0.18 * math.sin(v["yaw"]), 0.18 * math.cos(v["yaw"]),
                 width=0.01, color="C1")  # fmt: skip
    if route:
        r = np.array(route)
        ax.plot(r[:, 0], r[:, 1], "r-", lw=2, label="route from his start")
    ax.plot(goal[0], goal[2], "*", ms=18, c="gold", mec="k", label="keys (house map pin)")
    ax.plot(walk_keys[0], walk_keys[2], "x", ms=10, c="m", label="keys (walk's own pin, placed)")
    ax.set_aspect("equal")
    ax.set_xlabel("house map x (m)")
    ax.set_ylabel("house map z (m)")
    ax.set_title("Take me to my keys: ARKit walk replayed on the house map")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(wp[:, 0].min() - 3, wp[:, 0].max() + 3)
    ax.set_ylim(wp[:, 1].min() - 3, wp[:, 1].max() + 3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
