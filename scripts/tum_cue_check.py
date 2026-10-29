"""The reflex cue against true metric depth: TUM RGB-D sequences, Kinect depth as the label.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/tum_cue_check.py --extract
    PYTHONPATH=. .venv/bin/python scripts/tum_cue_check.py
    PYTHONPATH=. .venv/bin/python scripts/tum_cue_check.py --knobs  # after the line above

Stage 1 (heavy, under the heavy-job lock) runs every 3rd colour frame of each sequence in
data/tum through the live safety loop the way replay_false_buzz.extract does: the glasses
profile, a fresh loop per sequence, the Kinect's own hfov. Frames are centre-cropped from 640x480
to 640x360 first, because the live grid is 16:9 and a 4:3 frame squeezed into it would bend the
floor. The camera height is the sequence's own (the floor fitted in the Kinect depth), since the
profile's 1.22 m is a phone at the chest. Stage 1 caches the raw depth on the live grid, the
Kinect depth on the same grid (zeros are holes, not near) and the loop's own strips.

Stage 2 builds the truth strips by running safety.analyse on the Kinect depth at scale 1 (the
same floor removal and obstacle rule as the live strips, so floor is not an obstacle in either),
and replays the live geometry at a fixed scale: the profile's 0.45 and the scale the Kinect says
is metric. Cues are replay_false_buzz.commands(..., "sector", 2, 26), as in score_labels.py.

  (a) scale: median and quartiles of Kinect depth / (raw depth x 0.45) per grid pixel, Kinect
      0.3 to 4 m; metric scale = 0.45 x median. The same ratio on strips is given too.
  (b) false front cue: the front motor felt while the Kinect shows nothing inside 2 m in the
      centre third; events per minute and % of front-cue-on frames.
  (c) over-urgent: the live centre third reads under 0.5 m (solid) while the Kinect reads 0.5 m
      or more; % of those frames. The Kinect sees nothing under about 0.5 m, so frames whose
      centre third is mostly holes are counted apart.
  (d) missed inside 1 m: the Kinect shows an obstacle under 1 m in any third and no motor is felt.

--knobs diagnoses the pioneer_slam false front cues at the metric scale (which pixels, and what
the Kinect says they are) and sweeps analyse's floor and path knobs on a local copy of it.

Writes bench/results/tum-cue-check.json. The scale applies to the Kinect camera only; the cue
rates at the metric scale measure the cue scheme itself.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from replay_false_buzz import STRIPS, commands  # noqa: E402
from score_labels import OFF_M, SLOW_M, runs  # noqa: E402

from metasight import safety  # noqa: E402
from metasight.safety import BODY_HALF_M, GRID, _fit_floor, analyse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TUM = ROOT / "data" / "tum"
SEQS = (  # fr1_room and fr3_long_office_household are here without depth: skipped
    "rgbd_dataset_freiburg2_pioneer_slam",
    "rgbd_dataset_freiburg1_360",
    "rgbd_dataset_freiburg2_desk",
)
FX = {"freiburg1": 517.3, "freiburg2": 520.9, "freiburg3": 535.4}  # TUM's published intrinsics
CACHE = ROOT / "bench" / "results" / "tum-strips.npz"
OUT = ROOT / "bench" / "results" / "tum-cue-check.json"
STEP = 3  # every 3rd colour frame: 10 per second
CROP = (60, 420)  # rows kept: 640x480 -> 640x360
KINECT = 5000.0  # depth PNG units per metre
LIVE_SCALE = 0.45  # metasight.glasses.toml depth_scale
FAST_M = 0.5  # the solid cue
NEAR_M, FAR_M = 0.3, 4.0  # Kinect range trusted for the scale ratio


def _list(path: Path) -> list[tuple[float, str]]:
    rows = [ln.split() for ln in path.read_text().splitlines() if not ln.startswith("#")]
    return [(float(t), f) for t, f in rows]


def frames(seq: Path) -> list[tuple[float, Path, Path]]:
    """Every STEP-th colour frame with the nearest depth frame, when one is within 20 ms."""
    depth = _list(seq / "depth.txt")
    dt = np.array([t for t, _ in depth])
    out = []
    for t, f in _list(seq / "rgb.txt")[::STEP]:
        i = int(np.abs(dt - t).argmin())
        if abs(dt[i] - t) < 0.02:
            out.append((t, seq / f, seq / depth[i][1]))
    return out


def kinect_grid(path: Path) -> np.ndarray:
    """Kinect depth in metres on the live grid; a cell is a hole (nan) when half of it is."""
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[CROP[0] : CROP[1]].astype(np.float32) / KINECT
    ok = (d > 0).astype(np.float32)
    num = cv2.resize(d, GRID, interpolation=cv2.INTER_AREA)
    den = cv2.resize(ok, GRID, interpolation=cv2.INTER_AREA)
    return np.where(den >= 0.5, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)


def _points(d: np.ndarray, hfov: float) -> np.ndarray:
    h, w = d.shape
    f = 0.5 * w / np.tan(np.radians(hfov) / 2)
    u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    return np.stack([u * d / f, v * d / f, d], -1)


def camera_height(grids: np.ndarray, hfov: float) -> tuple[float, int]:
    """Median height of the camera over the floor fitted in the Kinect depth, and frame count."""
    cs = [
        p[1]
        for g in grids[::10]
        if (p := _fit_floor(_points(g, hfov), None, 0.04, 0.2, 2.5)) is not None
    ]
    return float(np.median(cs)), len(cs)


def _strips(grid: np.ndarray) -> list[float]:
    g = grid.astype(np.float32) * 2 / 100
    g[grid == 255] = math.inf
    return [float(g[:, c].min()) for c in np.array_split(np.arange(g.shape[1]), STRIPS)]


def extract() -> None:
    from metasight.config import Settings
    from metasight.safety import SafetyLoop

    out: dict[str, np.ndarray] = {}
    for name in SEQS:
        seq = TUM / name
        if not (seq / "depth.txt").exists():
            print(name, "no depth: skipped")
            continue
        fr = frames(seq)
        hfov = math.degrees(2 * math.atan(320 / FX[name.split("_")[2]]))
        kin = np.array([kinect_grid(p) for _, _, p in fr])
        cam_h, n_h = camera_height(kin, hfov)
        s = Settings.load(ROOT / "metasight.glasses.toml")
        s.hfov_deg, s.eye_height_m = hfov, cam_h
        loop = SafetyLoop(s)
        raws, strips, path, ttc, scale = [], [], [], [], []
        for _, rgb, _ in fr:
            bgr = cv2.imread(str(rgb))[CROP[0] : CROP[1]]
            raw = loop.estimate_depth(bgr)
            scene, _, tt = loop._geometry(raw)  # reflex() without the depth copy
            raws.append(cv2.resize(raw.astype(np.float32), GRID, interpolation=cv2.INTER_AREA))
            strips.append(_strips(scene.grid))
            path.append(scene.path_m)
            ttc.append(tt)
            scale.append(loop.scale)
        out[f"{name}/ts"] = np.array([t for t, _, _ in fr])
        out[f"{name}/raw"] = np.array(raws, np.float32)
        out[f"{name}/kinect"] = kin
        out[f"{name}/strips"] = np.array(strips)
        out[f"{name}/path"] = np.array(path)
        out[f"{name}/ttc"] = np.array(ttc)
        out[f"{name}/scale"] = np.array(scale)
        out[f"{name}/hfov"] = np.array([hfov])
        out[f"{name}/cam_h"] = np.array([cam_h, n_h])
        print(name, len(fr), "frames, camera", round(cam_h, 2), "m", flush=True)
    np.savez(CACHE, **out)


def replay(grids, ts, scale: float, hfov: float, eye_h: float, fn=analyse, keep=None) -> tuple:
    """safety._geometry at a fixed scale (the profile's depth_scale set to it), on frame time.
    `fn` is analyse or a knob variant; `keep`, a list, collects (scene, this frame's floor)."""
    plane, plane_t = None, -math.inf
    hist: deque[tuple[float, float]] = deque()
    strips, path, ttc = [], [], []
    for t, g in zip(ts, grids, strict=True):
        prev = plane if t - plane_t < 1.5 else None
        hold = plane[0] if plane is not None and t - plane_t < 5.0 else None
        scene, pl = fn(g, scale, scale, hfov, eye_h, prev, hold)
        if pl is not None:
            plane, plane_t = pl, t
            scene, _ = fn(g, scale, scale, hfov, eye_h, pl)
        if keep is not None:
            keep.append((scene, pl))
        hist.append((t, scene.path_m))
        while hist and t - hist[0][0] > 0.6:
            hist.popleft()
        pts = [(a, d) for a, d in hist if math.isfinite(d)]
        tt = math.inf
        if len(pts) >= 5 and math.isfinite(scene.path_m):
            a, d = np.array(pts).T
            slope = np.polyfit(a - a[-1], d, 1)[0]
            if slope < -0.15:
                tt = scene.path_m / -slope
        strips.append(_strips(scene.grid))
        path.append(scene.path_m)
        ttc.append(tt)
    return np.array(strips), np.array(path), np.array(ttc)


def _q(x: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"median": round(float(med), 3), "q1": round(float(q1), 3), "q3": round(float(q3), 3)}


def depth_scale(raw: np.ndarray, kin: np.ndarray, live: np.ndarray, true: np.ndarray) -> dict:
    ok = np.isfinite(kin) & (kin >= NEAR_M) & (kin <= FAR_M) & (raw > 0.05)
    r = kin[ok] / (raw[ok] * LIVE_SCALE)
    per_frame = [
        float(np.median(k[m] / (w[m] * LIVE_SCALE)))
        for k, w, m in zip(kin, raw, ok, strict=True)
        if m.sum() >= 200
    ]
    so = np.isfinite(live) & np.isfinite(true) & (true >= NEAR_M) & (true <= FAR_M)
    return {
        "pixel_ratio_true_over_live": {**_q(r), "pixels": int(ok.sum())},
        "per_frame_median_ratio": {**_q(np.array(per_frame)), "frames": len(per_frame)},
        "strip_ratio_true_over_live": {**_q(true[so] / live[so]), "strips": int(so.sum())},
        "metric_scale": round(LIVE_SCALE * float(np.median(r)), 3),
    }


def _rate(bad: np.ndarray, base: np.ndarray, minutes: float) -> dict:
    return {
        "events_per_min": round(runs(bad) / minutes, 1),
        "frames": int(bad.sum()),
        "of_frames": int(base.sum()),
        "pct": round(100 * float(bad.sum()) / max(1, int(base.sum())), 1),
    }


def cue_scores(live, path, ttc, hfov: float, true, t_path, holes, minutes: float) -> dict:
    on = commands(live, path, ttc, hfov, "sector", 2, 26)
    front_on, any_on = on[:, 0], on.any(axis=1)
    thirds = [true[:, i * STRIPS // 3 : (i + 1) * STRIPS // 3].min(axis=1) for i in range(3)]
    t_centre, t_near = thirds[1], np.minimum.reduce(thirds)
    urgent = live[:, 5:10].min(axis=1) < FAST_M
    over = urgent & (t_centre >= FAST_M)
    blind = holes >= 0.5  # the Kinect centre third is mostly holes: maybe too close to read
    near1 = t_near < SLOW_M
    false_front = front_on & (t_centre >= OFF_M)
    # what the front motor reads: the nearer of the centre wedge and the walking path
    false_vs_front = front_on & (np.minimum(t_centre, t_path) >= OFF_M)
    return {
        "cue_on_pct": round(100 * float(any_on.mean()), 1),
        "front_cue_on_pct": round(100 * float(front_on.mean()), 1),
        "b_false_front_cue": {
            **_rate(false_front, front_on, minutes),
            "kinect_centre_mostly_holes": int((false_front & blind).sum()),
        },
        "b_false_front_vs_kinect_path": _rate(false_vs_front, front_on, minutes),
        "b_false_any_cue": _rate(any_on & (t_near >= OFF_M), any_on, minutes),
        "c_over_urgent": {
            **_rate(over, urgent, minutes),
            "kinect_centre_mostly_holes": int((over & blind).sum()),
        },
        "d_missed_inside_1m": _rate(near1 & ~any_on, near1, minutes),
    }


def score_one(d, name: str) -> dict:
    ts, raw, kin = d[f"{name}/ts"], d[f"{name}/raw"], d[f"{name}/kinect"]
    hfov, (cam_h, n_h) = float(d[f"{name}/hfov"][0]), d[f"{name}/cam_h"]
    minutes = float((ts[-1] - ts[0]) / 60)
    true, t_path, _ = replay(np.nan_to_num(kin, nan=0.0), ts, 1.0, hfov, cam_h)
    third = slice(GRID[0] // 3, 2 * GRID[0] // 3)
    holes = np.isnan(kin[:, :, third]).mean(axis=(1, 2))
    fixed, _, _ = replay(raw, ts, LIVE_SCALE, hfov, cam_h)
    sc = depth_scale(raw, kin, fixed, true)
    res = {
        "frames": len(ts),
        "minutes": round(minutes, 2),
        "hfov_deg": round(hfov, 1),
        "camera_height_m": {"median": round(float(cam_h), 2), "frames": int(n_h)},
        "a_depth_scale": sc,
        "kinect_near_pct": {
            "centre_inside_2m": round(100 * float((true[:, 5:10].min(axis=1) < OFF_M).mean()), 1),
            "any_inside_1m": round(100 * float((true.min(axis=1) < SLOW_M).mean()), 1),
        },
        "live_running_scale": _q(d[f"{name}/scale"]),
    }
    res["live_adaptive"] = cue_scores(
        d[f"{name}/strips"], d[f"{name}/path"], d[f"{name}/ttc"], hfov, true, t_path, holes, minutes
    )
    for key, s in ((f"fixed_{LIVE_SCALE}", LIVE_SCALE), ("fixed_metric", sc["metric_scale"])):
        st, p, tt = replay(raw, ts, s, hfov, cam_h)
        res[key] = {"scale": s, **cue_scores(st, p, tt, hfov, true, t_path, holes, minutes)}
    return res


def show(name: str, r: dict) -> None:
    a = r["a_depth_scale"]["pixel_ratio_true_over_live"]
    print(
        f"{name}: {r['frames']} frames, {r['minutes']} min, camera {r['camera_height_m']}, "
        f"true/live {a['median']} ({a['q1']}..{a['q3']}) -> metric "
        f"{r['a_depth_scale']['metric_scale']}"
    )
    for key in ("live_adaptive", f"fixed_{LIVE_SCALE}", "fixed_metric"):
        c = r[key]
        print(
            f"  {key:14s} cue on {c['cue_on_pct']}%  false front {c['b_false_front_cue']}  "
            f"over-urgent {c['c_over_urgent']}  missed<1m {c['d_missed_inside_1m']}"
        )


def variant(tol: float = 0.04, hmin: float = 0.05, half: float = BODY_HALF_M, drop: float = 0.0):
    """analyse with knobs moved, from a local copy of its source (metasight/ is not edited):
    floor-fit tolerance, obstacle height over the floor, path half-width, and the share of the
    lowest grid rows taken out of the walking path."""
    src = inspect.getsource(safety.analyse)
    for a, b in (
        ("0.04 / scale", f"{tol} / scale"),
        ("(hgt > 0.05)", f"(hgt > {hmin})"),
        ("(np.abs(x) < BODY_HALF_M)", f"(np.abs(x) < {half})"),
        ("    sectors = []\n", f"    corridor[int({1 - drop} * h) :] = False\n    sectors = []\n"),
    ):
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    ns = dict(vars(safety))
    exec(src, ns)  # noqa: S102 (our own module's source, with numbers swapped)
    return ns["analyse"]


KNOBS = (
    {},
    {"tol": 0.06},
    {"tol": 0.08},
    {"hmin": 0.08},
    {"hmin": 0.10},
    {"hmin": 0.15},
    {"drop": 0.05},
    {"drop": 0.10},
    {"drop": 0.15},
    {"drop": 0.20},
    {"half": 0.35},
    {"half": 0.30},
    {"half": 0.25},
)


def _pixel_kinds(kin: np.ndarray, P_true: np.ndarray, tplane, near: np.ndarray) -> dict:
    """What the Kinect says the near-path pixels are, in true metres."""
    n, c = tplane
    k = kin[near]
    p = P_true[near]
    kh = p @ n + c
    kr = np.linalg.norm(p - np.outer(p @ n, n), axis=1)
    filled = np.nan_to_num(kin, nan=0.0)
    k3 = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(filled, k3) - cv2.erode(np.where(np.isnan(kin), 99, kin), k3)
    edge = spread[near] > 0.3
    kind = np.full(len(k), "other", object)
    hole = np.isnan(k)
    floor = ~hole & (np.abs(kh) < 0.05)
    real = ~hole & ~floor & (kr < OFF_M)
    kind[hole], kind[floor], kind[real] = "kinect_hole", "floor_off_plane", "real_object"
    kind[~hole & ~floor & ~real & edge] = "edge_noise"
    return {"kind": kind, "true_h": kh, "true_r": kr}


def diagnose(d, name: str, scale: float) -> dict:
    """Pioneer false front cues at the metric scale on frames whose floor fit succeeded: which
    pixels put the walking path under 2 m, and what the Kinect says they are."""
    ts, raw, kin = d[f"{name}/ts"], d[f"{name}/raw"], d[f"{name}/kinect"]
    hfov, cam_h = float(d[f"{name}/hfov"][0]), float(d[f"{name}/cam_h"][0])
    kt, kl = [], []
    true, _, _ = replay(np.nan_to_num(kin, nan=0.0), ts, 1.0, hfov, cam_h, keep=kt)
    st, p, tt = replay(raw, ts, scale, hfov, cam_h, keep=kl)
    on = commands(st, p, tt, hfov, "sector", 2, 26)
    false_front = on[:, 0] & (true[:, 5:10].min(axis=1) >= OFF_M)
    found = np.array([pl is not None for _, pl in kl])
    tplanes, last = [], None
    for _, pl in kt:
        last = pl if pl is not None else last
        tplanes.append(last)
    h, w = GRID[1], GRID[0]
    rows_all, cols_all, frames = [], [], []
    for i in np.where(false_front & found)[0]:
        scene, (n, c) = kl[i]
        near = (scene.grid < OFF_M * 50) & scene.corridor
        source = "path"
        if not near.any():  # the centre wedge, not the path, is under 2 m
            near = scene.grid < OFF_M * 50
            near[:, : w // 3], near[:, 2 * w // 3 :] = False, False
            source = "centre_wedge" if near.any() else "neither"
        if not near.any():  # carried over by the persistence filter
            frames.append({"i": int(i), "source": source, "kind": "persist_carry"})
            continue
        r, col = np.nonzero(near)
        live_h = (_points(raw[i], hfov)[near] @ n + c) * scale
        pk = _pixel_kinds(kin[i], _points(np.nan_to_num(kin[i], nan=0.0), hfov), tplanes[i], near)
        kinds, cnt = np.unique(pk["kind"], return_counts=True)
        ok = np.isfinite(pk["true_h"]) & (pk["kind"] != "kinect_hole")
        frames.append(
            {
                "i": int(i),
                "source": source,
                "pixels": int(near.sum()),
                "kind": str(kinds[cnt.argmax()]),
                "kind_share": round(float(cnt.max() / cnt.sum()), 2),
                "row_from_top_median": round(float(np.median(r) / h), 2),
                "strips": sorted({int(x * STRIPS // w) for x in col}),
                "live_h_median_m": round(float(np.median(live_h)), 3),
                "true_h_median_m": round(float(np.median(pk["true_h"][ok])), 3)
                if ok.any()
                else None,
                "kinect_path_m": round(float(kt[i][0].path_m), 2),
            }
        )
        rows_all.append(r / h)
        cols_all.append(col * STRIPS // w)
    rows_all, cols_all = np.concatenate(rows_all), np.concatenate(cols_all)
    kinds, cnt = np.unique([f["kind"] for f in frames], return_counts=True)
    th = np.array([f["true_h_median_m"] for f in frames if f.get("true_h_median_m") is not None])
    return {
        "scale": scale,
        "false_front_frames": int(false_front.sum()),
        "with_floor_fit": len(frames),
        "frame_kinds": {str(k): int(v) for k, v in zip(kinds, cnt, strict=True)},
        "sources": {
            k: sum(f["source"] == k for f in frames) for k in ("path", "centre_wedge", "neither")
        },
        "kinect_path_under_2m_frames": sum(f.get("kinect_path_m", 9) < OFF_M for f in frames),
        "near_pixel_rows_from_top": {
            f"{lo:.1f}-{lo + 0.1:.1f}": round(
                float(((rows_all >= lo) & (rows_all < lo + 0.1)).mean()), 3
            )
            for lo in np.arange(0.4, 1.0, 0.1)
        },
        "near_pixel_strips": {int(s): int((cols_all == s).sum()) for s in range(STRIPS)},
        "live_h_median_m": _q(np.array([f["live_h_median_m"] for f in frames if "pixels" in f])),
        "true_h_median_m": _q(th),
        "frames": frames,
    }


def knob_sweep(d, res: dict) -> dict:
    """Every knob setting at each sequence's metric scale, against the stock Kinect truth."""
    out = {}
    for name in res:
        ts, raw, kin = d[f"{name}/ts"], d[f"{name}/raw"], d[f"{name}/kinect"]
        hfov, cam_h = float(d[f"{name}/hfov"][0]), float(d[f"{name}/cam_h"][0])
        minutes = float((ts[-1] - ts[0]) / 60)
        true, t_path, _ = replay(np.nan_to_num(kin, nan=0.0), ts, 1.0, hfov, cam_h)
        third = slice(GRID[0] // 3, 2 * GRID[0] // 3)
        holes = np.isnan(kin[:, :, third]).mean(axis=(1, 2))
        s = res[name]["fixed_metric"]["scale"]
        for kn in KNOBS:
            key = ",".join(f"{k}={v}" for k, v in kn.items()) or "stock"
            st, p, tt = replay(raw, ts, s, hfov, cam_h, fn=variant(**kn))
            c = cue_scores(st, p, tt, hfov, true, t_path, holes, minutes)
            out.setdefault(key, {})[name] = {
                "cue_on_pct": c["cue_on_pct"],
                "b_false_front": c["b_false_front_cue"],
                "b_false_front_vs_kinect_path": c["b_false_front_vs_kinect_path"],
                "c_over_urgent": c["c_over_urgent"],
                "d_missed_inside_1m": c["d_missed_inside_1m"],
            }
            b, u, m = c["b_false_front_cue"], c["c_over_urgent"], c["d_missed_inside_1m"]
            print(
                f"{key:10s} {name[14:]:22s} false {b['events_per_min']}/min {b['frames']}/"
                f"{b['of_frames']}  over {u['frames']}/{u['of_frames']}  missed<1m "
                f"{m['frames']}/{m['of_frames']}",
                flush=True,
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="run the depth model (heavy)")
    ap.add_argument("--knobs", action="store_true", help="diagnose pioneer, sweep analyse knobs")
    a = ap.parse_args()
    if a.extract:
        extract()
        return
    if a.knobs:
        d, res = np.load(CACHE), json.loads(OUT.read_text())
        pio = "rgbd_dataset_freiburg2_pioneer_slam"
        diag = diagnose(d, pio, res[pio]["fixed_metric"]["scale"])
        print(json.dumps({k: v for k, v in diag.items() if k != "frames"}, indent=1))
        res["pioneer_false_front_diagnosis"] = diag
        res["knobs_at_metric_scale"] = knob_sweep(
            d, {k: v for k, v in res.items() if k.startswith("rgbd")}
        )
        OUT.write_text(json.dumps(res, indent=1))
        return
    d = np.load(CACHE)
    res = {}
    for name in sorted({k.split("/")[0] for k in d.files}):
        res[name] = score_one(d, name)
        show(name, res[name])
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
