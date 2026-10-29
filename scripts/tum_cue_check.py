"""The reflex cue against true metric depth: TUM RGB-D sequences, Kinect depth as the label.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/tum_cue_check.py --extract
    PYTHONPATH=. .venv/bin/python scripts/tum_cue_check.py

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

Writes bench/results/tum-cue-check.json. The scale applies to the Kinect camera only; the cue
rates at the metric scale measure the cue scheme itself.
"""

from __future__ import annotations

import argparse
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

from metasight.safety import GRID, _fit_floor, analyse  # noqa: E402

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


def replay(grids, ts, scale: float, hfov: float, eye_h: float) -> tuple:
    """safety._geometry at a fixed scale (the profile's depth_scale set to it), on frame time."""
    plane, plane_t = None, -math.inf
    hist: deque[tuple[float, float]] = deque()
    strips, path, ttc = [], [], []
    for t, g in zip(ts, grids, strict=True):
        prev = plane if t - plane_t < 1.5 else None
        hold = plane[0] if plane is not None and t - plane_t < 5.0 else None
        scene, pl = analyse(g, scale, scale, hfov, eye_h, prev, hold)
        if pl is not None:
            plane, plane_t = pl, t
            scene, _ = analyse(g, scale, scale, hfov, eye_h, pl)
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


def cue_scores(live, path, ttc, hfov: float, true, holes, minutes: float) -> dict:
    on = commands(live, path, ttc, hfov, "sector", 2, 26)
    front_on, any_on = on[:, 0], on.any(axis=1)
    thirds = [true[:, i * STRIPS // 3 : (i + 1) * STRIPS // 3].min(axis=1) for i in range(3)]
    t_centre, t_near = thirds[1], np.minimum.reduce(thirds)
    urgent = live[:, 5:10].min(axis=1) < FAST_M
    over = urgent & (t_centre >= FAST_M)
    blind = holes >= 0.5  # the Kinect centre third is mostly holes: maybe too close to read
    near1 = t_near < SLOW_M
    false_front = front_on & (t_centre >= OFF_M)
    return {
        "cue_on_pct": round(100 * float(any_on.mean()), 1),
        "front_cue_on_pct": round(100 * float(front_on.mean()), 1),
        "b_false_front_cue": {
            **_rate(false_front, front_on, minutes),
            "kinect_centre_mostly_holes": int((false_front & blind).sum()),
        },
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
    true, _, _ = replay(np.nan_to_num(kin, nan=0.0), ts, 1.0, hfov, cam_h)
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
        d[f"{name}/strips"], d[f"{name}/path"], d[f"{name}/ttc"], hfov, true, holes, minutes
    )
    for key, s in ((f"fixed_{LIVE_SCALE}", LIVE_SCALE), ("fixed_metric", sc["metric_scale"])):
        st, p, tt = replay(raw, ts, s, hfov, cam_h)
        res[key] = {"scale": s, **cue_scores(st, p, tt, hfov, true, holes, minutes)}
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="run the depth model (heavy)")
    if ap.parse_args().extract:
        extract()
        return
    d = np.load(CACHE)
    res = {}
    for name in sorted({k.split("/")[0] for k in d.files}):
        res[name] = score_one(d, name)
        show(name, res[name])
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
