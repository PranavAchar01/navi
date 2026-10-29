"""Score the house map on a walk: drift against ground truth, and duplicate pins on revisits.

python scripts/bench_map.py data/tum/rgbd_dataset_freiburg1_room      # TUM RGB-D folder
python scripts/bench_map.py data/pov/indoor1.mp4 --seconds 90         # any video, no truth

MapAnything outputs are cached per window under data/cache/, so changing the stitching or the
pin rules re-scores a walk in seconds. Writes a JSON line per run to bench/results/map.jsonl and
a top-down picture (trail, truth, pins) next to it.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metasight.align import map_resolution, umeyama  # noqa: E402
from metasight.config import Settings  # noqa: E402
from metasight.memory import Keyframe, MemoryEngine  # noqa: E402

REVISIT_GAP_S = 15.0  # two keyframes this far apart in time ...
REVISIT_NEAR_M = 0.5  # ... and this close in truth are a revisit
DUP_RADIUS_M = {"object": 1.0, "landmark": 2.5}  # same label this close is very likely one thing


class CachedRunner:
    """MapAnything with its outputs kept on disk, keyed by the source and the frame indices."""

    def __init__(self, source: str, cache: Path, hfov_deg: float | None = None):
        self.source, self.cache, self._runner = source, cache, None
        self.hfov_deg = hfov_deg
        self.calls = self.hits = 0
        cache.mkdir(parents=True, exist_ok=True)

    def infer(self, frames_bgr: list[np.ndarray]) -> list[dict]:
        self.calls += 1
        h = hashlib.sha1(self.source.encode())
        if self.hfov_deg:
            h.update(f"hfov{self.hfov_deg:.2f}".encode())
        for b in frames_bgr:
            h.update(np.ascontiguousarray(b[::8, ::8]).tobytes())
        key = h.hexdigest()[:16]
        path = self.cache / f"{key}.npz"
        if path.exists():
            self.hits += 1
            z = np.load(path)
            return [
                dict(
                    pts=z["pts"][i].astype(np.float32),
                    mask=z["mask"][i],
                    pose=z["pose"][i],
                    depth_z=z["depth_z"][i].astype(np.float32),
                    rgb=cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255,
                )
                for i, b in enumerate(frames_bgr)
            ]
        if self._runner is None:
            from metasight.memory import MapAnythingRunner

            self._runner = MapAnythingRunner(hfov_deg=self.hfov_deg)
        out = self._runner.infer(frames_bgr)
        np.savez(
            path,
            pts=np.stack([o["pts"] for o in out]).astype(np.float16),
            mask=np.stack([o["mask"] for o in out]),
            pose=np.stack([o["pose"] for o in out]),
            depth_z=np.stack([o["depth_z"] for o in out]).astype(np.float16),
        )
        return out


# ---------------------------------------------------------------------- inputs
def _sharpness(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(cv2.resize(bgr, (320, 240)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _novel(stamps: list[float], read, hfov: float, policy: str = "novel") -> list[int]:
    """Indices the novelty policy keeps (metasight/keyframes.py), fed every frame in order.
    `policy` "novel:10:2.5" sets the turn threshold (degrees) and the longest gap (seconds)."""
    from metasight.keyframes import KeyframePolicy

    knobs = dict(zip(("turn_deg", "max_gap_s"), map(float, policy.split(":")[1:]), strict=False))
    pol, kept = KeyframePolicy(hfov, **knobs), []
    for i, t in enumerate(stamps):
        g = cv2.cvtColor(read(i), cv2.COLOR_BGR2GRAY)
        k = pol.offer(g, float(cv2.Laplacian(cv2.resize(g, (320, 240)), cv2.CV_32F).var()), t, i)
        if k is not None:
            kept.append(k)
    return kept


def _pick(stamps: list[float], hz: float) -> list[list[int]]:
    """Indices grouped into keyframe intervals of 1/hz seconds."""
    groups: list[list[int]] = []
    t_end = -math.inf
    for i, t in enumerate(stamps):
        if t >= t_end:
            groups.append([])
            t_end = (stamps[0] if t_end == -math.inf else t_end) + 1 / hz
            while t >= t_end:
                t_end += 1 / hz
        groups[-1].append(i)
    return groups


# Horizontal field of view of the three TUM Kinects, from their published fx at 640 px.
TUM_HFOV = {"freiburg1": 63.48, "freiburg2": 63.13, "freiburg3": 61.73}


def load_tum(folder: Path, hz: float, seconds: float, policy: str = "slots"):
    rgb = [
        ln.split()
        for ln in (folder / "rgb.txt").read_text().splitlines()
        if ln and not ln.startswith("#")
    ]
    stamps = [float(t) for t, _ in rgb]
    stamps = [t for t in stamps if t - stamps[0] <= seconds]
    frames = []
    if policy.startswith("novel"):
        hfov = next((v for k, v in TUM_HFOV.items() if k in folder.name), 62.0)
        for i in _novel(stamps, lambda i: cv2.imread(str(folder / rgb[i][1])), hfov, policy):
            frames.append((i, stamps[i], cv2.imread(str(folder / rgb[i][1]))))
    for g in _pick(stamps, hz) if not policy.startswith("novel") else []:
        imgs = [cv2.imread(str(folder / rgb[i][1])) for i in g]
        best = max(range(len(g)), key=lambda k: _sharpness(imgs[k]))
        frames.append((g[best], stamps[g[best]], imgs[best]))
    gt = np.array(
        [
            [float(v) for v in ln.split()]
            for ln in (folder / "groundtruth.txt").read_text().splitlines()
            if ln and not ln.startswith("#")
        ]
    )
    return frames, gt


def load_walk(folder: Path, hz: float, seconds: float, policy: str = "slots"):
    """A walk recorded by the iPhone app: frames plus ARKit poses, which serve as the truth."""
    fr = [json.loads(ln) for ln in (folder / "frames.jsonl").read_text().splitlines() if ln]
    stamps = [f["ts"] / 1000 for f in fr]
    stamps = [t for t in stamps if t - stamps[0] <= seconds]

    def read(i):
        return cv2.imread(str(folder / "frames" / fr[i]["file"]))

    if policy.startswith("novel"):
        hfov = json.loads((folder / "camera.json").read_text())["hfov"]
        picks = _novel(stamps, read, hfov, policy)
    else:
        picks = [max(g, key=lambda k: _sharpness(read(k))) for g in _pick(stamps, hz)]
    frames = [(i, stamps[i], read(i)) for i in picks]
    gt = []
    for ln in (folder / "poses.jsonl").read_text().splitlines():
        p = json.loads(ln) if ln else {}
        if p.get("type") == "arpose":
            m = p["m"]  # column-major 4x4: translation in 12..14
            gt.append([p["ts"] / 1000, m[12], m[13], m[14], 0, 0, 0, 1])
    return frames, np.array(gt)


def load_video(path: Path, hz: float, seconds: float, src_hfov: float | None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    raw, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or i / fps > seconds:
            break
        raw.append(f)
        i += 1
    if src_hfov:  # crop to the glasses' portrait view, as the emulator does
        from metasight.glasses_emu import crop_hfov

        cols, _ = crop_hfov(raw[0].shape[1], raw[0].shape[0], src_hfov)
        raw = [f[:, cols] for f in raw]
    stamps = [i / fps for i in range(len(raw))]
    frames = []
    for g in _pick(stamps, hz):
        best = max(g, key=lambda k: _sharpness(raw[k]))
        frames.append((best, stamps[best], raw[best]))
    return frames, None, _cuts(raw, fps)


def _cuts(raw: list[np.ndarray], fps: float) -> list[float]:
    """Times of edits in a downloaded clip: one frame changes far more than its neighbours do
    (a fast head turn changes a run of frames). The glasses stream has none; a step across one
    is the footage jumping, not the map."""
    small = [
        cv2.cvtColor(cv2.resize(f, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in raw
    ]
    d = [float(np.mean(np.abs(b - a))) for a, b in zip(small[:-1], small[1:], strict=True)]
    # Repeated frames (a low frame-rate source padded up) are skipped, or every change is a spike.
    moved = [(i + 1, v) for i, v in enumerate(d) if v > 0.5]
    if not moved:
        return []
    floor = 4 * float(np.median([v for _, v in moved]))
    return [
        i / fps
        for k, (i, v) in enumerate(moved)
        if v > floor
        and v
        > 3 * max(moved[k - 1][1] if k else 0.0, moved[k + 1][1] if k + 1 < len(moved) else 0.0)
    ]


# ---------------------------------------------------------------------- scoring
def _gt_at(gt: np.ndarray, t: float) -> np.ndarray | None:
    i = int(np.searchsorted(gt[:, 0], t))
    cands = [k for k in (i - 1, i) if 0 <= k < len(gt) and abs(gt[k, 0] - t) < 0.05]
    return gt[min(cands, key=lambda k: abs(gt[k, 0] - t)), 1:4] if cands else None


def score_poses(est: dict[int, np.ndarray], times: dict[int, float], gt: np.ndarray) -> dict:
    ids = [k for k in est if _gt_at(gt, times[k]) is not None]
    e = np.array([est[k] for k in ids])
    g = np.array([_gt_at(gt, times[k]) for k in ids])
    s, rot, t = umeyama(e, g)
    al = s * e @ rot.T + t
    err = np.linalg.norm(al - g, axis=1)
    # The map is meant to be in metres, so also score it with scale held at 1: what a wearer hears.
    _, rot1, t1 = umeyama(e, g, with_scale=False)
    err1 = np.linalg.norm(e @ rot1.T + t1 - g, axis=1)
    path = float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum())
    rev = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if times[ids[b]] - times[ids[a]] >= REVISIT_GAP_S:
                if np.linalg.norm(g[a] - g[b]) < REVISIT_NEAR_M:
                    rev.append(float(np.linalg.norm((al[a] - al[b]) - (g[a] - g[b]))))
    return {
        "ate_m": round(float(np.sqrt((err**2).mean())), 3),
        "ate_pct_path": round(100 * float(np.sqrt((err**2).mean())) / max(path, 1e-6), 2),
        "path_m": round(path, 2),
        "scale": round(float(s), 3),
        "ate_metric_m": round(float(np.sqrt((err1**2).mean())), 3),
        "revisit_pairs": len(rev),
        "revisit_err_med_m": round(float(np.median(rev)), 3) if rev else None,
        "revisit_err_p90_m": round(float(np.percentile(rev, 90)), 3) if rev else None,
        "_aligned": al,
        "_gt": g,
        "_sim": (s, rot, t),
    }


def score_jumps(trail: np.ndarray, ts: list[float], cuts: list[float]) -> dict:
    """Without ground truth: a step between keyframes far longer than the usual one is the map
    breaking (people do not teleport). Steps across an edit in the footage are left out."""
    if len(trail) < 3:
        return {}
    steps = np.linalg.norm(np.diff(trail[:, [0, 2]], axis=0), axis=1)
    cut = np.array([any(a < c <= b for c in cuts) for a, b in zip(ts, ts[1:], strict=False)])
    steps = steps[~cut]
    med = float(np.median(steps))
    return {
        "step_med_m": round(med, 3),
        "jumps": int(np.sum(steps > 3 * med + 0.05)),
        "worst_step_m": round(float(steps.max()), 2),
        "cuts": int(cut.sum()),
    }


def score_pins(pins) -> dict:
    by: dict[tuple[str, str], list] = {}
    for p in pins:
        by.setdefault((p.kind, p.label), []).append(p)
    dups = 0
    for (kind, _), ps in by.items():
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                if math.dist(ps[a].pos, ps[b].pos) < DUP_RADIUS_M[kind]:
                    dups += 1
    return {
        "pins": len(pins),
        "objects": sum(p.kind == "object" for p in pins),
        "landmarks": sum(p.kind == "landmark" for p in pins),
        "single_sightings": sum(p.seen == 1 for p in pins),
        "dup_pairs": dups,
        "per_label": {f"{k[0][0]}:{k[1]}": len(v) for k, v in sorted(by.items())},
    }


def draw(path: Path, trail: np.ndarray, gt: np.ndarray | None, pins, title: str) -> None:
    pts = [trail[:, [0, 2]]] + ([gt[:, [0, 2]]] if gt is not None else [])
    pts += [np.array([[p.pos[0], p.pos[2]] for p in pins])] if pins else []
    allp = np.concatenate(pts)
    lo, hi = allp.min(0) - 0.5, allp.max(0) + 0.5
    size = 900
    sc = (size - 40) / max(hi - lo)

    def px(xz):
        return (int(20 + (xz[0] - lo[0]) * sc), int(size - 20 - (xz[1] - lo[1]) * sc))

    img = np.full((size, size, 3), 250, np.uint8)
    if gt is not None:
        cv2.polylines(img, [np.array([px(p) for p in gt[:, [0, 2]]])], False, (60, 160, 60), 2)
    cv2.polylines(img, [np.array([px(p) for p in trail[:, [0, 2]]])], False, (200, 90, 30), 2)
    for p in pins:
        c = (30, 30, 220) if p.kind == "object" else (120, 120, 120)
        q = px((p.pos[0], p.pos[2]))
        cv2.circle(img, q, 5, c, -1)
        cv2.putText(img, p.label, (q[0] + 6, q[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    cv2.putText(img, title, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--hz", type=float, default=None, help="keyframes per second")
    ap.add_argument("--policy", default="slots", help="slots, novel, or novel:TURN:GAP")
    ap.add_argument("--src-hfov", type=float, default=None, help="crop video to glasses view")
    ap.add_argument("--finder", default="yoloe", help="yoloe (free) or hybrid (bills SAM)")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-loops", action="store_true")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument(
        "--intrinsics", action="store_true", help="give MapAnything the lens (TUM: per camera)"
    )
    ap.add_argument("--mode", default="home", help="home or outside (short-horizon settings)")
    a = ap.parse_args()

    s = Settings.load()
    s.finder = a.finder
    if a.mode == "outside":
        s = replace(s, **s.outside)
    s.window, s.overlap = a.window or s.window, a.overlap or s.overlap
    hz = a.hz or s.keyframe_hz
    src = Path(a.source)
    t_load = time.perf_counter()
    if (src / "frames.jsonl").exists():
        frames, gt = load_walk(src, hz, a.seconds, a.policy)
        cuts = []
    elif src.is_dir():
        frames, gt = load_tum(src, hz, a.seconds, a.policy)
        cuts = []
    else:
        frames, gt, cuts = load_video(src, hz, a.seconds, a.src_hfov)
    print(f"{len(frames)} keyframes in {time.perf_counter() - t_load:.1f}s", flush=True)

    from ultralytics import YOLO

    from metasight.providers import make_finder

    class NoFinder:
        name = "none"

        def find(self, bgr, labels):
            return []

    hfov = None
    if a.intrinsics:
        hfov = next((v for k, v in TUM_HFOV.items() if k in src.name), None) or s.hfov_deg
    key = f"{src.resolve()}|{a.src_hfov}|{hz}" + (f"|{a.policy}" if a.policy != "slots" else "")
    runner = CachedRunner(key, ROOT / "data" / "cache", hfov)
    finder = NoFinder() if a.no_objects else make_finder(s)
    from metasight.loops import Describer

    describer = None if a.no_loops else Describer()
    engine = MemoryEngine(
        s, runner, finder, YOLO(str(s.weights_dir / "yolo26n.pt")), describer=describer
    )
    if a.mode == "outside":
        engine.set_mode("outside")
    kfs, ids, times = [], [], {}
    for n, (idx, t, img) in enumerate(frames):
        w, h = map_resolution(img.shape[1], img.shape[0])
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        kfs.append(Keyframe(n, t, small, hi=img))
        ids.append(idx)
        times[n] = t
    per = s.window - s.overlap
    t0 = time.perf_counter()
    for k in range(0, len(kfs), per):
        chunk = kfs[k : k + per]
        if len(chunk) < 2:
            break
        engine.process_window(chunk)
    est = {k: pose[:3, 3].copy() for k, pose in engine.poses.items()}
    run_s = time.perf_counter() - t0

    out = {
        "source": str(a.source),
        "tag": a.tag,
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "keyframes": len(kfs),
        "hz": hz,
        "policy": a.policy,
        "kf_per_min": round(60 * len(kfs) / max(1e-6, frames[-1][1] - frames[0][1]), 1),
        "run_s": round(run_s, 1),
        "infer_calls": runner.calls,
        "cache_hits": runner.hits,
        "win_scale": [round(float(w[0]), 3) for w in engine._sims],
        "loops_closed": sum(r.get("result") == "closed" for r in engine.loops),
        "loop_checks": [r for r in engine.loops if "shift_m" in r][:12],
        "loop_results": {
            k: sum(r.get("result") == k for r in engine.loops)
            for k in {r.get("result") for r in engine.loops}
        },
    }
    trail = np.array([est[k] for k in sorted(est)])
    out.update(score_jumps(trail, [times[k] for k in sorted(est)], cuts))
    if gt is None:
        np.savez(
            ROOT / "bench" / "results" / f"traj-{src.stem}-{a.tag}.npz",
            est=trail,
            seq=np.array(sorted(est)),
            t=np.array([times[k] for k in sorted(est)]),
        )
    al = gtp = None
    if gt is not None:
        sc = score_poses(est, times, gt)
        al, gtp, sim = sc.pop("_aligned"), sc.pop("_gt"), sc.pop("_sim")
        np.savez(
            ROOT / "bench" / "results" / f"traj-{src.stem}-{a.tag}.npz",
            est=al,
            gt=gtp,
            seq=np.array([k for k in est if _gt_at(gt, times[k]) is not None]),
            stats=np.array([engine._windows]),
        )
        out.update(sc)
    pins = list(engine.pins.pins.values())
    if al is not None:  # draw pins in the truth frame too
        s_, rot, t_ = sim
        for p in pins:
            p.pos = list(s_ * rot @ np.array(p.pos) + t_)
        trail = al
    out.update(score_pins(pins))
    res = ROOT / "bench" / "results"
    res.mkdir(parents=True, exist_ok=True)
    name = f"map-{src.stem}{'-' + a.tag if a.tag else ''}"
    draw(res / f"{name}.png", trail, gtp, pins, f"{src.stem} {a.tag}")
    with open(res / "map.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_label"}))
    print("per label:", out["per_label"])


if __name__ == "__main__":
    main()
