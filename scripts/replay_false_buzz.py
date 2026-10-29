"""Transient buzzes on recorded walks, with the real depth stream (no simulated noise).

Walks recorded on the iPhone (data/walks) are run through the safety loop exactly as the server
does (stage 1, heavy: depth model, cached in bench/results/walk-strips.npz). Stage 2 turns the
cached depth strips into belt commands with the old sector scheme and with panning, with and
without the persistence filter, and counts TRANSIENT BUZZES: a motor cue that turns on and is
gone again in under 0.5 s. The walks have no ground truth, so this is a proxy that over- and
under-counts: a real obstacle passing the edge of the view also flickers, and a persistent
phantom is not counted. It is a ceiling for flicker, not a false-alarm rate.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/replay_false_buzz.py --extract
    PYTHONPATH=. .venv/bin/python scripts/replay_false_buzz.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from metasight import pan
from metasight.belt import MODES, plan

ROOT = Path(__file__).resolve().parent.parent
WALKS = sorted((ROOT / "data" / "walks").glob("*"))
CACHE = ROOT / "bench" / "results" / "walk-strips.npz"
STRIPS = 15
STEP = 2  # every second recorded frame (about 12 per second)


def extract() -> None:
    import cv2

    from metasight.config import Settings
    from metasight.safety import SafetyLoop

    s = Settings.load()
    loop = SafetyLoop(s)
    out: dict[str, np.ndarray] = {}
    for w in WALKS:
        cam = json.loads((w / "camera.json").read_text())
        rows = [json.loads(line) for line in (w / "frames.jsonl").read_text().splitlines()]
        ts, strips, sect, path, ttc = [], [], [], [], []
        for r in rows[::STEP]:
            bgr = cv2.imread(str(w / "frames" / r["file"]))
            if bgr is None:
                continue
            res = loop.process(bgr)
            g = res.scene.grid.astype(np.float32) * 2 / 100  # cm / 2 -> metres; 255 -> 5.1 m
            g[res.scene.grid == 255] = math.inf
            cols = np.array_split(np.arange(g.shape[1]), STRIPS)
            strips.append([float(g[:, c].min()) for c in cols])
            ts.append(r["ts"] / 1000)
            sect.append(res.sectors)
            path.append(res.scene.path_m)
            ttc.append(res.ttc_s)
        k = w.name
        out[f"{k}/ts"] = np.array(ts)
        out[f"{k}/strips"] = np.array(strips)
        out[f"{k}/sectors"] = np.array(sect)
        out[f"{k}/path"] = np.array(path)
        out[f"{k}/ttc"] = np.array(ttc)
        out[f"{k}/hfov"] = np.array([cam.get("hfov", 63.0)])
        print(k, len(ts), "frames", flush=True)
    np.savez(CACHE, **out)


def _chain(*stages):
    def run(targets, p=pan.DEFAULT):
        for st in stages:
            targets = st(targets, p)
        return targets

    return run


def commands(strips, path, ttc, hfov, scheme, persist, felt=1):
    filt = pan.Persist(persist)
    hold = {
        "panned+hold": pan.Hold,
        "panned+track": pan.Tracker,
        "panned+track+hold": lambda: _chain(pan.Tracker(), pan.Hold()),
    }.get(scheme, lambda: None)()
    cmds = []
    for st, p, tt in zip(strips, path, ttc, strict=True):
        st = filt(list(st))
        if scheme == "sector":
            n = len(st)
            front = [min(st[i * n // 3 : (i + 1) * n // 3]) for i in range(3)]
            merged = [
                front[1],
                front[2],
                math.inf,
                math.inf,
                math.inf,
                math.inf,
                math.inf,
                front[0],
            ]
            known = [True, True, False, False, False, False, False, True]
            c, _ = plan(merged, known, p, tt, math.inf)
        else:
            tg = pan.cluster_strips(st, pan.strip_bearings(len(st), hfov))
            c, _ = pan.plan_pan(tg, [True] * 8, p, tt, math.inf, hold=hold)
        cmds.append([i >= felt and m != MODES["unknown"] for m, i in c])
    return np.array(cmds)


def transient(on: np.ndarray, ts: np.ndarray, max_s: float = 0.5) -> int:
    """Cue runs (per motor) that last less than max_s."""
    n = 0
    for m in range(on.shape[1]):
        col, i = on[:, m], 0
        while i < len(col):
            if col[i]:
                j = i
                while j < len(col) and col[j]:
                    j += 1
                if j < len(col) and i > 0 and ts[min(j, len(ts) - 1)] - ts[i] < max_s:
                    n += 1
                i = j
            else:
                i += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
        return
    d = np.load(CACHE)
    res = {}
    for w in sorted({k.split("/")[0] for k in d.files}):
        ts = d[f"{w}/ts"]
        minutes = (ts[-1] - ts[0]) / 60
        for scheme in ("sector", "panned", "panned+hold", "panned+track", "panned+track+hold"):
            for persist in (1, 2):
                for felt in (1, 26):  # 26/255 is the skin model's 0.10 detection floor
                    on = commands(
                        d[f"{w}/strips"], d[f"{w}/path"], d[f"{w}/ttc"],
                        float(d[f"{w}/hfov"][0]), scheme, persist, felt,
                    )  # fmt: skip
                    n = transient(on, ts)
                    key = f"{w} {scheme} persist={persist} felt>={felt}"
                    res[key] = {
                        "minutes": round(float(minutes), 2),
                        "transient_per_min": round(float(n / minutes), 1),
                        "on_fraction": round(float(on.mean()), 3),
                    }
    tot: dict[str, list[float]] = {}
    for k, v in res.items():
        tot.setdefault(k.split(" ", 1)[1], []).append(v["transient_per_min"])
    for k, v in tot.items():
        print(f"{k:34s} mean transient/min over {len(v)} walks: {sum(v) / len(v):5.1f}")
    (ROOT / "bench" / "results" / "replay-false-buzz.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
