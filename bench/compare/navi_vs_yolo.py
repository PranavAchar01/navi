"""Navi's obstacle cue against plain YOLO, on hand-labelled frames of the 11-02 glasses take.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python bench/compare/navi_vs_yolo.py

Inputs:
  bench/labels/take1-glasses-path.json   89 frames (one a second) labelled near / clear / unsure
  bench/results/navi-vs-yolo/navi-replay.jsonl
      the take replayed through the real server (scripts/judge_replay.py, glasses profile),
      every safety message it sent, keyed by frame seq

Navi warns on a frame when a front belt motor it commanded is on (slow, fast or solid). The
four-motor belt carries the front on its two diagonal front motors (cells 1 and 7,
metasight/pan.py), so cells 0, 1 and 7 are read. In the glasses profile that means
something in view within cue_off_m = 1.3 m.

Plain YOLO runs here on the same frames, the same YOLO26n Navi uses for labels, with no depth.
It warns two ways:
  yolo_size  a box (conf >= 0.35) whose range from its pixel size and a typical real size for
             the class (bench/compare/depth_vs_truth.py PRIOR) is under 1.3 m and whose lateral
             offset at that range is inside the 0.45 m half path. The best a box-only detector
             can do for distance.
  yolo_any   any confident box covering the middle of the view and reaching the lower half of
             it: no distance at all, an upper bound on what boxes alone can catch.

Latency: YOLO's predict() per frame on the take's frames, here; Navi's per-frame stage times come
from the same replay's safety messages. Writes bench/results/navi-vs-yolo.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.compare.depth_vs_truth import PRIOR  # noqa: E402
from metasight.roomprior import LIVE_CONF  # noqa: E402
from metasight.safety import BODY_HALF_M  # noqa: E402

LABELS = ROOT / "bench" / "labels" / "take1-glasses-path.json"
REPLAY = ROOT / "bench" / "results" / "navi-vs-yolo" / "navi-replay.jsonl"
OUT = ROOT / "bench" / "results" / "navi-vs-yolo.json"
WALK = ROOT / "data" / "walks" / "20261102-030055-glasses"
HFOV_DEG = 44.8  # metasight.glasses.toml, the lens the replay used
OFF_M = 1.3  # metasight.glasses.toml cue_off_m: past it the belt is silent
ON = {"slow", "fast", "solid"}
FRONT = (0, 1, 7)  # ahead and the two front diagonals
LAT_EVERY = 4  # latency on every 4th frame of the take


def front_on(msg: dict) -> bool:
    return any(msg["cues"][i]["mode"] in ON for i in FRONT)


def pct(xs: list[float], q: float) -> float:
    return round(float(np.percentile(xs, q)), 1)


def yolo_warn(r, w: int, h: int) -> tuple[bool, bool]:
    fx = 0.5 * w / math.tan(math.radians(HFOV_DEG) / 2)
    size_hit = any_hit = False
    for b, c, p in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
        strict=True,
    ):
        if p < LIVE_CONF:
            continue
        x0, y0, x1, y1 = map(float, b)
        if x0 < 0.75 * w and x1 > 0.25 * w and y1 > 0.5 * h:
            any_hit = True
        name = r.names[int(c)]
        if name in PRIOR:
            axis, size = PRIOR[name]
            px = (y1 - y0) if axis == "h" else (x1 - x0)
            rng = fx * size / max(px, 1.0)
            lateral = abs((0.5 * (x0 + x1) - w / 2) / fx) * rng
            if rng < OFF_M and lateral < BODY_HALF_M:
                size_hit = True
    return size_hit, any_hit


def score(frames: list[dict], key: str) -> dict:
    out = {}
    near = [f for f in frames if f["label"] == "near"]
    for grp in ("all", "desk/table", "chair/bed", "wall/door"):
        sel = near if grp == "all" else [f for f in near if f["group"] == grp]
        hit = sum(f[key] for f in sel)
        out[f"recall_{grp}"] = {"hit": hit, "n": len(sel), "rate": round(hit / len(sel), 3)}
    clear = [f for f in frames if f["label"] == "clear"]
    out["warned_on_clear"] = {"hit": sum(f[key] for f in clear), "n": len(clear)}
    return out


def main() -> None:
    from ultralytics import YOLO

    lab = json.loads(LABELS.read_text())
    navi = {}
    for line in REPLAY.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "safety":
            navi[r["seq"]] = r
    frames = [f for f in lab["frames"] if f["label"] != "unsure"]

    models = {
        "coreml_384": (YOLO(str(ROOT / "yolo26n.mlpackage"), task="detect"), 384, None),
        "pt_640_mps": (YOLO(str(ROOT / "yolo26n.pt")), 640, "mps"),
    }
    for f in frames:
        s = navi[f["seq"]]
        f["navi"] = front_on(s)
        f["navi_front_m"] = s["sectors"][0]
        f["navi_path_m"] = s["path"]
    lat: dict[str, list[float]] = {}
    for name, (m, imgsz, dev) in models.items():
        blank = np.zeros((896, 504, 3), np.uint8)
        for _ in range(5):
            m.predict(blank, imgsz=imgsz, device=dev, verbose=False)
        for f in frames:
            img = cv2.imread(str(WALK / "frames" / f["file"]))
            r = m.predict(img, imgsz=imgsz, device=dev, verbose=False, conf=0.2)[0]
            f[f"{name}_size"], f[f"{name}_any"] = yolo_warn(r, img.shape[1], img.shape[0])
            f[f"{name}_boxes"] = sorted(
                {
                    r.names[int(c)]
                    for c, p in zip(r.boxes.cls, r.boxes.conf, strict=True)
                    if p >= LIVE_CONF
                }
            )
        ts = []
        files = sorted((WALK / "frames").iterdir(), key=lambda p: int(p.stem.split("-")[1]))
        for p in files[::LAT_EVERY]:
            img = cv2.imread(str(p))
            t0 = time.perf_counter()
            m.predict(img, imgsz=imgsz, device=dev, verbose=False, conf=0.2)
            ts.append(1000 * (time.perf_counter() - t0))
        lat[name] = ts

    stages = {
        k: [s["ms"][k] for s in navi.values() if s.get("ms", {}).get(k) is not None]
        for k in ("depth", "detect", "total")
    }
    cue_on_all = sum(front_on(s) for s in navi.values())
    res = {
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "data": {
            "walk": lab["walk"],
            "video": lab["video"],
            "labels": str(LABELS.relative_to(ROOT)),
            "labelled": len(lab["frames"]),
            "scored": len(frames),
            "unsure_dropped": len(lab["frames"]) - len(frames),
            "rule": lab["rule"],
        },
        "methods": {
            "navi": "a front belt motor on (slow/fast/solid) in the replay of the take through "
            "metasight.server, glasses profile: Depth Anything V2 metric depth, floor fit, "
            "walking-path range, 3-frame median and 0.25 s hold (scripts/judge_replay.py)",
            "yolo_size": "YOLO26n boxes only, conf >= 0.35; range from box size and a typical "
            "class size under 1.3 m, inside the 0.45 m half path",
            "yolo_any": "YOLO26n boxes only, conf >= 0.35; any box over the middle of the view "
            "reaching its lower half (no distance: an upper bound for boxes)",
        },
        "recall": {
            "navi": score(frames, "navi"),
            **{f"{m}_{k}": score(frames, f"{m}_{k}") for m in models for k in ("size", "any")},
        },
        "navi_cue_on_share_all_replayed_frames": {
            "on": cue_on_all,
            "n": len(navi),
            "rate": round(cue_on_all / len(navi), 3),
        },
        "latency_ms": {
            "navi_stages_from_replay": {
                k: {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95)} for k, v in stages.items()
            },
            **{
                f"yolo_{m}_predict": {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95)}
                for m, v in lat.items()
            },
        },
        "frames": frames,
    }
    rc, ry, ry640 = (
        res["recall"]["navi"],
        res["recall"]["coreml_384_size"],
        res["recall"]["pt_640_mps_size"],
    )
    lab_caveat = (
        "one labeller, by eye, on one 88 s take in one small bedroom; distance judged from the "
        "picture, not measured; neighbouring frames are alike (51 desk frames are mostly him "
        "standing or sitting at the same desk)"
    )
    res["summary"] = [
        {
            "metric": f"recall, {g}",
            "navi": rc[f"recall_{g}"],
            "yolo": ry[f"recall_{g}"],
            "yolo_640": ry640[f"recall_{g}"],
            "n": rc[f"recall_{g}"]["n"],
            "method": "share of frames labelled near where the method warned; yolo = "
            "yolo_size on Navi's own detector (Core ML, 384 px), yolo_640 = stock yolo26n.pt "
            "at 640 px",
            "caveats": lab_caveat,
        }
        for g in ("all", "wall/door", "chair/bed", "desk/table")
    ] + [
        {
            "metric": "compute per frame, ms",
            "navi": res["latency_ms"]["navi_stages_from_replay"]["total"],
            "navi_buzz_stage": res["latency_ms"]["navi_stages_from_replay"]["depth"],
            "yolo": res["latency_ms"]["yolo_coreml_384_predict"],
            "yolo_640": res["latency_ms"]["yolo_pt_640_mps_predict"],
            "n": {"navi": len(stages["total"]), "yolo": len(lat["coreml_384"])},
            "method": "Navi: the server's own stage timers on every replayed frame (depth = "
            "depth net + floor fit, the stage the buzz waits on; total adds the label "
            "detector). YOLO: wall time of predict() on every 4th frame of the take",
            "caveats": "both ran at background priority (nice 19, taskpolicy -b) on the Mac "
            "mini while other agents' jobs were running; compute only, no camera, network or "
            "motor time (frame-to-cue on the same replay was p50 55 ms, p95 97 ms, n = 1517)",
        },
    ]
    res["not_measured"] = {
        "false_alarms_per_minute": "only 4 of 89 labelled frames had a clear path past 1.3 m: "
        f"too few to rate. Navi's front motors were on in {rc['warned_on_clear']['hit']} of "
        f"them and YOLO warned on {ry['warned_on_clear']['hit']}. Over all "
        f"{len(navi)} replayed frames Navi's front motors were on "
        f"{round(100 * cue_on_all / len(navi))}% of the time, so in this room its recall "
        "comes with near-constant buzzing; a fair false-alarm number needs a walk with open "
        "floor and a depth sensor for truth",
        "time_to_first_warning": "the take has no contact moments (he stops short of every "
        "object) and no measured distance to time an approach against",
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {k: res[k] for k in ("recall", "latency_ms", "navi_cue_on_share_all_replayed_frames")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
