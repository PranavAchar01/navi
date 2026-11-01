"""Does the room's object map help the detector? Scores the live detector with and without the
room prior on a walk the objects were rebuilt without (room_objects.py --exclude).

Reference boxes come from YOLOE-26L at 640 with text classes: an automatic labeller that is
stronger than the live model, not hand labels. Frames the room matcher cannot place keep the
plain detector in the prior arm, so both arms score the same frames.

    python scripts/eval_room_prior.py --walk 20261031-004732-glasses \
        --objects data/rooms/bedroom/seed/objects-s1-without-20261031-004732-glasses.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metasight import home as homemod  # noqa: E402
from metasight.relocalize import RoomMatcher  # noqa: E402
from metasight.roomprior import CONFIRM_CONF, LIVE_CONF, Box, RoomPrior  # noqa: E402

CLASSES = ["bed", "laptop", "tv", "keyboard", "mouse", "cell phone", "chair", "bottle", "book"]
CLASSES += ["backpack", "cup", "dining table"]
MIN_INLIERS = 220  # server.ROOM_VIEW_MIN_INLIERS
IOU = 0.3


def iou(a: tuple, b: tuple) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def boxes(res, names: dict) -> list[Box]:
    h, w = res.orig_shape
    return [
        Box(names[int(c)], float(p), (x0 / w, y0 / h, x1 / w, y1 / h))
        for (x0, y0, x1, y1), c, p in zip(
            res.boxes.xyxy.cpu().numpy().tolist(),
            res.boxes.cls.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            strict=True,
        )
        if names[int(c)] in CLASSES
    ]


def score(pred: list[Box], ref: list[Box], tally: dict) -> None:
    """Greedy same-label matching by IoU; boxes below the live threshold (the ones the room
    confirmed) are also counted on their own."""
    free = list(range(len(ref)))
    for b in sorted(pred, key=lambda b: -b.score):
        best, j = 0.0, None
        for i in free:
            if ref[i].label == b.label and (s := iou(b.box, ref[i].box)) > best:
                best, j = s, i
        hit = j is not None and best >= IOU
        if hit:
            free.remove(j)
        tally["tp" if hit else "fp"] += 1
        if b.score < LIVE_CONF:
            key = "confirmed_tp" if hit else "confirmed_fp"
            tally[key] = tally.get(key, 0) + 1
    tally["fn"] += len(free)


def pr(t: dict) -> str:
    p = t["tp"] / max(1, t["tp"] + t["fp"])
    r = t["tp"] / max(1, t["tp"] + t["fn"])
    extra = {k: v for k, v in t.items() if k not in ("tp", "fp", "fn")}
    return f"P {p:.2f}  R {r:.2f}  (tp {t['tp']} fp {t['fp']} fn {t['fn']}) {extra}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--home", default=str(ROOT / "data/rooms/bedroom/home"))
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default=None, help="write per-frame results here (json)")
    a = ap.parse_args()

    from ultralytics import YOLO

    h = homemod.load(Path(a.home))
    m = RoomMatcher([cv2.imdecode(np.frombuffer(v.jpg, np.uint8), 1) for v in h.views], h.views)
    hh, ww = m.shape
    kn = m.k / np.array([[ww], [hh], [1]])  # pinhole in normalised image units
    prior = RoomPrior(json.loads(Path(a.objects).read_text()), Path(a.home) / "room.ply")
    live = YOLO(str(ROOT / "yolo26n.pt"))
    refm = YOLO(str(ROOT / "yoloe-26l-seg.pt"))
    refm.set_classes(CLASSES)

    wdir = ROOT / "data/walks" / a.walk
    rows = [json.loads(ln) for ln in (wdir / "frames.jsonl").read_text().splitlines() if ln]
    pick = rows[:: max(1, len(rows) // a.frames)]
    arms = {
        k: {"tp": 0, "fp": 0, "fn": 0} for k in ("base", "prior", "base_placed", "prior_placed")
    }
    per = []
    placed = 0
    for r in pick:
        img = cv2.imread(str(wdir / "frames" / r["file"]))
        if img is None:
            continue
        ref = boxes(
            refm.predict(img, imgsz=640, conf=0.4, verbose=False, device="mps")[0], refm.names
        )
        det = boxes(
            live.predict(img, imgsz=320, conf=CONFIRM_CONF, verbose=False, device="mps")[0],
            live.names,
        )
        base = [b for b in det if b.score >= LIVE_CONF]
        hit = m.locate(cv2.resize(img, (ww, hh), interpolation=cv2.INTER_AREA))
        ok = hit is not None and hit[1] >= MIN_INLIERS
        withp = prior.apply(det, hit[0], kn) if ok else base
        placed += ok
        score(base, ref, arms["base"])
        score(withp, ref, arms["prior"])
        if ok:
            score(base, ref, arms["base_placed"])
            score(withp, ref, arms["prior_placed"])
        per.append(
            {
                "file": r["file"],
                "placed": bool(ok),
                "ref": [(b.label, [round(x, 3) for x in b.box]) for b in ref],
                "base": [(b.label, round(b.score, 2)) for b in base],
                "prior": [
                    (b.label, round(b.score, 2), [round(x, 3) for x in b.box]) for b in withp
                ],
            }
        )
    print(f"{a.walk}: {len(per)} frames, {placed} placed by the room matcher")
    for k, t in arms.items():
        print(f"{k:13s} {pr(t)}")
    if a.out:
        Path(a.out).write_text(json.dumps(per, indent=1))


if __name__ == "__main__":
    main()
