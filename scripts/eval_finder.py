"""Federated LVIS scoring of an open-vocabulary finder on the watchlist classes.

The set comes from scripts/fetch_lvis.py: LVIS v1 val images that have a watch-list item boxed,
plus images LVIS checked to be free of it. An image whose status for an item is unknown is not
scored for that item, and a hit on an image LVIS marks as not exhaustively boxed is not counted
against the finder.

usage: eval_finder.py TAG                    (the finder as shipped)
       eval_finder.py TAG --raw [--model M] [--imgsz N] [--prompts JSON]
Writes data/lvis/dets-TAG.json (every detection >= 0.05) and prints AP50 and, per threshold,
recall / precision / share of item-free images with a false hit.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.home() / "helloworld/meta-sight"
LV = ROOT / "data/lvis"


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def score(st, dets, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """dets: {image_id: [(label, score, box xyxy)]}. Returns per-class metrics."""
    res = {}
    for lab, cid in st["classes"].items():
        n_gt, scored, neg_imgs, pos_imgs = 0, [], 0, 0
        neg_hit = {t: 0 for t in thresholds}
        for im in st["images"]:
            gts = [b for c, b in im["boxes"] if c == cid]
            if not gts and cid not in im["neg"]:
                continue  # class status unknown for this image: not scored (federated)
            gts = [(x, y, x + w, y + h) for x, y, w, h in gts]
            n_gt += len(gts)
            mine = sorted(
                [d for d in dets.get(str(im["id"]), []) if d[0] == lab], key=lambda d: -d[1]
            )
            if not gts:
                neg_imgs += 1
                for t in thresholds:
                    neg_hit[t] += any(d[1] >= t for d in mine)
            else:
                pos_imgs += 1
            used = set()
            for _, s, b in mine:
                best, bi = 0.0, -1
                for k, g in enumerate(gts):
                    if k not in used and (v := iou(b, g)) > best:
                        best, bi = v, k
                if best >= 0.5:
                    used.add(bi)
                    scored.append((s, 1))
                elif cid not in im["nex"]:
                    scored.append((s, 0))
        scored.sort(key=lambda x: -x[0])
        tp = np.cumsum([x[1] for x in scored]) if scored else np.array([0])
        fp = np.cumsum([1 - x[1] for x in scored]) if scored else np.array([0])
        rec = tp / max(n_gt, 1)
        prec = tp / np.maximum(tp + fp, 1)
        # all-point interpolated AP
        mrec = np.concatenate([[0], rec, [1]])
        mpre = np.concatenate([[0], prec, [0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
        at = {}
        for t in thresholds:
            k = sum(1 for s, _ in scored if s >= t)
            at[t] = (
                round(float(rec[k - 1]) if k else 0.0, 3),
                round(float(prec[k - 1]) if k else 1.0, 3),
                round(neg_hit[t] / max(neg_imgs, 1), 3),
            )
        res[lab] = dict(ap50=round(ap, 3), n_gt=n_gt, pos_imgs=pos_imgs, neg_imgs=neg_imgs, at=at)
    return res


def show(tag, res):
    print(
        f"== {tag}  (at threshold: recall / precision / share of item-free images with a false hit)"
    )
    for lab, r in res.items():
        at = "  ".join(f"{t}:{a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f}" for t, a in r["at"].items())
        print(f"{lab:15s} AP50 {r['ap50']:.3f}  gt {r['n_gt']:3d}  {at}")
    print(f"{'mean':15s} AP50 {np.mean([r['ap50'] for r in res.values()]):.3f}")


def shipped(labels, tiles=2):
    """The finder as the app runs it (detector plus crop check), with no final cut."""
    from metasight.config import Settings
    from metasight.providers.finder import YoloeFinder

    f = YoloeFinder(Settings.load(), conf=0.05, keep=0.0, tiles=tiles)
    return lambda bgr: [(d.label, d.score, list(d.box)) for d in f.find(bgr, labels)]


def finder(a, labels):
    """The bare detector, for trying models, sizes and prompts."""
    from ultralytics import YOLO

    from metasight.providers.finder import PROMPTS

    prompts = json.loads(a.prompts) if a.prompts else PROMPTS
    m = YOLO(str(ROOT / a.model) if (ROOT / a.model).exists() else a.model)
    m.set_classes([prompts.get(lb, lb) for lb in labels])

    def run(bgr):
        r = m.predict(bgr, imgsz=a.imgsz, conf=0.05, device="mps", verbose=False)[0]
        return [
            (labels[int(c)], float(p), [float(v) for v in b])
            for b, c, p in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                strict=True,
            )
        ]

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--model", default="yoloe-26l-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--prompts", default=None, help="JSON; default: the finder's PROMPTS")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--tiles", type=int, default=2, help="zoom grid, n by n")
    ap.add_argument("--raw", action="store_true", help="the bare detector, no crop check")
    a = ap.parse_args()
    st = json.load(open(LV / "set.json"))
    out = LV / f"dets-{a.tag}.json"
    if not a.score_only:
        import cv2

        labels = list(st["classes"])
        find = finder(a, labels) if a.raw else shipped(labels, a.tiles)
        dets, t0 = {}, time.time()
        for im in st["images"]:
            bgr = cv2.imread(str(LV / "img" / f"{im['id']}.jpg"))
            dets[str(im["id"])] = find(bgr)
        print(
            f"{a.tag}: {1000 * (time.time() - t0) / len(st['images']):.0f} ms/image",
            file=sys.stderr,
        )
        json.dump(dets, open(out, "w"))
    show(a.tag, score(st, json.load(open(out))))


if __name__ == "__main__":
    main()
