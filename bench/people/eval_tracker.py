"""Tracker identity switches on synthetic walkers (exact ground truth), and the speed and track
statistics of the real person detector on local test footage (no ground truth there)."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metasight.people import Det, PeopleTracker, iou  # noqa: E402


def synthetic(seed, n_people=5, fps=15, secs=30, miss=0.1, jitter=0.004):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.1, 0.9, n_people)
    vel = rng.uniform(-0.15, 0.15, n_people)
    dist = rng.uniform(1.5, 6.0, n_people)
    w = 0.5 / dist
    tk = PeopleTracker()
    votes = {}
    for k in range(fps * secs):
        t = k / fps
        pos += vel / fps
        bounce = (pos < 0.05) | (pos > 0.95)
        vel[bounce] *= -1
        dets, gt = [], []
        for i in range(n_people):
            b = (pos[i] - w[i] / 2, 0.15, pos[i] + w[i] / 2, 0.85)
            gt.append(b)
            if rng.random() > miss:
                j = rng.normal(0, jitter, 4)
                d = dist[i] * rng.uniform(0.9, 1.1)
                dets.append(Det(tuple(np.array(b) + j), float(rng.uniform(0.4, 0.95)), d))
        for tr in tk.update(dets, t):
            if tr.lost:
                continue
            cx = 0.5 * (tr.box[0] + tr.box[2])
            i = int(
                np.argmin(
                    [
                        abs(cx - 0.5 * (g[0] + g[2])) + 0.02 * abs(tr.dist_m - dist[j])
                        for j, g in enumerate(gt)
                    ]
                )
            )
            votes.setdefault(tr.id, []).append(i)
    # a track that follows more than one true person swapped identity: count its minority frames
    switches = sum(len(v) - max(v.count(x) for x in set(v)) for v in votes.values())
    return switches, tk.next_id - 1 - n_people


def real(path, seconds=60, fps=10):
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(Path(__file__).resolve().parents[2] / "weights" / "yolo26n.pt"))
    cap = cv2.VideoCapture(path)
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / fps))
    tk = PeopleTracker()
    life, k, t_det, t_trk, frames = {}, 0, 0.0, 0.0, 0
    while frames < seconds * fps:
        ok, f = cap.read()
        if not ok:
            break
        k += 1
        if k % step:
            continue
        frames += 1
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=320, device="mps", verbose=False, conf=0.1, classes=[0])[0]
        t1 = time.perf_counter()
        h, w = f.shape[:2]
        boxes, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        dets = [
            Det((a / w, b / h, c / w, d / h), float(s))
            for (a, b, c, d), s in zip(boxes, conf, strict=True)
        ]
        out = tk.update(dets, frames / fps)
        t_det += t1 - t0
        t_trk += time.perf_counter() - t1
        for tr in out:
            life[tr.id] = life.get(tr.id, 0) + 1
    L = np.array(list(life.values()) or [0]) / fps
    print(
        f"{Path(path).name}: {frames} frames, {len(life)} tracks, median life {np.median(L):.1f}s, "
        f"p90 {np.percentile(L, 90):.1f}s, detector {1000 * t_det / frames:.1f} ms, "
        f"tracker {1000 * t_trk / frames:.2f} ms"
    )


if __name__ == "__main__":
    res = [synthetic(s) for s in range(20)]
    sw = np.array([r[0] for r in res])
    fr = np.array([r[1] for r in res])
    print(
        f"SYNTH 20 runs x 5 walkers x 30 s, 10% missed boxes: wrong-person frames mean {sw.mean():.2f} "
        f"(max {sw.max()}), extra ids mean {fr.mean():.2f} (max {fr.max()})"
    )
    for p in sys.argv[1:]:
        real(p)
