"""Mask what moves out of a splat dataset: hands, people, the laptop and phones change between
frames, so the splat can't learn them and smears them over the room instead. YOLOE finds them by
name; Brush ignores the black pixels of each frame's mask (mask_path in transforms.json).

    python scripts/splat_masks.py data/rooms/bedroom/splat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parent.parent
MOVERS = ["person", "hand", "arm", "foot", "leg", "laptop", "keyboard", "cell phone"]
GROW = 15  # px of margin, so a blurred edge of a hand is masked too


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args()
    ds = Path(a.dataset)
    (ds / "masks").mkdir(exist_ok=True)
    model = YOLOE(str(ROOT / "yoloe-26l-seg.pt"))
    model.set_classes(MOVERS, model.get_text_pe(MOVERS))
    tf = json.loads((ds / "transforms.json").read_text())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1, 2 * GROW + 1))
    covered = []
    for f in tf["frames"]:
        img = cv2.imread(str(ds / f["file_path"]))
        moving = np.zeros(img.shape[:2], np.uint8)
        r = model.predict(img, conf=a.conf, verbose=False, retina_masks=True)[0]
        if r.masks is not None:
            moving = (r.masks.data.cpu().numpy().max(axis=0) > 0.5).astype(np.uint8)
        moving = cv2.dilate(moving, kernel)
        name = Path(f["file_path"]).with_suffix(".png").name
        cv2.imwrite(str(ds / "masks" / name), (1 - moving) * 255)
        f["mask_path"] = f"masks/{name}"
        covered.append(moving.mean())
    (ds / "transforms.json").write_text(json.dumps(tf))
    print(f"{len(covered)} masks, {100 * np.mean(covered):.0f}% of pixels masked on average")


if __name__ == "__main__":
    main()
