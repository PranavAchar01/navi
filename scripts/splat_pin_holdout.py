"""Reorder a splat dataset so Brush holds out the same frames as a reference run.

Brush's --eval-split-every N holds out frames 0, N, 2N, ... in transforms.json order. A dataset
with more frames shifts that split, so its scores are not comparable. This copies SRC to OUT with
the reference's held-out source frames moved to those slots and every image renamed by its new
index, so splat_score.py reads the right ground truth.

    python scripts/splat_pin_holdout.py data/rooms/bedroom/splat-x4-ba \
        data/rooms/bedroom/splat-x4 \
        --src-sources data/rooms/bedroom/splat-x2/sources.txt \
        --ref-sources data/rooms/bedroom/splat/sources.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--src-sources", type=Path, required=True)
    ap.add_argument("--ref-sources", type=Path, required=True)
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args()

    src_names = a.src_sources.read_text().split()
    ref_names = a.ref_sources.read_text().split()
    held = [ref_names[i] for i in range(0, len(ref_names), a.every)]
    t = json.loads((a.src / "transforms.json").read_text())
    frames = t["frames"]

    pairs = list(zip(src_names, frames, strict=True))
    by_name = dict(pairs)
    held_set = set(held)
    rest = iter([p for p in pairs if p[0] not in held_set])
    held_iter = iter(held)
    order: list[tuple[str, dict]] = []
    for i in range(len(frames)):
        if i % a.every == 0 and i // a.every < len(held):
            n = next(held_iter)
            order.append((n, by_name[n]))
        else:
            order.append(next(rest))

    (a.out / "images").mkdir(parents=True, exist_ok=True)
    has_masks = (a.src / "masks").is_dir()
    if has_masks:
        (a.out / "masks").mkdir(exist_ok=True)
    new_frames = []
    for i, (_, f) in enumerate(order):
        stem = f"{i:04d}"
        shutil.copy2(a.src / f["file_path"], a.out / "images" / f"{stem}.jpg")
        g = dict(f, file_path=f"images/{stem}.jpg")
        if has_masks and "mask_path" in f:
            shutil.copy2(a.src / f["mask_path"], a.out / "masks" / f"{stem}.png")
            g["mask_path"] = f"masks/{stem}.png"
        new_frames.append(g)
    (a.out / "transforms.json").write_text(json.dumps(dict(t, frames=new_frames), indent=1))
    (a.out / "sources.txt").write_text("\n".join(n for n, _ in order) + "\n")
    if (a.src / "points.ply").exists():
        shutil.copy2(a.src / "points.ply", a.out / "points.ply")
    print(f"{len(new_frames)} frames, {len(held)} pinned held-out frames")


if __name__ == "__main__":
    main()
