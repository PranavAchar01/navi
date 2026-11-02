"""Before/after clips of the field-of-view memory for the demo cut, from the fovsim scenes.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Each clip plays one scene (seed 0): the glasses' view in
the middle, the belt as the live cue drives it on the left, the belt with the hazard memory on the
right. A sector lights by how near its cue is; a red ring marks a hazard under 1 m that the cue
missed. The caption carries the scene's miss share over 20 seeds from bench/results/fov-sim.json.
Writes data/demo-takes/fov/<scene>.mp4 (1280x720, 30 fps).

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/render_fov_clips.py [scenes]
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from metasight import fovsim as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "demo-takes" / "fov"
ARM = F.Arms(low_slope=0.02, mask=True, people=True)
ARM_KEYS = ("memory, low slope 0.02", "memory")  # REC, by its old and its current name
TITLES = {
    "desk_lookup": "Looks up from the desk: the edge drops out of view",
    "shelf_head": "Shelf at head height: above the frame on the approach",
    "low_box": "Shoebox on the floor: under the frame for the last metre",
    "chair_legs": "Chair on thin legs: the seat drops under the frame",
    "doorway_turn": "Head turned in a doorway: the jambs are out of view",
}
W, H = 1280, 720
BG = (18, 18, 20)
INK = (235, 235, 230)
DIM = (130, 130, 128)
HOT = (255, 120, 40)
MISS = (230, 40, 40)


def _font(size: int, mono: bool = False):
    path = "/System/Library/Fonts/SFNSMono.ttf" if mono else "/System/Library/Fonts/SFNS.ttf"
    return ImageFont.truetype(path, size)


def belt(draw: ImageDraw.ImageDraw, cx: int, cy: int, sec, path, truth, label: str) -> None:
    """Eight sectors around a top-down body, clockwise from ahead (up)."""
    r0, r1 = 46, 150
    tsec, _ = truth
    for i in range(8):
        a0 = -90 + i * 45 - 22.5 + 2
        a1 = a0 + 45 - 4
        d = sec[i]
        on = max(0.0, min(1.0, (2.0 - d) / 1.6)) if math.isfinite(d) else 0.0
        col = tuple(int(40 + (c - 40) * on) for c in HOT) if on else (40, 40, 44)
        box = [cx - r1, cy - r1, cx + r1, cy + r1]
        draw.pieslice(box, a0, a1, fill=col)
        if tsec[i] < F.DANGER_M and not d <= tsec[i] + F.MISS_TOL_M:
            draw.arc([cx - r1 - 10, cy - r1 - 10, cx + r1 + 10, cy + r1 + 10], a0, a1, MISS, 7)
    draw.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], fill=BG)
    draw.polygon([(cx, cy - 28), (cx - 14, cy - 6), (cx + 14, cy - 6)], fill=DIM)
    draw.text((cx, cy - r1 - 48), label, font=_font(26), fill=INK, anchor="mm")
    txt = f"path {path:.2f} m" if math.isfinite(path) else "path clear"
    draw.text((cx, cy + r1 + 34), txt, font=_font(22, True), fill=INK, anchor="mm")


def clip(name: str, stats: dict) -> Path:
    sc = F.SCENES[name]
    rec: list[dict] = []
    F.run(sc, [ARM], seed=0, record=rec)
    live = stats["arms"]["live"]["miss"]
    mem = next(stats["arms"][k] for k in ARM_KEYS if k in stats["arms"])["miss"]
    caption = f"missed hazards under 1 m: live {live:.2f}  ->  with memory {mem:.2f}   (20 seeds)"
    tmp = OUT / f".{name}"
    tmp.mkdir(parents=True, exist_ok=True)
    for k, fr in enumerate(rec):
        img = Image.new("RGB", (W, H), BG)
        view = Image.fromarray(fr["rgb"]).resize((360, 640))
        if fr["blind"]:
            view = view.point(lambda v: v // 4)
        img.paste(view, ((W - 360) // 2, 40))
        d = ImageDraw.Draw(img)
        (lsec, lpath), (msec, mpath) = fr["cues"]
        belt(d, 220, 380, lsec, lpath, fr["truth"], "Live cue")
        belt(d, W - 220, 380, msec, mpath, fr["truth"], "With memory")
        d.text((W // 2, 22), TITLES.get(name, name), font=_font(24), fill=INK, anchor="mm")
        d.text((W // 2, 700), caption, font=_font(20, True), fill=INK, anchor="mm")
        d.text((16, 700), "SIMULATED", font=_font(16, True), fill=DIM, anchor="lm")
        d.text((W - 16, 700), "red ring = missed", font=_font(16, True), fill=MISS, anchor="rm")
        img.save(tmp / f"{k:04d}.png")
    out = OUT / f"{name}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(F.FPS),
            "-i", str(tmp / "%04d.png"), "-vf", "fps=30,format=yuv420p",
            "-c:v", "libx264", "-crf", "18", str(out),
        ],
        check=True,
    )  # fmt: skip
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    return out


def main() -> None:
    stats = json.loads((ROOT / "bench" / "results" / "fov-sim.json").read_text())["scenes"]
    names = sys.argv[1:] or ["desk_lookup", "chair_legs", "shelf_head", "low_box"]
    for n in names:
        print(clip(n, stats[n]))


if __name__ == "__main__":
    main()
