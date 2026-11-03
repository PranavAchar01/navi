"""The parallel-walks reel, laid out in 2D from per-room tile clips.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Sixty find-the-chair walks (seeds 1000-1029 on the
eight-motor belt, top three rows, and the four built diagonals, bottom three), each rendered as
its own small clip by scripts/blender_sim_tile.py and laid out here. The hero room starts
full screen from its own close-up render and pulls back into its cell while the other rooms
fade in; each iteration plays on every room at once; a ring flashes red on each contact and
holds green once the walker sat without touching anything. The timeline, camera keys and the
tallies come from data/demo-takes/sim/grid/timeline.json (scripts/build_sim_grid.py plan).

    PYTHONPATH=. .venv/bin/python scripts/build_sim_reel.py \
        --grid data/demo-takes/sim/grid --tiles data/demo-takes/sim/tiles \
        --hero data/demo-takes/sim/hero --out data/demo-takes/sim/montage-v2.mp4 [--only 600]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
CW, CH = 192, 170  # a room's cell; 10 x 6 of them leave 60 px under the grid for the caption
CROP = (92, 7, 92 + CW, 7 + CH)  # the room inside a 384 x 216 tile frame
TILE_W, TILE_H = 384, 216
FONT = "/System/Library/Fonts/SFNS.ttf"
GREEN = (64, 255, 115)
RED = (255, 41, 31)
FLASH = 24  # frames a red flash takes to fade


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


class Reel:
    def __init__(self, grid: Path, tiles: Path, hero: Path):
        self.tl = json.loads((grid / "timeline.json").read_text())
        self.tiles_dir, self.hero_dir = tiles, hero
        self.tiles = self.tl["tiles"]
        self.cell = {t["id"]: (t["col"] * CW, t["row"] * CH) for t in self.tiles}
        self.hero = self.tl["hero"]
        self.idx: dict[tuple[str, str], list[int]] = {}
        self.bg = None
        self.f_small = ImageFont.truetype(FONT, 26)
        self.f_big = ImageFont.truetype(FONT, 34)
        self.f_tag = ImageFont.truetype(FONT, 22)

    def files(self, base: Path, tile: str, arm: str) -> list[int]:
        key = (str(base), f"{tile}/{arm}")
        if key not in self.idx:
            d = base / tile / arm
            self.idx[key] = sorted(int(p.stem) for p in d.glob("*.jpg")) if d.is_dir() else []
        return self.idx[key]

    def phase_of(self, g: int) -> tuple[int, dict]:
        for i, ph in enumerate(self.tl["phases"]):
            if ph["start"] <= g < ph["start"] + ph["len"]:
                return i, ph
        return len(self.tl["phases"]) - 1, self.tl["phases"][-1]

    def tile_frame(self, base: Path, tile: str, ph: dict, g: int, ext="jpg"):
        walk = g - ph["start"] - ph.get("delay", {}).get(tile, 0)
        have = self.files(base, tile, ph["arm"])
        if not have:
            return None
        step = have[1] - have[0] if len(have) > 1 else 1  # tiles are rendered every few frames
        want = max(0, walk) // step * step + 1
        k = max((n for n in have if n <= want), default=have[0])
        img = Image.open(base / tile / ph["arm"] / f"{k:05d}.{ext}").convert("RGB")
        if base == self.tiles_dir and img.size != (TILE_W, TILE_H):
            img = img.resize((TILE_W, TILE_H), Image.BILINEAR)
        return img

    def camera(self, g: int):
        """(zoom 0 close .. 1 wide, fade-in 0..1 of the other rooms)."""
        keys = self.tl["camera"]
        for (f0, m0), (f1, m1) in zip(keys, keys[1:], strict=False):
            if f0 <= g <= f1:
                a, b = (m0 == "wide") * 1.0, (m1 == "wide") * 1.0
                z = lerp(a, b, smooth((g - f0) / max(1, f1 - f0)))
                break
        else:
            z = 1.0 if keys[-1][1] == "wide" else 0.0
        first = [k for k in keys if k[1] == "wide"][0][0]
        fade = smooth((g - keys[1][0]) / max(1, first - keys[1][0] - 40)) if g < first else 1.0
        return z, fade

    def frame(self, g: int) -> Image.Image:
        z, fade = self.camera(g)
        pi, ph = self.phase_of(g)
        hx, hy = self.cell[self.hero]
        fx, fy = hx - CROP[0], hy - CROP[1]  # the hero's tile frame on the canvas
        hires = self.tile_frame(self.hero_dir, self.hero, ph, g, "jpg")
        use_hires = hires is not None and (z < 0.999 or g <= 255 or g >= 1050)
        # canvas: every room's cropped tile frame
        canvas = Image.new("RGB", (W, H), self.bg_color())
        for t in self.tiles:
            fr = self.tile_frame(self.tiles_dir, t["id"], ph, g)
            if fr is None:
                continue
            canvas.paste(fr.crop(CROP), self.cell[t["id"]])
        # camera rectangle on the canvas: the hero's tile frame (close) to the whole canvas (wide)
        wz = math.exp(lerp(math.log(TILE_W), math.log(W), z))
        hh = wz * 9 / 16
        cx = lerp(fx + TILE_W / 2, W / 2, z)
        cy = lerp(fy + TILE_H / 2, H / 2, z)
        rx = min(max(cx - wz / 2, min(0.0, fx)), max(W - wz, 0.0) if wz < W else 0.0)
        ry = min(max(cy - hh / 2, min(0.0, fy)), max(H - hh, 0.0) if hh < H else 0.0)
        if z < 1e-6:
            rx, ry = fx, fy
        s = W / wz
        base = canvas.resize((W, H), Image.BILINEAR, box=(rx, ry, rx + wz, ry + hh))
        arr = np.asarray(base, dtype=np.float32)
        bgc = np.array(self.bg_color(), dtype=np.float32)
        if fade < 1.0:
            arr = bgc + (arr - bgc) * fade
        out = Image.fromarray(arr.astype(np.uint8))
        # the hero's own render: its tile frame scaled with the camera, solid inside its cell
        if use_hires:
            hw, hhh = round(TILE_W * s), round(TILE_H * s)
            himg = hires.resize((hw, hhh), Image.LANCZOS)
            ox, oy = round((fx - rx) * s), round((fy - ry) * s)
            mask = Image.new("L", (hw, hhh), round(255 * (1 - fade)))
            ImageDraw.Draw(mask).rectangle(
                (
                    round(CROP[0] * s),
                    round(CROP[1] * s),
                    round(CROP[2] * s) - 1,
                    round(CROP[3] * s) - 1,
                ),
                fill=255,
            )
            out.paste(himg, (ox, oy), mask)
        draw = ImageDraw.Draw(out, "RGBA")
        self.rings(draw, g, pi, ph, rx, ry, s, fade)
        self.hud(draw, g, pi, ph)
        return out

    def bg_color(self):
        if self.bg is None:
            t = self.tiles[0]["id"]
            fr = Image.open(self.tiles_dir / t / "I0" / "00001.jpg").convert("RGB")
            self.bg = fr.getpixel((10, TILE_H - 10))
        return self.bg

    def outcome(self, tile: str, pi: int, g: int):
        info = self.tl["tally"][pi][tile]
        red = 0.0
        for c in info["contacts"]:
            if g >= c:
                red = max(red, 1 - (g - c) / FLASH)
        done = info["clean"] and g >= info["end"]
        return red, done

    def rings(self, draw, g, pi, ph, rx, ry, s, fade):
        for t in self.tiles:
            red, done = self.outcome(t["id"], pi, g)
            if red <= 0 and not done:
                continue
            x, y = self.cell[t["id"]]
            box = ((x - rx) * s, (y - ry) * s, (x + CW - rx) * s, (y + CH - ry) * s)
            if box[2] < 0 or box[3] < 0 or box[0] > W or box[1] > H:
                continue
            w = max(3, round(2.2 * s))
            col, a = (GREEN, 235) if done else (RED, round(255 * min(1.0, red * 1.3)))
            if done and red > 0:
                col, a = RED, round(255 * red)
            draw.rectangle(box, outline=(*col, a), width=w)
            if done:
                draw.rectangle(box, fill=(*GREEN, 26))

    def hud(self, draw, g, pi, ph):
        clean = contacts = 0
        for t in self.tiles:
            info = self.tl["tally"][pi][t["id"]]
            contacts += sum(1 for c in info["contacts"] if c <= g)
            clean += 1 if info["clean"] and g >= info["end"] else 0
        n = len(self.tiles)
        draw.rectangle((0, H - 60, W, H), fill=(14, 15, 18, 255))
        arm = ph["arm"]
        draw.text((28, H - 47), f"{arm}", font=self.f_big, fill=(255, 255, 255, 255))
        draw.text((100, H - 42), ph["caption"], font=self.f_small, fill=(214, 214, 210, 255))
        right = f"seated without touching anything {clean}/{n}    contacts {contacts}"
        wid = draw.textlength(right, font=self.f_small)
        draw.text((W - 28 - wid, H - 42), right, font=self.f_small, fill=(236, 236, 230, 255))
        tag = "SIMULATION  -  simulated user model, not a human study"
        draw.rounded_rectangle(
            (20, 18, 20 + draw.textlength(tag, font=self.f_tag) + 28, 58),
            radius=8,
            fill=(14, 15, 18, 200),
        )
        draw.text((34, 24), tag, font=self.f_tag, fill=(255, 255, 255, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--hero", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", type=int)
    ap.add_argument("--seconds", type=float, default=20.0, help="reel length (sped up)")
    a = ap.parse_args()
    reel = Reel(a.grid, a.tiles, a.hero)
    if a.only:
        reel.frame(a.only).save(a.out)
        return
    n = reel.tl["frames"]
    enc = subprocess.Popen(
        [
            "nice",
            "-n",
            "19",
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            "30",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(a.out),
        ],
        stdin=subprocess.PIPE,
    )
    out_n = round(a.seconds * 30)
    for o in range(out_n):
        g = 1 + round(o * (n - 1) / (out_n - 1))
        enc.stdin.write(reel.frame(g).tobytes())
    enc.stdin.close()
    enc.wait()


if __name__ == "__main__":
    main()
