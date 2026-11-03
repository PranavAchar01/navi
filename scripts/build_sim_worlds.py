"""The six-worlds reel: the walker in every room it was tuned in, all at once, before and after.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Twelve rooms play together: the six worlds of the walk
ledger (the reconstructed bedroom twice, chair tucked at the desk and chair pulled into the
aisle, plus the four field-of-view catalogue rooms) on both belt layouts, seed 1000. Each room is
its own Blender render (scripts/blender_sim_tile.py, scenes from scripts/run_sim_worlds.py) and
ends the way its recorded episode did: a red flash at each recorded contact, green once the
walker reaches the goal without touching anything. Round one is I0, the glasses as configured;
the five policy updates between are scored on the full ledger; round two is I6. The chart is
the whole ledger, 360 walks per iteration (6 worlds x 2 layouts x seeds 1000-1029).

    PYTHONPATH=. .venv/bin/python scripts/build_sim_worlds.py \
        --out data/demo-takes/sim/montage-v4.mp4 [--still 100 200 ...]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "demo-takes" / "sim" / "worlds"
LEDGER = ROOT / "bench" / "results" / "sim-walk.json"
W, H = 1920, 1080
FPS = 30
SEED = 1000
ORDER = ("avoid_chair", "find_chair", "desk_lookup", "shelf_head", "low_box", "chair_legs")
WORLD = {
    "avoid_chair": "Bedroom: chair in the aisle",
    "find_chair": "Bedroom: find the chair, sit",
    "desk_lookup": "Desk edge",
    "shelf_head": "Shelf at head height",
    "low_box": "Low box on the floor",
    "chair_legs": "Chair on thin legs",
}
LAYOUT = {"eight": "8 motors", "diag4": "4 diagonals"}
ARMS = ("I0", "I1", "I2", "I3", "I4", "I5", "I6")
SHOWN = ("I0", "I6")
ROLE = {
    "I0": "the glasses as configured: only what the camera sees right now",
    "I6": "after five updates: map memory, held bearing, navigator, two cues",
}
# layout
GX, GY = 36, 132
TW, TH = 360, 250  # tile
PX, PY = 372, 290  # cell pitch (tile + caption + gap)
CROP = (60, 0, 516, 317)  # the room inside a 576 x 400 render
PANEL = 1556
FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
BG = (12, 14, 18)
CARD = (22, 25, 31)
INK = (238, 240, 244)
DIM = (140, 147, 160)
GREEN = (62, 226, 124)
RED = (255, 72, 60)
BAR = (92, 100, 116)
LABEL = "SIMULATED USER MODEL, NOT A HUMAN STUDY"
# timing, output frames
SPEED = 3.0  # sim frames per output frame
INTRO, HOLD, UPDATE, OUTRO = 30, 18, 54, 96
FLASH = 12


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


@dataclass
class Walk:
    world: str
    layout: str
    frames_dir: Path
    files: list[int]
    contacts: list[int]  # sim frames where the contact count rises
    end: int  # last walking frame
    clean: bool
    length: int


def load_walk(arm: str, world: str, layout: str, rows: dict) -> Walk:
    tid = f"{world}-{layout}-{SEED}"
    s = json.loads((BASE / "scenes" / f"{arm}-{tid}.json").read_text())
    fr = s["frames"]
    col = [f["collisions"] for f in fr]
    contacts = [k for k in range(len(col)) if col[k] > (col[k - 1] if k else 0)]
    end = max(k for k, f in enumerate(fr) if f.get("walking", True))
    d = BASE / "tiles" / tid / arm
    files = sorted(int(p.stem) for p in d.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no tile frames in {d}")
    r = rows[(arm, world, layout, SEED)]
    clean = bool(s["metrics"]["success"])
    assert len(contacts) == r["collisions"], (arm, tid)
    assert clean == bool(r["success"] and r["collisions"] == 0), (arm, tid)
    assert all(c <= files[-1] for c in contacts), (arm, tid)  # every contact is on screen
    return Walk(world, layout, d, files, contacts, end, clean, len(fr))


class Reel:
    def __init__(self):
        rows = json.loads(LEDGER.read_text())["rows"]
        self.rows = {(r["arm"], r["scenario"], r["layout"], r["seed"]): r for r in rows}
        self.tally = {}
        for a in ARMS:
            rs = [r for r in rows if r["arm"] == a]
            assert len(rs) == 360, a
            self.tally[a] = (
                sum(r["collisions"] for r in rs),
                sum(bool(r["success"] and r["collisions"] == 0) for r in rs),
            )
        self.walks = {
            a: [load_walk(a, w, lay, self.rows) for w in ORDER for lay in ("eight", "diag4")]
            for a in SHOWN
        }
        self.len = {
            a: min(max(w.length for w in ws), max(w.files[-1] for w in ws))
            for a, ws in self.walks.items()
        }
        n0 = math.ceil(self.len["I0"] / SPEED)
        n6 = math.ceil(self.len["I6"] / SPEED)
        t = 0
        self.seg = {}
        for name, n in (
            ("intro", INTRO),
            ("I0", n0),
            ("hold", HOLD),
            ("update", UPDATE),
            ("I6", n6),
            ("outro", OUTRO),
        ):
            self.seg[name] = (t, n)
            t += n
        self.total = t
        self.cache: dict = {}
        f = lambda p, s: ImageFont.truetype(p, s)  # noqa: E731
        self.f = {
            "tag": f(FONT, 19),
            "title": f(FONT, 38),
            "sub": f(FONT, 21),
            "cap": f(FONT, 17),
            "chip": f(FONT, 15),
            "arm": f(FONT, 96),
            "role": f(FONT, 18),
            "k": f(FONT, 17),
            "num": f(FONT, 46),
            "bar": f(MONO, 15),
            "foot": f(FONT, 15),
            "banner": f(FONT, 30),
        }
        self.bg = self.backdrop()

    # ---------------------------------------------------------------- time

    def where(self, o: int) -> tuple[str, float]:
        for name, (t0, n) in self.seg.items():
            if o < t0 + n:
                return name, (o - t0)
        return "outro", self.seg["outro"][1] - 1

    def state(self, o: int) -> tuple[str, int, float]:
        """(arm on the tiles, sim frame, wipe 0..1 toward I6 during the update)."""
        name, u = self.where(o)
        if name == "intro":
            return "I0", 0, 0.0
        if name == "I0":
            return "I0", min(int(u * SPEED), self.len["I0"] - 1), 0.0
        if name == "hold":
            return "I0", self.len["I0"] - 1, 0.0
        if name == "update":
            return "I0", self.len["I0"] - 1, smooth((u - 30) / 18)
        if name == "I6":
            return "I6", min(int(u * SPEED), self.len["I6"] - 1), 0.0
        return "I6", self.len["I6"] - 1, 0.0

    # ---------------------------------------------------------------- tiles

    def load(self, w: Walk, n: int) -> Image.Image:
        key = (str(w.frames_dir), n)
        im = self.cache.get(key)
        if im is None:
            im = Image.open(w.frames_dir / f"{n:05d}.jpg").convert("RGB")
            im = im.crop(CROP).resize((TW, TH), Image.LANCZOS)
            if len(self.cache) > 6000:
                self.cache.clear()
            self.cache[key] = im
        return im

    def tile_img(self, w: Walk, k: float) -> Image.Image:
        files = w.files
        step = files[1] - files[0]
        pos = (k + 1 - files[0]) / step
        i0 = int(min(max(math.floor(pos), 0), len(files) - 1))
        i1 = min(i0 + 1, len(files) - 1)
        a = min(max(pos - i0, 0.0), 1.0) if i1 != i0 else 0.0
        im = self.load(w, files[i0])
        return im if a < 0.05 else Image.blend(im, self.load(w, files[i1]), a)

    def backdrop(self) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for i in range(12):
            x, y = GX + (i % 4) * PX, GY + (i // 4) * PY
            d.rounded_rectangle((x - 1, y - 1, x + TW + 1, y + TH + 31), 10, fill=CARD)
        d.rounded_rectangle((PANEL - 14, GY - 1, W - 30, GY + 3 * PY - 21), 14, fill=CARD)
        return img

    # ---------------------------------------------------------------- frame

    def frame(self, o: int) -> Image.Image:
        arm, k, wipe = self.state(o)
        img = self.bg.copy()
        d = ImageDraw.Draw(img, "RGBA")
        name, u = self.where(o)
        shown_contacts = shown_clean = walking = 0
        for i, w in enumerate(self.walks[arm]):
            x, y = GX + (i % 4) * PX, GY + (i // 4) * PY
            tile = self.tile_img(w, k)
            if wipe > 0:  # the next policy takes the rooms over, left to right
                nxt = self.tile_img(self.walks["I6"][i], 0)
                cut = int(round(wipe * (W + 400) - 200 - x))
                if cut > 0:
                    tile = tile.copy()
                    tile.paste(nxt.crop((0, 0, min(cut, TW), TH)), (0, 0))
            mask = Image.new("L", (TW, TH), 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, TW - 1, TH - 1), 9, fill=255)
            img.paste(tile, (x, y), mask)
            # outcome
            red = 0.0
            n_hit = 0
            for c in w.contacts:
                if k >= c:
                    n_hit += 1
                    if name == arm:
                        red = max(red, 1 - (o - self.seg[arm][0] - c / SPEED) / FLASH)
            done = w.clean and k >= w.end
            in_update = wipe > 0 and (wipe * (W + 400) - 200 - x) > 0
            if in_update:
                n_hit, done, red = 0, False, 0.0
            shown_contacts += n_hit
            shown_clean += done
            walking += k < w.end
            if red > 0:
                d.rounded_rectangle((x, y, x + TW, y + TH), 9, fill=(*RED, int(90 * red)))
                d.rounded_rectangle(
                    (x - 2, y - 2, x + TW + 2, y + TH + 2),
                    11,
                    outline=(*RED, int(255 * red)),
                    width=4,
                )
            if done:
                d.rounded_rectangle(
                    (x - 2, y - 2, x + TW + 2, y + TH + 2), 11, outline=(*GREEN, 235), width=3
                )
                self.chip(d, x + TW - 12, y + 10, "reached, nothing touched", GREEN)
            elif n_hit:
                self.chip(
                    d, x + TW - 12, y + 10, f"{n_hit} contact" + ("s" if n_hit > 1 else ""), RED
                )
            cap = f"{WORLD[w.world]}"
            d.text((x + 4, y + TH + 6), cap, font=self.f["cap"], fill=INK)
            lw = d.textlength(LAYOUT[w.layout], font=self.f["cap"])
            d.text((x + TW - 4 - lw, y + TH + 6), LAYOUT[w.layout], font=self.f["cap"], fill=DIM)
        if wipe > 0:
            xw = wipe * (W + 400) - 200
            if GX <= xw <= GX + 4 * PX:
                for g in range(10):
                    d.line(
                        (xw - g, GY - 6, xw - g, GY + 3 * PY - 18),
                        fill=(255, 255, 255, 110 - g * 11),
                    )
        self.header(d, name, u)
        self.panel(d, arm, name, u, shown_contacts, shown_clean, walking)
        if name == "update":
            self.banner(img, u)
        return img

    def chip(self, d, right, top, text, col):
        tw = d.textlength(text, font=self.f["chip"])
        d.rounded_rectangle((right - tw - 18, top, right, top + 24), 12, fill=(*col, 230))
        d.text((right - tw - 9, top + 3), text, font=self.f["chip"], fill=(10, 12, 14))

    def header(self, d, name, u):
        tw = d.textlength(LABEL, font=self.f["tag"])
        d.rounded_rectangle((GX, 30, GX + tw + 28, 64), 8, fill=(240, 241, 244))
        d.text((GX + 14, 36), LABEL, font=self.f["tag"], fill=(12, 14, 18))
        d.text(
            (GX + tw + 52, 26),
            "Six training worlds, every walk at once",
            font=self.f["title"],
            fill=INK,
        )
        sub = (
            "the bedroom twice and four obstacle rooms, each on both belt layouts: "
            "12 of the 360 walks scored per policy"
        )
        d.text((GX, 80), sub, font=self.f["sub"], fill=DIM)

    def panel(self, d, arm, name, u, contacts, clean, walking):
        x = PANEL + 6
        y = GY + 14
        d.text((x, y - 18), arm, font=self.f["arm"], fill=INK)
        role = ROLE[arm]
        lines, cur = [], ""
        for word in role.split():
            t = (cur + " " + word).strip()
            if d.textlength(t, font=self.f["role"]) > W - 60 - x:
                lines.append(cur)
                cur = word
            else:
                cur = t
        lines.append(cur)
        for i, ln in enumerate(lines):
            d.text((x, y + 102 + i * 23), ln, font=self.f["role"], fill=DIM)
        y += 102 + len(lines) * 23 + 18
        d.text((x, y), "these 12 rooms", font=self.f["k"], fill=DIM)
        d.text((x, y + 22), f"{clean}", font=self.f["num"], fill=GREEN)
        cw = d.textlength(f"{clean}", font=self.f["num"])
        d.text((x + cw + 8, y + 44), "/12 clean", font=self.f["k"], fill=DIM)
        d.text((x + 190, y + 22), f"{contacts}", font=self.f["num"], fill=RED)
        cw = d.textlength(f"{contacts}", font=self.f["num"])
        d.text((x + 190 + cw + 8, y + 44), "contacts", font=self.f["k"], fill=DIM)
        status = {
            "intro": "about to start",
            "hold": "every walk has finished",
            "update": "policy updates I1 to I5",
            "outro": "every walk has finished",
        }.get(name, f"{walking} still walking" if walking else "every walk has finished")
        d.text((x, y + 82), status, font=self.f["k"], fill=(120, 200, 255))
        self.chart(d, x, y + 128, name, u)

    def chart(self, d, x, y, name, u):
        d.text((x, y), "contacts, all 360 walks per policy", font=self.f["k"], fill=DIM)
        bh, bw, gap = 230, 34, 9
        base = y + 40 + bh
        top = max(c for c, _ in self.tally.values())
        t_i0 = self.seg["I0"][0] + self.seg["I0"][1]
        t_up = self.seg["update"][0]
        t_i6 = self.seg["I6"][0] + self.seg["I6"][1]
        o = self.seg[name][0] + u
        for i, a in enumerate(ARMS):
            if a == "I0":
                grow = smooth((o - t_i0 + 6) / 14)
            elif a == "I6":
                grow = smooth((o - t_i6) / 14)
            else:
                grow = smooth((o - t_up - (i - 1) * 6) / 12)
            bx = x + i * (bw + gap)
            c, k = self.tally[a]
            hgt = bh * c / top * grow
            col = RED if a in SHOWN else BAR
            if hgt > 0.5:
                d.rectangle((bx, base - hgt, bx + bw, base), fill=col)
            if grow > 0.6:
                s = f"{c}"
                d.text(
                    (bx + bw / 2 - d.textlength(s, font=self.f["bar"]) / 2, base - hgt - 20),
                    s,
                    font=self.f["bar"],
                    fill=INK,
                )
                s = f"{k}"
                d.text(
                    (bx + bw / 2 - d.textlength(s, font=self.f["bar"]) / 2, base + 26),
                    s,
                    font=self.f["bar"],
                    fill=GREEN,
                )
            d.text(
                (bx + bw / 2 - d.textlength(a, font=self.f["bar"]) / 2, base + 6),
                a,
                font=self.f["bar"],
                fill=INK if a in SHOWN else DIM,
            )
        d.line((x - 4, base, x + 7 * (bw + gap), base), fill=(*DIM, 160), width=1)
        d.text((x, base + 48), "green: reached clean, of 360", font=self.f["foot"], fill=DIM)
        if name == "outro" and u > 20:
            a = smooth((u - 20) / 15)
            c0, k0 = self.tally["I0"]
            c6, k6 = self.tally["I6"]
            msg = (
                f"I0 to I6, all six worlds:  contacts {c0} to {c6},"
                f"  reached clean {k0} to {k6} of 360"
            )
            tw = d.textlength(msg, font=self.f["banner"])
            yy = GY + 3 * PY + 4
            d.text(
                ((GX + PANEL - 20) / 2 - tw / 2, yy),
                msg,
                font=self.f["banner"],
                fill=(*INK, int(255 * a)),
            )

    def banner(self, img, u):
        a = smooth(u / 8) * (1 - smooth((u - 44) / 10))
        if a <= 0:
            return
        d = ImageDraw.Draw(img, "RGBA")
        msg = "five policy updates, each scored on all 360 walks"
        tw = d.textlength(msg, font=self.f["banner"])
        cx, cy = GX + 2 * PX - 6, GY + 1.5 * PY - 20
        box = (cx - tw / 2 - 28, cy - 34, cx + tw / 2 + 28, cy + 34)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).rounded_rectangle(box, 14, fill=(10, 12, 16, int(225 * a)))
        img.paste(
            Image.alpha_composite(
                img.convert("RGBA"), layer.filter(ImageFilter.GaussianBlur(1))
            ).convert("RGB")
        )
        d = ImageDraw.Draw(img, "RGBA")
        d.text((cx - tw / 2, cy - 19), msg, font=self.f["banner"], fill=(*INK, int(255 * a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--still", type=int, nargs="*")
    a = ap.parse_args()
    reel = Reel()
    for arm in SHOWN:
        ws = reel.walks[arm]
        print(
            arm,
            "shown: contacts",
            sum(len(w.contacts) for w in ws),
            "clean",
            sum(w.clean for w in ws),
            "of",
            len(ws),
            "| ledger 360:",
            reel.tally[arm],
        )
    print("output frames", reel.total, f"({reel.total / FPS:.1f} s)")
    if a.still:
        for o in a.still:
            reel.frame(o).save(a.out.with_name(f"{a.out.stem}-{o:04d}.png"))
        return
    enc = subprocess.Popen(
        [
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
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "16",
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
    for o in range(reel.total):
        enc.stdin.write(np.asarray(reel.frame(o)).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()
