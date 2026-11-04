"""The training-canvas reel: the Blender camera move (scripts/blender_sim_canvas.py) with the
statistics drawn over it, upscaled to 1080p.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Every walk of the ledger has its own tile, 360 in all
(6 worlds x 2 belt layouts x 30 tuning seeds), and all of them walk at once: first with I0 (the
glasses as configured), then with I6 (after five policy updates); then the camera drops into the
bedroom tile and the chair-finding walk replays, I0 then I6. The live counters add up the tiles'
own recorded contacts and finishes as they happen on screen. Every panel is computed from
bench/results/sim-walk.json, 360 walks per policy version, and drawn in the cut's own style: a small
title card top left, and one card top right with the round (I0 to I6) and what changed going into
it, contacts with a sparkline of every round so far, clean-reach rate with its 95% Wilson interval
as a bar, the median closest pass, and once the camera is in the bedroom, that world's contacts.

    PYTHONPATH=. .venv/bin/python scripts/build_sim_canvas.py \
        --out data/demo-takes/sim/montage-v6.mp4 [--canvas DIR] [--still 60 200 ...]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim_canvas_plan as P  # noqa: E402

ROOT = P.ROOT
LEDGER = ROOT / "bench" / "results" / "sim-walk.json"
W, H = 1920, 1080
FONT = "/System/Library/Fonts/SFNS.ttf"
# the light palette of the rest of the cut: dark text on white panels
INK = (17, 17, 19)
DIM = (110, 110, 118)
FAINT = (196, 196, 204)
CYAN = (10, 132, 255)
AMBER = (230, 120, 20)
GREEN = (36, 160, 70)
RED = (255, 69, 58)
LINE = (214, 214, 220)  # the cut's hairline (showcase/rough/build_rough.py)
LABEL = "SIMULATED USER MODEL, NOT A HUMAN STUDY"
ARMS = ("I0", "I1", "I2", "I3", "I4", "I5", "I6")
WORLDS = [w for row in P.GRID for w in row]
SHORT = {
    "avoid_chair": "W1 chair in aisle",
    "find_chair": "W2 bedroom chair",
    "desk_lookup": "W3 desk edge",
    "shelf_head": "W4 shelf",
    "low_box": "W5 low box",
    "chair_legs": "W6 thin legs",
}
WCOL = {
    "avoid_chair": (230, 90, 60),
    "find_chair": (225, 150, 0),
    "desk_lookup": (40, 160, 80),
    "shelf_head": (50, 110, 230),
    "low_box": (150, 80, 220),
    "chair_legs": (0, 150, 160),
}
PHASE_FRAMES = (1, P.UPDATES[0], P.OVERVIEW_I6 + 1, P.FLY[0] + 20, P.CLOSE_I0, P.CLOSE_I6)
# what changed going into each round: the ledger's own hypotheses (sim-walk.json arms), shortened
HYP = {
    "I0": "Start: the glasses as configured",
    "I1": "Changed: add the map's side and back",
    "I2": "Changed: remember what left the view",
    "I3": "Changed: hold the bearing on the cue",
    "I4": "Changed: obstacles must hold 2 frames",
    "I5": "Changed: guide through the navigator",
    "I6": "Changed: cue the two nearest, 45° apart",
}
Z = 1.96


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def wilson(k: int, n: int) -> tuple[float, float, float]:
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return p, c - h, c + h


def shown_version(f: int) -> float:
    """How far along I0..I6 the panels are, as a fractional version index."""
    u0, u1 = P.UPDATES
    if f < u0:
        return 0.0
    if f >= u1:
        return 6.0
    return 5.0 * smooth((f - u0) / (u1 - u0 - 2)) + smooth((f - u1 + 4) / 4)


class Reel:
    def __init__(self, canvas: Path):
        self.canvas = canvas
        self.meta = json.loads((canvas / "canvas.json").read_text())
        self.wins = {w.key: w for w in P.windows()}
        rows = json.loads(LEDGER.read_text())["rows"]
        self.rows = rows
        st = {}
        for a in ARMS:
            rs = [r for r in rows if r["arm"] == a]
            assert len(rs) == 360, a
            clean = [bool(r["success"] and r["collisions"] == 0) for r in rs]
            st[a] = {
                "contacts": sum(r["collisions"] for r in rs),
                "clean": sum(clean),
                "ci": wilson(sum(clean), 360),
                "world": {
                    w: sum(r["collisions"] for r in rs if r["scenario"] == w) for w in WORLDS
                },
                "hist": [sum(min(r["collisions"], 4) == b for r in rs) for b in range(5)],
                "clear_med": statistics.median(r["min_clear_m"] for r in rs),
                "clear_mean": statistics.fmean(r["min_clear_m"] for r in rs),
                "t_med": statistics.median(r["time_s"] for r in rs if r["success"]),
                "reached": sum(bool(r["success"]) for r in rs),
            }
            assert all(sum(r["scenario"] == w for r in rs) == 60 for w in WORLDS)
        self.st = st
        assert (st["I0"]["contacts"], st["I0"]["clean"]) == (467, 69)
        assert (st["I6"]["contacts"], st["I6"]["clean"]) == (57, 305)
        self.hero = {}
        for a in ("I0", "I6"):
            rs = [r for r in rows if r["arm"] == a and r["scenario"] == P.HERO]
            self.hero[a] = (
                sum(r["collisions"] for r in rs),
                sum(bool(r["success"] and r["collisions"] == 0) for r in rs),
            )
        assert self.hero == {"I0": (149, 0), "I6": (2, 58)}, self.hero
        byk = {(r["arm"], r["scenario"], r["layout"], r["seed"]): r for r in rows}
        for world, inf in self.meta["worlds"].items():
            for key, wk in inf["walks"].items():
                r = byk[(wk["arm"], world, P.LAYOUT, P.SEED)]
                assert wk["contacts"] == r["collisions"], key
                assert wk["clean"] == bool(r["success"] and r["collisions"] == 0), key
        # the lattice: one tile per ledger walk, each replay matching its ledger row
        tiles = self.meta["tiles"]
        assert len({(t["world"], t["layout"], t["seed"]) for t in tiles}) == len(tiles) == 360
        for t in tiles:
            for arm, wk in t["walks"].items():
                r = byk[(arm, t["world"], t["layout"], t["seed"])]
                assert wk["contacts"] == r["collisions"] == len(wk["hits"]), (t, arm)
                assert wk["clean"] == bool(r["success"] and r["collisions"] == 0), (t, arm)
        for arm in ("I0", "I6"):
            assert sum(t["walks"][arm]["contacts"] for t in tiles) == st[arm]["contacts"]
        self.live = {
            arm: sorted(
                (
                    (t["walks"][arm]["end"], t["world"], t["layout"], t["seed"], t["walks"][arm])
                    for t in tiles
                ),
                key=lambda e: e[:4],
            )
            for arm in ("I0", "I6")
        }

        def sans(size, weight="Regular"):
            ft = ImageFont.truetype(FONT, size)
            ft.set_variation_by_name(weight)
            return ft

        self.s = {
            "pill": sans(15, "Semibold"),
            "title": sans(28, "Semibold"),
            "phase": sans(19, "Medium"),
            "k": sans(17),
            "kb": sans(17, "Medium"),
            "arm": sans(46, "Semibold"),
            "num": sans(40, "Semibold"),
            "small": sans(15),
        }

    # ------------------------------------------------------------ geometry

    def project(self, f: int, x: float, y: float, z: float):
        m = np.array(self.meta["cams"][f - 1])
        v = m @ np.array([x, y, z, 1.0])
        if v[3] <= 1e-6:
            return None
        return (v[0] / v[3] + 1) / 2 * W, (1 - v[1] / v[3]) / 2 * H

    def walk_state(self, key: str, f: int):
        w = self.wins[key]
        wk = self.meta["worlds"][w.world]["walks"][key]
        n = wk["n"]
        k = w.sim_at(f, n)
        return wk, k, wk["coll"][k], f >= w.start + w.n_out(n) - 1

    def active(self, world: str, f: int) -> str:
        return next(w.key for w in self.wins.values() if w.world == world and w.show <= f < w.hide)

    # ------------------------------------------------------------ pieces

    def chip(self, d, x, y, text, col, a=1.0):
        tw = d.textlength(text, font=self.s["pill"])
        d.rounded_rectangle((x, y, x + tw + 16, y + 24), 6, fill=(*col, int(235 * a)))
        d.text((x + 8, y + 3), text, font=self.s["pill"], fill=(255, 255, 255, int(255 * a)))

    def phase(self, f: int) -> str:
        if f < P.UPDATES[0]:
            return "Round I0: all 360 walks at once"
        if f < P.OVERVIEW_I6:
            return "Rounds I1 to I5: each change scored on the same 360 walks"
        if f < P.FLY[0] + 20:
            return "Round I6: the same 360 walks"
        if f < P.CLOSE_I0:
            return "Into the bedroom: find the chair and sit"
        if f < P.CLOSE_I6:
            return "Bedroom replay with I0, the first try"
        return "Bedroom replay with I6, after five changes"

    def walker_chip(self, d, f):
        if f < P.CLOSE_I0:
            return
        key = self.active(P.HERO, f)
        wk, k, coll, done = self.walk_state(key, f)
        ox, oy = self.meta["offsets"][P.HERO]
        x, y = wk["xy"][k]
        p = self.project(f, x + ox, y + oy, 2.05)
        if p is None:
            return
        arm = wk["arm"]
        if done and wk["clean"]:
            txt, col = f"{arm}  sat down, 0 contacts", GREEN
        elif done:
            txt, col = f"{arm}  {coll} contacts", RED
        else:
            txt = f"{arm}  contacts {coll}"
            col = RED if coll else (AMBER if arm == "I0" else CYAN)
        self.chip(d, p[0] - 60, p[1] - 34, txt, col)

    def glass(self, box, radius=28):
        """The cut's white card: soft shadow, white fill, hairline border."""
        x0, y0, x1, y1 = box
        sh = Image.new("L", self.img.size, 0)
        ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + 10, x1, y1 + 10), radius, fill=40)
        self.img.paste((200, 200, 208), (0, 0), sh.filter(ImageFilter.GaussianBlur(24)))
        ImageDraw.Draw(self.img).rounded_rectangle(
            box, radius, fill=(255, 255, 255), outline=LINE, width=2
        )

    def title_card(self, f):
        """Top left: the sim label, what this is, and the phase."""
        d = ImageDraw.Draw(self.img, "RGBA")
        wide = max(
            d.textlength(LABEL, font=self.s["pill"]) + 24,
            d.textlength("Tuning the belt cues", font=self.s["title"]),
            *(d.textlength(self.phase(k), font=self.s["phase"]) for k in PHASE_FRAMES),
        )
        box = (24, 22, 24 + 52 + int(wide), 152)
        self.glass(box)
        d = ImageDraw.Draw(self.img, "RGBA")
        x, y = box[0] + 26, box[1] + 20
        tw = d.textlength(LABEL, font=self.s["pill"])
        d.rounded_rectangle((x, y, x + tw + 24, y + 26), 13, fill=INK)
        d.text((x + 12, y + 4), LABEL, font=self.s["pill"], fill=(255, 255, 255))
        d.text((x, y + 38), "Tuning the belt cues", font=self.s["title"], fill=INK)
        d.text((x, y + 76), self.phase(f), font=self.s["phase"], fill=CYAN)

    def round_at(self, f: int) -> float:
        if f >= P.CLOSE_I0:
            return 0.0 if f < P.CLOSE_I6 else 6.0
        return shown_version(f)

    def stats_card(self, f):
        """Top right: one round at a time, I0 to I6, from the ledger."""
        box = (1356, 22, 1896, 356)
        self.glass(box)
        d = ImageDraw.Draw(self.img, "RGBA")
        x0, y0, x1 = box[0] + 26, box[1] + 20, box[2] - 26
        v = self.round_at(f)
        i = int(round(v))
        arm = ARMS[i]
        s = self.st[arm]
        # ticker: seven dots, the current round filled
        d.text((x0, y0), "Round", font=self.s["k"], fill=DIM)
        d.text((x0, y0 + 18), arm, font=self.s["arm"], fill=INK)
        for j in range(7):
            cx = x0 + 120 + j * 26
            on = j <= v + 0.01
            col = CYAN if j == i else (INK if on else FAINT)
            r = 7 if j == i else 5
            d.ellipse((cx - r, y0 + 46 - r, cx + r, y0 + 46 + r), fill=col)
        d.text((x0 + 120 - 6, y0 + 60), "I0", font=self.s["small"], fill=DIM)
        d.text((x0 + 120 + 6 * 26 - 8, y0 + 60), "I6", font=self.s["small"], fill=DIM)
        what = HYP[arm]
        d.text((x0, y0 + 92), what, font=self.s["k"], fill=DIM)
        # contacts with a sparkline of every round so far
        y = y0 + 128
        d.text((x0, y), "Contacts, 360 walks", font=self.s["kb"], fill=INK)
        d.text((x0, y + 22), f"{s['contacts']}", font=self.s["num"], fill=RED)
        sx0, sx1, sy0, sy1 = x0 + 110, x0 + 230, y + 30, y + 66
        cmax = max(self.st[a]["contacts"] for a in ARMS)
        pts = [
            (sx0 + (sx1 - sx0) * j / 6, sy1 - (sy1 - sy0) * self.st[a]["contacts"] / cmax)
            for j, a in enumerate(ARMS)
            if j <= i
        ]
        d.line((sx0, sy1, sx1, sy1), fill=FAINT, width=1)
        if len(pts) > 1:
            d.line(pts, fill=RED, width=3, joint="curve")
        d.ellipse((pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4), fill=RED)
        # clean reach with its 95% interval as a bar
        cx = x0 + 262
        p, lo, hi = s["ci"]
        d.text((cx, y), "Clean reach", font=self.s["kb"], fill=INK)
        d.text((cx, y + 22), f"{100 * p:.0f}%", font=self.s["num"], fill=GREEN)
        bx0, bx1, by = cx, x1, y + 84
        d.rounded_rectangle((bx0, by, bx1, by + 8), 4, fill=(232, 232, 236))
        d.rounded_rectangle(
            (bx0 + (bx1 - bx0) * lo, by, bx0 + (bx1 - bx0) * hi, by + 8), 4, fill=(*GREEN, 120)
        )
        px = bx0 + (bx1 - bx0) * p
        d.ellipse((px - 5, by - 1, px + 5, by + 9), fill=GREEN)
        ci_txt = f"95% CI {100 * lo:.0f} to {100 * hi:.0f}%"
        d.text((cx, by + 14), ci_txt, font=self.s["small"], fill=DIM)
        # clearance, then the bedroom replay once the camera is down there
        y = y0 + 252
        d.line((x0, y - 8, x1, y - 8), fill=LINE, width=1)
        d.text((x0, y), "Closest pass, median", font=self.s["k"], fill=DIM)
        cl = f"{100 * s['clear_med']:.1f} cm"
        d.text((x1 - d.textlength(cl, font=self.s["kb"]), y), cl, font=self.s["kb"], fill=INK)
        if f >= P.FLY[0]:
            c0, c6 = self.hero["I0"][0], self.hero["I6"][0]
            d.text((x0, y + 28), "Bedroom chair, 60 walks", font=self.s["k"], fill=DIM)
            t = f"{c0} to {c6} contacts"
            d.text(
                (x1 - d.textlength(t, font=self.s["kb"]), y + 28), t, font=self.s["kb"], fill=INK
            )
        else:
            first = f"from {100 * self.st['I0']['clear_med']:.1f} cm at I0"
            d.text((x0, y + 28), first, font=self.s["k"], fill=DIM)

    def frame(self, f: int) -> Image.Image:
        src = Image.open(self.canvas / "frames" / f"{f:05d}.png").convert("RGB")
        self.img = src.resize((W, H), Image.LANCZOS)
        d = ImageDraw.Draw(self.img, "RGBA")
        self.walker_chip(d, f)
        self.title_card(f)
        self.stats_card(f)
        return self.img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--canvas", type=Path, default=ROOT / "data/demo-takes/sim/canvas6")
    ap.add_argument("--still", type=int, nargs="*")
    a = ap.parse_args()
    reel = Reel(a.canvas)
    for arm in ARMS:
        s = reel.st[arm]
        print(arm, s["contacts"], s["clean"], [round(q, 3) for q in s["ci"]], s["hist"])
    if a.still:
        for f in a.still:
            reel.frame(f).save(a.out.with_name(f"{a.out.stem}-{f:04d}.png"))
        return
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    cmd += ["-s", f"{W}x{H}", "-r", str(P.FPS), "-i", "-", "-c:v", "libx264", "-crf", "16"]
    cmd += ["-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(1, P.FRAMES + 1):
        enc.stdin.write(np.asarray(reel.frame(f)).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()
