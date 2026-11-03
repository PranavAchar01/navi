"""The training-canvas reel: the Blender camera move (scripts/blender_sim_canvas.py) with the
statistics drawn over it, upscaled to 1080p.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Every walk of the ledger has its own tile, 360 in all
(6 worlds x 2 belt layouts x 30 tuning seeds), and all of them walk at once: first with I0 (the
glasses as configured), then with I6 (after five policy updates); then the camera drops into the
bedroom tile and the chair-finding walk replays, I0 then I6. The live counters add up the tiles'
own recorded contacts and finishes as they happen on screen. Every panel is computed from
bench/results/sim-walk.json, 360 walks per policy version:

- contacts per world per version (60 walks a world),
- clean-reach rate per version with 95% Wilson intervals,
- the distribution of contacts per walk,
- minimum clearance and time to goal,
- a run log of the walks as their tiles finish (ledger rows during the updates and close-up).

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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim_canvas_plan as P  # noqa: E402

ROOT = P.ROOT
LEDGER = ROOT / "bench" / "results" / "sim-walk.json"
W, H = 1920, 1080
FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
# the light palette of the rest of the cut: dark text on white panels
INK = (17, 17, 19)
DIM = (110, 110, 118)
FAINT = (196, 196, 204)
CYAN = (10, 132, 255)
AMBER = (230, 120, 20)
GREEN = (36, 160, 70)
RED = (255, 69, 58)
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
ROLE = {
    "I0": "the glasses as configured: only what the camera sees right now",
    "I6": "after five updates: map memory, held bearing, navigator, two cues",
}
PX0, PX1 = 1478, 1896  # the statistics column
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
        # the run log: each version's rows in run order (seed, then world, then layout)
        order = {w: i for i, w in enumerate(WORLDS)}
        self.log = {
            a: sorted(
                (r for r in rows if r["arm"] == a),
                key=lambda r: (r["seed"], order[r["scenario"]], r["layout"]),
            )
            for a in ARMS
        }
        f = lambda p, s: ImageFont.truetype(p, s)  # noqa: E731
        self.f = {
            "tag": f(FONT, 18),
            "title": f(FONT, 38),
            "sub": f(FONT, 19),
            "phase": f(MONO, 19),
            "arm": f(FONT, 64),
            "role": f(FONT, 16),
            "k": f(FONT, 16),
            "num": f(FONT, 38),
            "h": f(MONO, 15),
            "m": f(MONO, 15),
            "chip": f(MONO, 15),
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

    def card(self, box, alpha=200):
        x0, y0, x1, y1 = (int(v) for v in box)
        layer = Image.new("RGBA", (x1 - x0, y1 - y0), (255, 255, 255, min(255, int(alpha * 1.1))))
        self.img.paste(layer, (x0, y0), layer)

    def chip(self, d, x, y, text, col, a=1.0):
        tw = d.textlength(text, font=self.f["chip"])
        d.rounded_rectangle((x, y, x + tw + 16, y + 24), 6, fill=(*col, int(235 * a)))
        d.text((x + 8, y + 3), text, font=self.f["chip"], fill=(255, 255, 255, int(255 * a)))

    def phase(self, f: int) -> str:
        if f < P.UPDATES[0]:
            return "ROUND 1   I0 on all 360 walks at once"
        if f < P.OVERVIEW_I6:
            return "UPDATES   I1 to I5, each scored on the same 360 walks"
        if f < P.FLY[0] + 20:
            return "ROUND 2   I6 on the same 360 walks"
        if f < P.CLOSE_I0:
            return "ZOOM      world W2, bedroom: find the chair and sit"
        if f < P.CLOSE_I6:
            return "REPLAY    W2 with I0, first try"
        return "REPLAY    W2 with I6, after five updates"

    def wrap(self, d, text, x, y, width=310):
        """Two lines at most, broken between words."""
        line, lines = "", []
        for word in text.split():
            cand = f"{line} {word}".strip()
            if line and d.textlength(cand, font=self.f["role"]) > width:
                lines.append(line)
                line = word
            else:
                line = cand
        lines.append(line)
        for i, ln in enumerate(lines[:2]):
            d.text((x, y + 20 * i), ln, font=self.f["role"], fill=DIM)

    def header(self, d, f):
        self.card((16, 14, 960, 142), 228)
        tw = d.textlength(LABEL, font=self.f["tag"])
        d.rounded_rectangle((32, 26, 32 + tw + 28, 58), 8, fill=INK)
        d.text((46, 31), LABEL, font=self.f["tag"], fill=(255, 255, 255))
        d.text((32 + tw + 48, 18), "How the walker was trained", font=self.f["title"], fill=INK)
        sub = (
            "360 simulated walks across 6 worlds (2 belt layouts × 30 tuning seeds)."
            "  Each tile replays one recorded walk."
        )
        d.text((32, 72), sub, font=self.f["sub"], fill=DIM)
        d.text((32, 104), self.phase(f), font=self.f["phase"], fill=CYAN)

    def headline(self, d, f, y):
        x = PX0
        if f < P.UPDATES[0] or P.OVERVIEW_I6 <= f < P.CLOSE_I0:
            arm = "I0" if f < P.UPDATES[0] else "I6"
            d.text((x, y - 6), arm, font=self.f["arm"], fill=INK)
            self.wrap(d, ROLE[arm], x + 96, y + 6)
            walks = [e[4] for e in self.live[arm]]
            done = [w for e, w in zip(self.live[arm], walks, strict=True) if e[0] <= f]
            c = sum(sum(h <= f for h in w["hits"]) for w in walks)
            k = sum(w["clean"] for w in done)
            label = f"live: {len(done)} of 360 walks finished"
            of = len(done)
        elif f < P.CLOSE_I0:
            v = shown_version(f)
            arm = ARMS[int(round(v))] if v < 0.5 or v > 5.5 else f"I{max(1, min(5, round(v)))}"
            d.text((x, y - 6), arm, font=self.f["arm"], fill=INK)
            role = ROLE.get(arm, "policy update, scored on all 360 walks")
            self.wrap(d, role, x + 96, y + 6)
            s = self.st[arm]
            label = "all six worlds, 360 walks"
            c, k, of = s["contacts"], s["clean"], 360
        else:
            arm = "I0" if f < P.CLOSE_I6 else "I6"
            d.text((x, y - 6), arm, font=self.f["arm"], fill=INK)
            what = "first try" if arm == "I0" else "after five updates"
            d.text((x + 96, y + 6), what, font=self.f["role"], fill=DIM)
            d.text((x + 96, y + 26), "W2 bedroom, find the chair", font=self.f["role"], fill=DIM)
            c, k = self.hero[arm]
            label, of = "W2 only, 60 walks (2 layouts × 30 seeds)", 60
        d.text((x, y + 74), label, font=self.f["k"], fill=DIM)
        d.text((x, y + 94), f"{c}", font=self.f["num"], fill=RED)
        cw = d.textlength(f"{c}", font=self.f["num"])
        d.text((x + cw + 8, y + 114), "contacts", font=self.f["k"], fill=DIM)
        d.text((x + 196, y + 94), f"{k}", font=self.f["num"], fill=GREEN)
        cw = d.textlength(f"{k}", font=self.f["num"])
        d.text((x + 196 + cw + 6, y + 114), f"/{of} clean", font=self.f["k"], fill=DIM)
        return y + 150

    def axis_x(self, d, x0, x1, y, v):
        step = (x1 - x0) / 6
        for i, a in enumerate(ARMS):
            col = INK if i <= v + 0.01 else FAINT
            tw = d.textlength(a, font=self.f["m"])
            d.text((x0 + i * step - tw / 2, y + 4), a, font=self.f["m"], fill=col)
        return step

    def lines(self, d, f, y):
        """Contacts per world, per version: six lines, drawn up to the version on screen."""
        x0, x1 = PX0 + 34, PX1 - 14
        d.text((PX0, y), "CONTACTS PER WORLD (60 walks each)", font=self.f["h"], fill=DIM)
        top, bot = y + 30, y + 170
        ymax = 150
        for t in (0, 50, 100, 150):
            yy = bot - (bot - top) * t / ymax
            d.line((x0, yy, x1, yy), fill=(*FAINT, 120), width=1)
            d.text((PX0, yy - 9), f"{t}", font=self.f["m"], fill=DIM)
        v = shown_version(f)
        step = self.axis_x(d, x0, x1, bot, v)
        hero_on = f >= P.FLY[0]
        for w in WORLDS:
            vals = [self.st[a]["world"][w] for a in ARMS]
            pts = []
            for i, c in enumerate(vals):
                if i > v:
                    break
                pts.append((x0 + i * step, bot - (bot - top) * c / ymax))
            fr = v - int(v)
            if int(v) < 6 and fr > 0:  # the segment growing toward the next version
                a, b = int(v), int(v) + 1
                ya = bot - (bot - top) * vals[a] / ymax
                yb = bot - (bot - top) * vals[b] / ymax
                pts.append((x0 + (a + fr) * step, ya + (yb - ya) * fr))
            hot = w == P.HERO and hero_on
            col = WCOL[w] if (not hero_on or hot) else tuple(int(c * 0.45) for c in WCOL[w])
            if len(pts) > 1:
                d.line(pts, fill=col, width=4 if hot else 2, joint="curve")
            for p in pts[: int(v) + 1]:
                d.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=col)
            if hot or not hero_on:
                c = vals[min(int(v), 6)]
                end = pts[min(int(v), len(pts) - 1)]
                if hot:
                    d.text((end[0] + 8, end[1] - 20), f"{c}", font=self.f["m"], fill=col)
        # legend, two rows of three
        ly = bot + 26
        for i, w in enumerate(WORLDS):
            lx = PX0 + (i % 2) * 210
            yy = ly + (i // 2) * 19
            hot = w == P.HERO and hero_on
            col = WCOL[w] if (not hero_on or hot) else tuple(int(c * 0.45) for c in WCOL[w])
            d.rectangle((lx, yy + 6, lx + 12, yy + 10), fill=col)
            d.text((lx + 17, yy), SHORT[w], font=self.f["m"], fill=INK if hot else DIM)
        return ly + 62

    def ci(self, d, f, y):
        """Clean-reach rate per version with 95% Wilson intervals."""
        x0, x1 = PX0 + 34, PX1 - 14
        d.text((PX0, y), "CLEAN-REACH RATE, 95% WILSON CI", font=self.f["h"], fill=DIM)
        top, bot = y + 28, y + 118
        for t in (0, 50, 100):
            yy = bot - (bot - top) * t / 100
            d.line((x0, yy, x1, yy), fill=(*FAINT, 120), width=1)
            d.text((PX0, yy - 9), f"{t}%", font=self.f["m"], fill=DIM)
        v = shown_version(f)
        step = self.axis_x(d, x0, x1, bot, v)
        for i, a in enumerate(ARMS):
            g = smooth((v - i + 1) * 1.5)
            if g <= 0.02:
                continue
            p, lo, hi = self.st[a]["ci"]
            x = x0 + i * step
            col = CYAN if a == "I6" else (AMBER if a == "I0" else (200, 208, 220))
            col = (*col, int(255 * g))
            if a not in ("I0", "I6"):
                s = f"{100 * p:.0f}"
                tw = d.textlength(s, font=self.f["m"])
                d.text((x - tw / 2, bot - (bot - top) * hi - 20), s, font=self.f["m"], fill=col)
            ylo, yhi, yp = (bot - (bot - top) * q for q in (lo, hi, p))
            d.line((x, ylo, x, yhi), fill=col, width=2)
            d.line((x - 6, ylo, x + 6, ylo), fill=col, width=2)
            d.line((x - 6, yhi, x + 6, yhi), fill=col, width=2)
            d.ellipse((x - 5, yp - 5, x + 5, yp + 5), fill=col)
            if a in ("I0", "I6"):
                s = f"{100 * p:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]"
                tw = d.textlength(s, font=self.f["m"])
                tx, ty = (x + 10, yp - 9) if a == "I0" else (PX1 - tw, top - 22)
                d.text((tx, ty), s, font=self.f["m"], fill=col)
        return bot + 30

    def hist(self, d, f, y):
        """Contacts per walk: share of the 360 walks with 0, 1, 2, 3, 4+ contacts."""
        d.text((PX0, y), "CONTACTS PER WALK, SHARE OF 360", font=self.f["h"], fill=DIM)
        cx0 = PX0 + 40
        cw, ch = 74, 19
        for b, lab in enumerate(("0", "1", "2", "3", "4+")):
            tw = d.textlength(lab, font=self.f["m"])
            d.text((cx0 + b * cw + cw / 2 - tw / 2, y + 22), lab, font=self.f["m"], fill=DIM)
        v = shown_version(f)
        for i, a in enumerate(ARMS):
            yy = y + 42 + i * (ch + 1)
            on = i <= v + 0.01
            d.text((PX0, yy + 1), a, font=self.f["m"], fill=INK if on else FAINT)
            for b, n in enumerate(self.st[a]["hist"]):
                x = cx0 + b * cw
                if not on:
                    d.rectangle((x, yy, x + cw - 3, yy + ch), outline=(*FAINT, 120))
                    continue
                share = n / 360
                base = GREEN if b == 0 else RED
                fill = (*base, int(40 + 200 * share))
                d.rectangle((x, yy, x + cw - 3, yy + ch), fill=fill)
                s = f"{n}"
                tw = d.textlength(s, font=self.f["m"])
                d.text((x + cw / 2 - tw / 2, yy + 1), s, font=self.f["m"], fill=INK)
        return y + 42 + 7 * (ch + 1) + 12

    def table(self, d, f, y):
        """Minimum clearance and time to goal per version."""
        d.text((PX0, y), "CLEARANCE AND TIME, 360 WALKS", font=self.f["h"], fill=DIM)
        cols = (PX0, PX0 + 70, PX0 + 190, PX0 + 310)
        heads = ("ver", "min clr med", "min clr mean", "t goal med")
        for x, h in zip(cols, heads, strict=True):
            d.text((x, y + 22), h, font=self.f["m"], fill=DIM)
        v = shown_version(f)
        rows = [a for i, a in enumerate(ARMS) if i <= v + 0.01]
        for j, a in enumerate(rows):
            s = self.st[a]
            yy = y + 42 + j * 19
            col = CYAN if a == "I6" else (AMBER if a == "I0" else INK)
            vals = (
                a,
                f"{100 * s['clear_med']:.1f} cm",
                f"{100 * s['clear_mean']:.1f} cm",
                f"{s['t_med']:.2f} s",
            )
            for x, t in zip(cols, vals, strict=True):
                d.text((x, yy), t, font=self.f["m"], fill=col)
        return y + 44 + 7 * 19

    def runlog(self, d, f):
        """A ticker of the ledger rows for the version on screen."""
        x0, y0, w, n = 28, H - 214, 700, 8
        self.card((x0 - 12, y0 - 10, x0 + w, H - 18), 205)
        d.text((x0, y0), "RUN LOG   bench/results/sim-walk.json", font=self.f["h"], fill=DIM)
        v = shown_version(f)
        if f < P.UPDATES[0] or P.OVERVIEW_I6 <= f < P.CLOSE_I0:
            arm = "I0" if f < P.UPDATES[0] else "I6"
            done = [e for e in self.live[arm] if e[0] <= f][::-1][: n - 1]
            head = f"{'ver':<4}{'world':<19}{'layout':<7}{'seed':<6}{'contacts':>8}  outcome"
            d.text((x0, y0 + 24), head, font=self.f["m"], fill=FAINT)
            for j, (_, world, lay, sd, wk) in enumerate(done):
                out = "clean" if wk["clean"] else "reached"
                line = f"{arm:<4}{SHORT[world]:<19}{lay:<7}{sd:<6}{wk['contacts']:>8}  {out}"
                col = GREEN if wk["clean"] else (RED if wk["contacts"] else INK)
                d.text((x0, y0 + 46 + j * 20), line, font=self.f["m"], fill=col)
            return
        if f >= P.CLOSE_I0:
            arm = "I0" if f < P.CLOSE_I6 else "I6"
            rows = [r for r in self.log[arm] if r["scenario"] == P.HERO]
            t = f - (P.CLOSE_I0 if arm == "I0" else P.CLOSE_I6)
        else:
            arm = ARMS[min(6, int(v + 0.5))]
            rows = self.log[arm]
            t = f - (1 if f < P.UPDATES[0] else P.UPDATES[0] if f < P.OVERVIEW_I6 else 150)
        head = f"{'ver':<4}{'world':<19}{'layout':<7}{'seed':<6}{'contacts':>8}  outcome"
        d.text((x0, y0 + 24), head, font=self.f["m"], fill=FAINT)
        first = int(t * 0.7)  # about 21 rows a second scroll past
        for j in range(n - 1):
            r = rows[(first + j) % len(rows)]
            clean = r["success"] and r["collisions"] == 0
            out = "clean" if clean else ("reached" if r["success"] else "timed out")
            line = (
                f"{r['arm']:<4}{SHORT[r['scenario']]:<19}{r['layout']:<7}{r['seed']:<6}"
                f"{r['collisions']:>8}  {out}"
            )
            col = GREEN if clean else (RED if r["collisions"] else INK)
            a = 255 if j < n - 2 else 140
            d.text((x0, y0 + 46 + j * 20), line, font=self.f["m"], fill=(*col, a))

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

    def summary(self, d, f):
        if f < 432:
            return
        a = smooth((f - 432) / 10)
        s0, s6 = self.st["I0"], self.st["I6"]
        msg = (
            f"all six worlds, I0 to I6:  contacts {s0['contacts']} to {s6['contacts']},"
            f"  clean {s0['clean']} to {s6['clean']} of 360"
        )
        font = self.f["sub"]
        tw = d.textlength(msg, font=font)
        x, y = 760, H - 64
        layer = Image.new("RGBA", (int(tw + 36), 42), (255, 255, 255, int(235 * a)))
        self.img.paste(layer, (x - 18, y - 10), layer)
        d.text((x, y), msg, font=font, fill=(*INK, int(255 * a)))

    def frame(self, f: int) -> Image.Image:
        src = Image.open(self.canvas / "frames" / f"{f:05d}.png").convert("RGB")
        self.img = src.resize((W, H), Image.LANCZOS)
        d = ImageDraw.Draw(self.img, "RGBA")
        self.walker_chip(d, f)
        self.header(d, f)
        self.card((PX0 - 18, 14, W - 10, H - 14), 238)
        y = self.headline(d, f, 26)
        y = self.lines(d, f, y + 6)
        y = self.ci(d, f, y + 4)
        y = self.hist(d, f, y)
        self.table(d, f, y)
        self.runlog(d, f)
        self.summary(d, f)
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
