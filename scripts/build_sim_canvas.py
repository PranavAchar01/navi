"""The training-canvas reel: the Blender camera move (scripts/blender_sim_canvas.py) with the
numbers drawn over it, upscaled to 1080p.

SIMULATED USER MODEL, NOT A HUMAN STUDY. The six worlds of the walk ledger sit on one ground and
all walk at once, first with I0 (the glasses as configured), then with I6 (after five policy
updates); then the camera drops into the bedroom and the chair-finding walk replays, I0 then I6.
Live counters are the recorded walks on screen (seed 1000, eight motors); every total comes from
bench/results/sim-walk.json, 360 walks per policy (6 worlds x 2 belt layouts x 30 seeds), the
seeds the policies were tuned on.

    PYTHONPATH=. .venv/bin/python scripts/build_sim_canvas.py \
        --out data/demo-takes/sim/montage-v5.mp4 [--still 60 200 ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim_canvas_plan as P  # noqa: E402

ROOT = P.ROOT
CANVAS = ROOT / "data" / "demo-takes" / "sim" / "canvas"
LEDGER = ROOT / "bench" / "results" / "sim-walk.json"
W, H = 1920, 1080
FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
INK = (236, 240, 245)
DIM = (138, 150, 166)
CYAN = (96, 210, 255)
AMBER = (255, 160, 70)
GREEN = (70, 226, 130)
RED = (255, 80, 64)
BAR = (88, 98, 116)
LABEL = "SIMULATED USER MODEL, NOT A HUMAN STUDY"
ARMS = ("I0", "I1", "I2", "I3", "I4", "I5", "I6")
ROLE = {
    "I0": "the glasses as configured: only what the camera sees right now",
    "I6": "after five updates: map memory, held bearing, navigator, two cues",
}
PANEL_X = 1528


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def fade(f: int, a: int, b: int, ramp: int = 8) -> float:
    """1 inside [a, b), easing in and out over `ramp` frames."""
    return smooth((f - a) / ramp) * (1 - smooth((f - b + ramp) / ramp))


class Reel:
    def __init__(self):
        self.meta = json.loads((CANVAS / "canvas.json").read_text())
        self.k = W / self.meta["w"]
        self.wins = {w.key: w for w in P.windows()}
        rows = json.loads(LEDGER.read_text())["rows"]
        self.tally = {}
        for a in ARMS:
            rs = [r for r in rows if r["arm"] == a]
            assert len(rs) == 360, a
            self.tally[a] = (
                sum(r["collisions"] for r in rs),
                sum(bool(r["success"] and r["collisions"] == 0) for r in rs),
            )
        self.hero = {}
        for a in ("I0", "I6"):
            rs = [r for r in rows if r["arm"] == a and r["scenario"] == P.HERO]
            assert len(rs) == 60
            self.hero[a] = (
                sum(r["collisions"] for r in rs),
                sum(bool(r["success"] and r["collisions"] == 0) for r in rs),
            )
        assert self.tally["I0"] == (467, 69) and self.tally["I6"] == (57, 305), self.tally
        assert self.hero == {"I0": (149, 0), "I6": (2, 58)}, self.hero
        # the walks on screen match their ledger rows
        byk = {(r["arm"], r["scenario"], r["layout"], r["seed"]): r for r in rows}
        for world, inf in self.meta["worlds"].items():
            for key, wk in inf["walks"].items():
                r = byk[(wk["arm"], world, P.LAYOUT, P.SEED)]
                assert wk["contacts"] == r["collisions"], key
                assert wk["clean"] == bool(r["success"] and r["collisions"] == 0), key
        f = lambda p, s: ImageFont.truetype(p, s)  # noqa: E731
        self.f = {
            "tag": f(FONT, 18),
            "title": f(FONT, 40),
            "sub": f(FONT, 20),
            "arm": f(FONT, 88),
            "role": f(FONT, 18),
            "k": f(FONT, 17),
            "num": f(FONT, 44),
            "bar": f(MONO, 15),
            "chip": f(MONO, 15),
            "banner": f(FONT, 30),
            "foot": f(FONT, 15),
        }

    # ------------------------------------------------------------ geometry

    def project(self, f: int, x: float, y: float, z: float):
        m = np.array(self.meta["cams"][f - 1])
        v = m @ np.array([x, y, z, 1.0])
        if v[3] <= 1e-6:
            return None
        nx, ny = v[0] / v[3], v[1] / v[3]
        return (nx + 1) / 2 * W, (1 - ny) / 2 * H

    def walk_state(self, key: str, f: int):
        w = self.wins[key]
        wk = self.meta["worlds"][w.world]["walks"][key]
        n = wk["n"]
        k = w.sim_at(f, n)
        done = f >= w.start + w.n_out(n) - 1
        return wk, k, wk["coll"][k], done

    def active(self, world: str, f: int) -> str:
        return next(w.key for w in self.wins.values() if w.world == world and w.show <= f < w.hide)

    # ------------------------------------------------------------ pieces

    def chip(self, d, x, y, text, col, a=1.0, fg=(10, 12, 16)):
        tw = d.textlength(text, font=self.f["chip"])
        d.rounded_rectangle((x, y, x + tw + 16, y + 24), 6, fill=(*col, int(235 * a)))
        d.text((x + 8, y + 3), text, font=self.f["chip"], fill=(*fg, int(255 * a)))

    def header(self, d, f):
        top = Image.new("RGBA", (W, 170), (0, 0, 0, 0))
        g = np.linspace(200, 0, 170).astype(np.uint8)
        top.putalpha(Image.fromarray(np.repeat(g[:, None], W, axis=1)))
        self.img.paste((6, 8, 12), (0, 0, W, 170), top)
        tw = d.textlength(LABEL, font=self.f["tag"])
        d.rounded_rectangle((36, 30, 36 + tw + 28, 62), 8, fill=(240, 241, 244))
        d.text((50, 35), LABEL, font=self.f["tag"], fill=(12, 14, 18))
        d.text((36 + tw + 52, 22), "How the walker was trained", font=self.f["title"], fill=INK)
        sub = (
            "six simulated rooms × 2 belt layouts × 30 seeds = 360 walks scored per policy"
            " version.  On screen: one walk per room (seed 1000, 8 motors)"
        )
        d.text((36, 78), sub, font=self.f["sub"], fill=DIM)

    def panel(self, d, f):
        x = PANEL_X
        y = 150
        layer = Image.new("RGBA", (W - x + 20, H - y - 40), (8, 10, 14, 205))
        self.img.paste(layer, (x - 20, y), layer)
        if f < P.CLOSE_I0:
            arm = "I0" if f < P.OVERVIEW_I6 else "I6"
            if P.UPDATES[0] <= f < P.OVERVIEW_I6:
                arm = "I1-I5"
            d.text((x, y + 4), arm, font=self.f["arm"], fill=INK)
            role = ROLE.get(arm, "five policy updates, each scored on all 360 walks")
            yy = self.wrap(d, role, x, y + 110, self.f["role"], DIM)
            d.text((x, yy + 14), "all six worlds, 360 walks", font=self.f["k"], fill=DIM)
            if arm in self.tally:
                c, k = self.tally[arm]
                self.big(d, x, yy + 36, c, k, 360)
            yy += 130
        else:
            arm = "I0" if f < P.CLOSE_I6 else "I6"
            d.text((x, y + 4), arm, font=self.f["arm"], fill=INK)
            what = "first try, before any update" if arm == "I0" else "after five updates"
            yy = self.wrap(d, what, x, y + 110, self.f["role"], DIM)
            d.text((x, yy + 14), "bedroom, find the chair", font=self.f["k"], fill=DIM)
            d.text((x, yy + 34), "60 walks: 2 layouts × 30 seeds", font=self.f["k"], fill=DIM)
            c, k = self.hero[arm]
            self.big(d, x, yy + 58, c, k, 60)
            yy += 152
        self.chart(d, x, yy + 18, f)

    def wrap(self, d, text, x, y, font, col):
        lines, cur = [], ""
        for word in text.split():
            t = (cur + " " + word).strip()
            if d.textlength(t, font=font) > W - 40 - x:
                lines.append(cur)
                cur = word
            else:
                cur = t
        lines.append(cur)
        for i, ln in enumerate(lines):
            d.text((x, y + i * 23), ln, font=font, fill=col)
        return y + len(lines) * 23

    def big(self, d, x, y, contacts, clean, of):
        d.text((x, y), f"{contacts}", font=self.f["num"], fill=RED)
        cw = d.textlength(f"{contacts}", font=self.f["num"])
        d.text((x + cw + 8, y + 22), "contacts", font=self.f["k"], fill=DIM)
        d.text((x + 186, y), f"{clean}", font=self.f["num"], fill=GREEN)
        cw = d.textlength(f"{clean}", font=self.f["num"])
        d.text((x + 186 + cw + 6, y + 22), f"/{of} clean", font=self.f["k"], fill=DIM)

    def chart(self, d, x, y, f):
        d.text((x, y), "contacts per policy, 360 walks each", font=self.f["k"], fill=DIM)
        bh, bw, gap = 190, 34, 9
        base = y + 36 + bh
        top = max(c for c, _ in self.tally.values())
        u0, u1 = P.UPDATES
        for i, a in enumerate(ARMS):
            if a == "I0":
                grow = smooth((f - 100) / 14)
            elif a == "I6":
                grow = smooth((f - u1) / 10)
            else:
                grow = smooth((f - u0 - (i - 1) * 5) / 8)
            bx = x + i * (bw + gap)
            c, k = self.tally[a]
            hgt = bh * c / top * grow
            col = (AMBER if a == "I0" else CYAN) if a in ("I0", "I6") else BAR
            if hgt > 0.5:
                d.rectangle((bx, base - hgt, bx + bw, base), fill=col)
            if grow > 0.6:
                s = f"{c}"
                tw = d.textlength(s, font=self.f["bar"])
                d.text((bx + bw / 2 - tw / 2, base - hgt - 20), s, font=self.f["bar"], fill=INK)
                s = f"{k}"
                tw = d.textlength(s, font=self.f["bar"])
                d.text((bx + bw / 2 - tw / 2, base + 26), s, font=self.f["bar"], fill=GREEN)
            tw = d.textlength(a, font=self.f["bar"])
            fill = INK if a in ("I0", "I6") else DIM
            d.text((bx + bw / 2 - tw / 2, base + 6), a, font=self.f["bar"], fill=fill)
        d.line((x - 4, base, x + 7 * (bw + gap), base), fill=(*DIM, 160), width=1)
        d.text((x, base + 48), "green: reached clean, of 360", font=self.f["foot"], fill=DIM)

    def room_chips(self, d, f):
        a = 1 - smooth((f - P.FLY[0] - 10) / 24)
        if a <= 0.02:
            return
        offs = self.meta["offsets"]
        for world, inf in self.meta["worlds"].items():
            key = self.active(world, f)
            if key.startswith("close"):
                continue
            wk, k, coll, done = self.walk_state(key, f)
            ox, oy = offs[world]
            tx, ty, tz = inf["tag"]
            p = self.project(f, tx + ox, ty + oy, tz)
            if p is None:
                continue
            arm = wk["arm"]
            if done and wk["clean"]:
                txt, col = f"{arm}  reached clean", GREEN
            elif done:
                txt, col = f"{arm}  {coll} contact{'s' * (coll != 1)}, done", RED
            elif coll:
                txt, col = f"{arm}  {coll} contact{'s' * (coll != 1)}", RED
            else:
                txt, col = f"{arm}  walking", CYAN
            self.chip(d, p[0] - 40, max(p[1] - 30, 108), txt, col, a)  # clear of the header

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

    def caption(self, d, f, text, a):
        if a <= 0.01:
            return
        tw = d.textlength(text, font=self.f["banner"])
        cx = (PANEL_X - 20) / 2
        y = H - 92
        layer = Image.new("RGBA", (int(tw + 56), 64), (8, 10, 14, int(215 * a)))
        self.img.paste(layer, (int(cx - tw / 2 - 28), y - 12), layer)
        d.text((cx - tw / 2, y), text, font=self.f["banner"], fill=(*INK, int(255 * a)))

    def frame(self, f: int) -> Image.Image:
        src = Image.open(CANVAS / "frames" / f"{f:05d}.png").convert("RGB")
        self.img = src.resize((W, H), Image.LANCZOS)
        d = ImageDraw.Draw(self.img, "RGBA")
        self.header(d, f)
        self.room_chips(d, f)
        self.walker_chip(d, f)
        self.panel(d, f)
        c0, k0 = self.tally["I0"]
        c6, k6 = self.tally["I6"]
        h0, h6 = self.hero["I0"], self.hero["I6"]
        caps = (
            (1, 100, "round 1: I0 walks all six worlds at once"),
            (P.UPDATES[0], P.OVERVIEW_I6 + 4, "five policy updates, each scored on all 360 walks"),
            (P.OVERVIEW_I6 + 4, P.FLY[0] + 30, "round 2: I6, same six worlds"),
            (P.FLY[0] + 30, P.CLOSE_I0 + 4, "into world 2: the bedroom, find the chair and sit"),
            (P.CLOSE_I0 + 4, P.CLOSE_I6, f"I0 here: {h0[0]} contacts over 60 walks, {h0[1]} clean"),
            (P.CLOSE_I6, 432, f"I6 here: {h6[0]} contacts over 60 walks, {h6[1]}/60 clean"),
            (
                432,
                P.FRAMES + 9,
                f"all six worlds, I0 to I6: contacts {c0} to {c6}, clean {k0} to {k6} of 360",
            ),
        )
        for a, b, text in caps:
            self.caption(d, f, text, fade(f, a, b))
        return self.img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--still", type=int, nargs="*")
    a = ap.parse_args()
    reel = Reel()
    print("ledger I0", reel.tally["I0"], "I6", reel.tally["I6"], "bedroom", reel.hero)
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
