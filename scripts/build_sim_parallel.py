"""Sixty simulated walks playing at once, round after round, laid out in 2D from the per-room
tile clips (scripts/blender_sim_tile.py). No new 3D render: each tile's frames are reused, the
walker's path is drawn over its room from the scene file through the tile camera, and the
counter, the rings and the chart are all driven by the same walks.

SIMULATED USER MODEL, NOT A HUMAN STUDY.

    montage   find the chair and sit, I0 -> I3 -> I5 -> I6 (data/demo-takes/sim/grid + tiles)
    retrieval find my laptop, I0 -> I6 (data/demo-takes/sim/retrieval, run_sim_retrieval.py)

Every walk plays from its own first frame at the same moment. A room flashes red on each
contact the walk really made and keeps a red edge; it turns green only when the walk reached its
goal without touching anything. The tallies are checked against the scored results before a
frame is drawn.

    PYTHONPATH=. .venv/bin/python scripts/build_sim_parallel.py montage \
        --out data/demo-takes/sim/montage-v3.mp4 [--still 300 --still-out x.png]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "data" / "demo-takes" / "sim"
CAM = SIM / "frames" / "final-I5-find_chair-eight" / "base" / "cam.json"  # the tiles' camera
W, H = 1920, 1080
FPS = 30
TILE_W, TILE_H = 384, 216
CROP = (92, 7, 284, 177)  # the room inside a tile frame
COLS, ROWS = 10, 6
GX, GY = 36, 138  # grid origin
CW, CH = 148, 131  # cell pitch
TW, TH = 142, 125  # tile size inside a cell
PANEL_X = 1540
FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
BG = (9, 11, 15)
INK = (236, 238, 242)
DIM = (128, 136, 150)
GREEN = (58, 232, 120)
RED = (255, 64, 52)
CYAN = (90, 200, 255)
AMBER = (255, 196, 70)
LABEL = "SIMULATED USER MODEL, NOT A HUMAN STUDY"

SPEED = 3.0  # sim frames per output frame while a round plays
INTRO, SWEEP, OUTRO = 30, 30, 75  # output frames
FLASH = 10  # output frames a contact flash takes to fade


@dataclass
class Walk:
    tile: str
    frames_dir: Path
    xy: np.ndarray  # (n, 2) floor positions per sim frame
    heading: np.ndarray
    contacts: list[int]  # sim frames where the walk's contact count went up
    end: int  # last walking frame
    clean: bool
    files: list[int] = field(default_factory=list)


@dataclass
class Round:
    arm: str
    caption: str
    change: str  # shown on the policy-update sweep into this round
    walks: list[Walk]
    length: int  # sim frames played


def load_cam() -> np.ndarray:
    return np.array(json.loads(CAM.read_text())["ndc_from_world"])


def project(m: np.ndarray, pts: np.ndarray, z: float = 0.05) -> np.ndarray:
    """Floor points -> pixel positions inside a tile cell (TW x TH)."""
    p = np.c_[pts, np.full(len(pts), z), np.ones(len(pts))] @ m.T
    p = p[:, :3] / p[:, 3:4]
    u = (p[:, 0] + 1) / 2 * TILE_W
    v = (1 - (p[:, 1] + 1) / 2) * TILE_H
    sx, sy = TW / (CROP[2] - CROP[0]), TH / (CROP[3] - CROP[1])
    return np.c_[(u - CROP[0]) * sx, (v - CROP[1]) * sy]


def walk_from_scene(scene: Path, frames_dir: Path, tile: str) -> Walk:
    s = json.loads(scene.read_text())
    fr = s["frames"]
    col = [f["collisions"] for f in fr]
    contacts = [k for k in range(len(col)) if col[k] > (col[k - 1] if k else 0)]
    end = max((k for k, f in enumerate(fr) if f.get("walking", True)), default=len(fr) - 1)
    files = sorted(int(p.stem) for p in frames_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no tile frames in {frames_dir}")
    return Walk(
        tile,
        frames_dir,
        np.array([[f["x"], f["y"]] for f in fr]),
        np.array([f["heading"] for f in fr]),
        contacts,
        end,
        bool(s["metrics"]["success"]),  # scene_file: reached and nothing touched
        files,
    )


# ------------------------------------------------------------------ the two reels


def montage_rounds() -> tuple[list[Round], dict]:
    grid = SIM / "grid"
    tl = json.loads((grid / "timeline.json").read_text())
    ledger = json.loads((ROOT / "bench" / "results" / "sim-walk.json").read_text())["rows"]
    led = {(r["arm"], r["layout"], r["seed"]): r for r in ledger if r["scenario"] == "find_chair"}
    tiles = sorted(tl["tiles"], key=lambda t: (t["row"], t["col"]))
    changes = {
        "I0": "",
        "I3": "the map's side and back sectors, and a held bearing on the cue",
        "I5": "the guide routed by the app's navigator",
        "I6": "the second-nearest object cued too, not just the nearest",
    }
    rounds = []
    for ph in tl["phases"]:
        a = ph["arm"]
        walks = []
        for t in tiles:
            w = walk_from_scene(
                grid / "scenes" / f"{a}-{t['id']}.json", SIM / "tiles" / t["id"] / a, t["id"]
            )
            r = led[(a, t["layout"], t["seed"])]
            assert len(w.contacts) == r["collisions"], (a, t["id"])
            assert w.clean == (r["success"] and r["collisions"] == 0), (a, t["id"])
            walks.append(w)
        # play until the last contact or clean arrival, every walk's whole story inside it
        length = min(330, max(len(w.xy) for w in walks))  # the tiles were rendered to 330
        assert all(c < length for w in walks for c in w.contacts), a
        rounds.append(Round(a, ph["caption"], changes[a], walks, length))
    meta = {
        "title": "Find the chair and sit",
        "sub": "60 rooms, one walker policy, 60 noise seeds, all running at once",
        "goal_word": "seated without touching anything",
        "chart_title": "contacts per round",
        "item": None,
        "flow": "policy update",
    }
    return rounds, meta


def retrieval_rounds() -> tuple[list[Round], dict]:
    base = SIM / "retrieval"
    res = json.loads((ROOT / "bench" / "results" / "sim-retrieval.json").read_text())
    rows = {(r["arm"], r["layout"], r["seed"]): r for r in res["rows"]}
    arms = list(res["summary"])
    tiles = [(lay, s) for lay in ("eight", "diag4") for s in range(1000, 1030)]
    caps = {
        "I0": "the glasses as configured: only what the camera sees right now",
        "I6": "the current walker: map sectors, held bearing, navigator, two cues",
    }
    rounds = []
    for a in arms:
        walks = []
        for lay, s in tiles:
            tid = f"{lay}-{s}"
            w = walk_from_scene(base / "scenes" / f"{a}-{tid}.json", base / "tiles" / tid / a, tid)
            r = rows[(a, lay, s)]
            assert len(w.contacts) == r["collisions"] and w.clean == r["reached"], (a, tid)
            walks.append(w)
        length = min(450, max(len(w.xy) for w in walks))  # these tiles were rendered to 450
        assert all(c < length for w in walks for c in w.contacts), a
        rounds.append(Round(a, caps.get(a, a), caps.get(a, ""), walks, length))
    item = (
        res["rows"][0]
        and json.loads((base / "scenes" / f"{arms[0]}-eight-1000.json").read_text())["item"]
    )
    meta = {
        "title": "Find my things: “where's my laptop?”",
        "sub": "60 rooms, 60 start spots and headings, all running at once",
        "goal_word": "reached the laptop without touching anything",
        "chart_title": "contacts per round",
        "item": item,
        "flow": "policy update",
    }
    return rounds, meta


# ------------------------------------------------------------------ drawing


class Reel:
    def __init__(self, rounds: list[Round], meta: dict):
        self.rounds, self.meta = rounds, meta
        self.m = load_cam()
        self.f = {
            k: ImageFont.truetype(FONT, s)
            for k, s in (
                ("tag", 19),
                ("title", 38),
                ("sub", 22),
                ("big", 64),
                ("mid", 30),
                ("small", 19),
                ("arm", 120),
                ("cap", 25),
                ("tiny", 15),
            )
        }
        self.fm = ImageFont.truetype(MONO, 20)
        self.cache: dict[tuple[str, int], Image.Image] = {}
        for r in rounds:
            for w in r.walks:
                w.px = project(self.m, w.xy)  # type: ignore[attr-defined]
        if meta["item"]:
            ix, iy, iz = meta["item"]["xyz"]
            self.item_px = project(self.m, np.array([[ix, iy]]), z=iz)[0]
        # timeline: intro, round 0, (sweep, round k)..., outro
        self.segs: list[tuple[str, int, int]] = [("intro", 0, INTRO)]
        t = INTRO
        for i, r in enumerate(rounds):
            if i:
                self.segs.append(("sweep", i, SWEEP))
                t += SWEEP
            n = math.ceil(r.length / SPEED)
            self.segs.append(("play", i, n))
            t += n
        self.segs.append(("outro", len(rounds) - 1, OUTRO))
        self.total = t + OUTRO
        self.bg = self.background()

    # -- static layers
    def background(self) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for c in range(COLS):
            for r in range(ROWS):
                x, y = GX + c * CW, GY + r * CH
                d.rounded_rectangle((x - 2, y - 2, x + TW + 1, y + TH + 1), 6, fill=(17, 20, 27))
        return img

    def tile_img(self, w: Walk, k: int) -> Image.Image:
        """The tile frame at sim frame k, blended between the rendered neighbours."""
        files = w.files
        step = files[1] - files[0] if len(files) > 1 else 1
        pos = (k + 1 - files[0]) / step
        i0 = int(min(max(math.floor(pos), 0), len(files) - 1))
        i1 = min(i0 + 1, len(files) - 1)
        a = min(max(pos - i0, 0.0), 1.0) if i1 != i0 else 0.0
        im0 = self.load(w, files[i0])
        if a < 0.08:
            return im0
        return Image.blend(im0, self.load(w, files[i1]), a)

    def load(self, w: Walk, n: int) -> Image.Image:
        key = (str(w.frames_dir), n)
        im = self.cache.get(key)
        if im is None:
            im = Image.open(w.frames_dir / f"{n:05d}.jpg").convert("RGB")
            if im.size != (TILE_W, TILE_H):
                im = im.resize((TILE_W, TILE_H), Image.BILINEAR)
            im = im.crop(CROP).resize((TW, TH), Image.LANCZOS)
            self.cache[key] = im
            if len(self.cache) > 9000:
                self.cache.clear()
        return im

    # -- per frame
    def where(self, o: int) -> tuple[str, int, float]:
        t = 0
        for kind, i, n in self.segs:
            if o < t + n:
                return kind, i, (o - t) / max(1, n - 1) if kind != "play" else (o - t)
            t += n
        return "outro", len(self.rounds) - 1, 1.0

    def sim_k(self, kind: str, i: int, u: float) -> int:
        r = self.rounds[i]
        if kind == "intro":
            return 0
        if kind == "play":
            return min(int(u * SPEED), r.length - 1)
        return r.length - 1  # sweep (outgoing round) and outro hold the end

    def frame(self, o: int) -> Image.Image:
        kind, i, u = self.where(o)
        img = self.bg.copy()
        self.backdrop(img, o)
        d = ImageDraw.Draw(img, "RGBA")
        sweep_x = None
        if kind == "sweep":
            sweep_x = GX - 60 + (COLS * CW + 120) * smooth(u)
        for n, _ in enumerate(self.rounds[i].walks):
            c, r = n % COLS, n // COLS
            x, y = GX + c * CW, GY + r * CH
            ri, k = i, self.sim_k(kind, i, u)
            if kind == "sweep":
                if x + TW / 2 < sweep_x:  # the update has reached this room: new round, frame 0
                    ri, k = i, 0
                else:
                    ri, k = i - 1, self.rounds[i - 1].length - 1
            w = self.rounds[ri].walks[n]
            tile = self.tile_img(w, k)
            if kind == "intro":
                tile = Image.blend(Image.new("RGB", tile.size, (17, 20, 27)), tile, smooth(u * 1.4))
            img.paste(tile, (x, y))
            self.overlay(d, w, k, x, y, kind, o)
        if sweep_x is not None:
            self.sweep(img, d, sweep_x, i)
        self.panel(d, kind, i, u)
        self.header(d, kind, i, u)
        return img

    def backdrop(self, img: Image.Image, o: int) -> None:
        """Slow moving scan lines behind the grid: the reel is live, never a still."""
        d = ImageDraw.Draw(img, "RGBA")
        off = (o * 1.6) % 48
        for y in range(-48, H + 48, 48):
            yy = int(y + off)
            d.line((0, yy, W, yy), fill=(40, 60, 90, 22))
        x = int((o * 9) % (W + 400)) - 200
        for k in range(14):
            d.line((x + k * 3, 0, x + k * 3 - 260, H), fill=(70, 140, 220, 3 + k))

    def overlay(self, d, w: Walk, k: int, x: int, y: int, kind: str, o: int) -> None:
        pts = w.px[: k + 1]
        stride = 2
        trail = [(x + p[0], y + p[1]) for p in pts[::stride]] + [(x + pts[-1][0], y + pts[-1][1])]
        hit = [c for c in w.contacts if c <= k]
        done = w.clean and k >= w.end
        col = GREEN if done else (CYAN if not hit else AMBER)
        if len(trail) > 1:
            d.line(trail, fill=(*col, 205), width=2, joint="curve")
        for c in hit:
            p = w.px[c]
            d.ellipse((x + p[0] - 3, y + p[1] - 3, x + p[0] + 3, y + p[1] + 3), fill=(*RED, 255))
        if self.meta["item"] is not None:
            ip = self.item_px
            pr = 5 + 2.5 * (1 + math.sin(o * 0.25))
            d.ellipse(
                (x + ip[0] - pr, y + ip[1] - pr, x + ip[0] + pr, y + ip[1] + pr),
                outline=(*AMBER, 230) if not done else (*GREEN, 240),
                width=2,
            )
        # the walker: a dot with its facing
        hx, hy = x + pts[-1][0], y + pts[-1][1]
        if self.meta["item"] is not None and len(w.heading):
            hd = w.heading[min(k, len(w.heading) - 1)]
            q = w.xy[min(k, len(w.xy) - 1)]
            tip = project(
                self.m, np.array([[q[0] + 0.45 * math.cos(hd), q[1] + 0.45 * math.sin(hd)]])
            )
            d.line((hx, hy, x + tip[0][0], y + tip[0][1]), fill=(255, 255, 255, 235), width=2)
        d.ellipse((hx - 3.5, hy - 3.5, hx + 3.5, hy + 3.5), fill=(255, 255, 255, 255))
        # the room's edge: red flash on a contact, a red edge after one, green on a clean reach
        box = (x - 1, y - 1, x + TW, y + TH)
        flash = max((1 - (k - c) / (FLASH * SPEED) for c in hit), default=0.0)
        if kind == "intro":
            return
        if done:
            d.rectangle(box, fill=(*GREEN, 34))
            d.rectangle(box, outline=(*GREEN, 255), width=3)
            cx, cy = x + TW - 15, y + 14
            d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(*GREEN, 255))
            d.line((cx - 5, cy, cx - 1, cy + 4, cx + 5, cy - 4), fill=(10, 30, 15, 255), width=3)
        elif hit:
            d.rectangle(box, outline=(*RED, 150 + round(105 * flash)), width=2 + round(2 * flash))
            if flash > 0:
                d.rectangle(box, fill=(*RED, round(90 * flash)))
            tag = f"{len(hit)}"
            tw = d.textlength(tag, font=self.f["tiny"])
            d.rounded_rectangle((x + TW - tw - 16, y + 5, x + TW - 5, y + 24), 5, fill=(*RED, 230))
            d.text((x + TW - tw - 10, y + 6), tag, font=self.f["tiny"], fill=(255, 255, 255, 255))

    def sweep(self, img: Image.Image, d, sx: float, i: int) -> None:
        y0, y1 = GY - 14, GY + ROWS * CH + 6
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        g = ImageDraw.Draw(glow)
        g.rectangle((sx - 22, y0, sx + 22, y1), fill=(*CYAN, 120))
        glow = glow.filter(ImageFilter.GaussianBlur(18))
        img.paste(glow, (0, 0), glow)
        d.line((sx, y0, sx, y1), fill=(220, 245, 255, 255), width=3)
        r = self.rounds[i]
        msg = f"{self.meta['flow'].upper()}   {self.rounds[i - 1].arm} → {r.arm}"
        bw = max(
            d.textlength(msg, font=self.f["mid"]), d.textlength("+ " + r.change, font=self.f["cap"])
        )
        bx, by = GX + (COLS * CW - bw) / 2 - 30, GY + ROWS * CH / 2 - 62
        d.rounded_rectangle(
            (bx, by, bx + bw + 60, by + 124),
            14,
            fill=(8, 12, 18, 236),
            outline=(*CYAN, 255),
            width=2,
        )
        d.text((bx + 30, by + 18), msg, font=self.f["mid"], fill=(*CYAN, 255))
        d.text((bx + 30, by + 70), "+ " + r.change, font=self.f["cap"], fill=(*INK, 255))

    def counts(self, kind: str, i: int, u: float) -> tuple[int, int, int, int]:
        """(reached clean, contacts, still walking, round shown) for the panel."""
        if kind == "sweep":
            i -= 1
        r = self.rounds[i]
        k = self.sim_k("play" if kind == "sweep" else kind, i, u if kind != "sweep" else 1e9)
        if kind == "intro":
            return 0, 0, len(r.walks), i
        reached = sum(1 for w in r.walks if w.clean and k >= w.end)
        contacts = sum(1 for w in r.walks for c in w.contacts if c <= k)
        moving = sum(1 for w in r.walks if k < w.end)
        return reached, contacts, moving, i

    def panel(self, d, kind: str, i: int, u: float) -> None:
        reached, contacts, moving, ri = self.counts(kind, i, u)
        r = self.rounds[ri]
        n = len(r.walks)
        x = PANEL_X
        d.text((x, GY - 10), r.arm, font=self.f["arm"], fill=(*INK, 255))
        d.text(
            (x + 4, GY + 128),
            f"round {ri + 1} of {len(self.rounds)}",
            font=self.f["small"],
            fill=(*DIM, 255),
        )
        y = GY + 180
        d.text((x, y), self.meta["goal_word"], font=self.f["small"], fill=(*DIM, 255))
        d.text((x, y + 22), f"{reached}", font=self.f["big"], fill=(*GREEN, 255))
        tw = d.textlength(f"{reached}", font=self.f["big"])
        d.text((x + tw + 8, y + 52), f"/ {n}", font=self.f["mid"], fill=(*DIM, 255))
        y += 112
        d.text((x, y), "contacts", font=self.f["small"], fill=(*DIM, 255))
        d.text((x, y + 22), f"{contacts}", font=self.f["big"], fill=(*RED, 255))
        y += 112
        if kind == "intro":
            live = f"{moving} walks about to start"
        elif moving == 0:
            live = "every walk has finished"
        elif kind == "play" and u * SPEED < r.length - 1:
            live = f"{moving} of {n} still walking"
        else:
            live = f"{moving} still walking at the {r.length / FPS:.0f} s cut"
        d.text((x, y), live, font=self.f["small"], fill=(*CYAN, 255))
        if kind == "play":
            d.text(
                (x, y + 26),
                f"sim t {min(u * SPEED, r.length) / FPS:4.1f} s   x{SPEED:.0f} speed",
                font=self.fm,
                fill=(*DIM, 255),
            )
        self.chart(d, kind, i, u, y + 74)

    def chart(self, d, kind: str, i: int, u: float, y0: float) -> None:
        x0 = PANEL_X
        d.text((x0, y0), self.meta["chart_title"], font=self.f["small"], fill=(*DIM, 255))
        totals = [sum(len(w.contacts) for w in r.walks) for r in self.rounds]
        cleans = [sum(w.clean for w in r.walks) for r in self.rounds]
        top = max(totals) or 1
        bh = 200
        base = y0 + 66 + bh
        bw = min(62, (330 - 10 * len(self.rounds)) / len(self.rounds))
        _, live_contacts, _, ri = self.counts(kind, i, u)
        for j, r in enumerate(self.rounds):
            if j > ri:
                break
            v = totals[j] if j < ri or kind in ("outro",) else live_contacts
            if kind == "sweep" and j == ri:
                v = totals[j]
            bx = x0 + j * (bw + 22)
            hgt = bh * v / top
            d.rectangle((bx, base - hgt, bx + bw, base), fill=(*RED, 220))
            d.text(
                (bx + bw / 2, base - hgt - 6),
                f"{v}",
                font=self.f["small"],
                fill=(*INK, 255),
                anchor="mb",
            )
            d.text(
                (bx + bw / 2, base + 8), r.arm, font=self.f["small"], fill=(*INK, 255), anchor="ma"
            )
            done = j < ri or kind in ("sweep", "outro")
            if done:
                d.text(
                    (bx + bw / 2, base + 32),
                    f"{cleans[j]}/{len(r.walks)}",
                    font=self.f["tiny"],
                    fill=(*GREEN, 255),
                    anchor="ma",
                )
        d.line((x0 - 4, base, x0 + 340, base), fill=(*DIM, 160), width=1)
        d.text((x0, base + 56), "green: clean reaches", font=self.f["tiny"], fill=(*DIM, 255))

    def header(self, d, kind: str, i: int, u: float) -> None:
        tw = d.textlength(LABEL, font=self.f["tag"])
        d.rounded_rectangle((GX, 26, GX + tw + 28, 60), 8, fill=(255, 255, 255, 235))
        d.text((GX + 14, 32), LABEL, font=self.f["tag"], fill=(10, 12, 16, 255))
        d.text((GX + tw + 52, 22), self.meta["title"], font=self.f["title"], fill=(*INK, 255))
        d.text((GX, 76), self.meta["sub"], font=self.f["sub"], fill=(*DIM, 255))
        ri = i - 1 if kind == "sweep" else i
        r = self.rounds[ri]
        y = GY + ROWS * CH + 22
        d.text((GX, y), f"{r.arm}", font=self.f["mid"], fill=(*INK, 255))
        cap = r.caption
        if kind == "outro" and u > 0.25:
            a, b = self.rounds[0], self.rounds[-1]
            ca, cb = (sum(len(w.contacts) for w in x.walks) for x in (a, b))
            ga, gb = (sum(w.clean for w in x.walks) for x in (a, b))
            cap = (
                f"{a.arm} → {b.arm}:  contacts {ca} → {cb},  "
                f"clean reaches {ga} → {gb} of {len(b.walks)}"
            )
        d.text((GX + 70, y + 4), cap, font=self.f["cap"], fill=(*INK, 255))
        legend = [
            (CYAN, "walking"),
            (RED, "contact"),
            (AMBER, "walked on after a contact"),
            (GREEN, "reached, nothing touched"),
        ]
        lx = GX
        for col, txt in legend:
            d.rectangle((lx, y + 52, lx + 14, y + 66), fill=(*col, 255))
            d.text((lx + 22, y + 49), txt, font=self.f["small"], fill=(*DIM, 255))
            lx += 44 + d.textlength(txt, font=self.f["small"])


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("montage", "retrieval"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--still", type=int, nargs="*", help="output frame numbers to save as PNGs")
    a = ap.parse_args()
    rounds, meta = montage_rounds() if a.which == "montage" else retrieval_rounds()
    reel = Reel(rounds, meta)
    for r in rounds:
        print(
            r.arm,
            "reached",
            sum(w.clean for w in r.walks),
            "/",
            len(r.walks),
            "contacts",
            sum(len(w.contacts) for w in r.walks),
            "sim frames",
            r.length,
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
    for o in range(reel.total):
        enc.stdin.write(reel.frame(o).tobytes())
    enc.stdin.close()
    enc.wait()


if __name__ == "__main__":
    main()
