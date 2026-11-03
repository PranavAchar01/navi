"""Timeline and finish for the parallel-walks montage (scripts/blender_sim_grid.py renders it).

SIMULATED USER MODEL, NOT A HUMAN STUDY. Sixty find-the-chair walks, seeds 1000-1029 on the
eight-motor belt (top three rows) and the four built diagonals (bottom three), all replayed at
once, one iteration after another. A tile's ring flashes red on each contact and turns green
once the walker is seated without having touched anything. The counter is tallied live from the
same walks and must land on the numbers in bench/results/sim-walk.json (checked in `plan`).

    PYTHONPATH=. .venv/bin/python scripts/build_sim_grid.py plan --grid data/demo-takes/sim/grid
    PYTHONPATH=. .venv/bin/python scripts/build_sim_grid.py finish --grid data/demo-takes/sim/grid \
        --frames DIR --belt-dir DIR --out data/demo-takes/sim/montage-v2.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import composite_sim_walk as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results" / "sim-walk.json"
FPS = 30
COLS, ROWS = 10, 6
SEEDS = range(1000, 1030)
ARMS = ["I0", "I3", "I5", "I6"]
CAPTIONS = {
    "I0": "the glasses as configured: only what the camera sees right now",
    "I3": "+ the map's side and back sectors, and a held bearing on the cue",
    "I5": "+ the guide routed by the app's navigator",
    "I6": "+ the second-nearest object cued too, not just the nearest",
}
HERO_AT = (2, 4)  # row, col: the eight-motor block's lowest row, just left of centre
OPEN_HOLD, PULL = 75, 180  # frames close on the hero, then the pull back
HOLD_AFTER = 36  # frames every phase holds its outcomes before the next
CAP_S = 11.0
HERO_DELAY = 96  # in the last phase the hero starts late, as the camera pushes back in
PUSH = 100
END_HOLD = 48
GREEN = (0.25, 1.0, 0.45)
RED = (1.0, 0.16, 0.12)
FONT = C.FONT


def events(tr: dict) -> dict:
    """Contact frames, the frame the walk ends (seated, or its last frame), and if it was clean."""
    col = tr["collisions"]
    contacts = [k for k in range(len(col)) if col[k] > (col[k - 1] if k else 0)]
    sat = next((k for k, s in enumerate(tr["sit"]) if s >= 0.999), None)
    end = sat if sat is not None else tr["frames"] - 1
    return {
        "contacts": contacts,
        "n_contacts": col[-1],
        "end": end,
        "sat": sat is not None,
        "clean": sat is not None and col[-1] == 0,
    }


def ring_frames(ev: dict, start: int, nxt: int) -> list:
    """[frame, [r, g, b, strength]] for one tile and phase, every frame start..nxt-1."""
    out = []
    for f in range(start, nxt):
        k = f - start
        red = sum(math.exp(-(k - c) / 7.0) for c in ev["contacts"] if c <= k)
        col, a = RED, min(red, 1.0)
        if k >= ev["end"]:
            u = min((k - ev["end"]) / 10.0, 1.0)
            if ev["clean"]:
                col, a = GREEN, max(0.85 * u, 0.0)
            else:
                a = max(a, 0.3 * u)
        a *= min(1.0, (nxt - 1 - f) / 10.0)
        out.append([f, [*col, round(a, 3)]])
    return out


def plan(grid: Path) -> None:
    rows = json.loads(RESULTS.read_text())["rows"]
    ledger = {(r["arm"], r["layout"], r["seed"]): r for r in rows if r["scenario"] == "find_chair"}
    tr = {}
    for a in ARMS:
        for lay in ("eight", "diag4"):
            for s in SEEDS:
                tr[(a, lay, s)] = json.loads((grid / "tracks" / f"{a}-{lay}-{s}.json").read_text())
    ev = {k: events(v) for k, v in tr.items()}
    for (a, lay, s), e in ev.items():  # the replay must be the scored walk
        r = ledger[(a, lay, s)]
        assert e["n_contacts"] == r["collisions"], (a, lay, s, e["n_contacts"], r["collisions"])
        assert e["clean"] == (r["success"] and r["collisions"] == 0), (a, lay, s)

    # the hero: an eight-motor seed that touches at I0 and sits clean at I6, quickest first
    cands = [
        s
        for s in SEEDS
        if ledger[("I0", "eight", s)]["collisions"]
        and ledger[("I0", "eight", s)]["success"]
        and ledger[("I6", "eight", s)]["success"]
        and not ledger[("I6", "eight", s)]["collisions"]
    ]
    hero_seed = min(cands, key=lambda s: (ledger[("I6", "eight", s)]["time_s"], s))
    order = {"eight": [s for s in SEEDS if s != hero_seed], "diag4": list(SEEDS)}
    tiles = []
    for lay, r0 in (("eight", 0), ("diag4", 3)):
        seq = iter(order[lay])
        for r in range(r0, r0 + 3):
            for c in range(COLS):
                s = hero_seed if lay == "eight" and (r, c) == HERO_AT else next(seq)
                tiles.append({"id": f"{lay}-{s}", "layout": lay, "seed": s, "col": c, "row": r})
    hero = f"eight-{hero_seed}"

    cap = int(CAP_S * FPS)
    phases, start = [], 1
    for i, a in enumerate(ARMS):
        longest = max(tr[(a, t["layout"], t["seed"])]["frames"] for t in tiles)
        if i == len(ARMS) - 1:
            n = HERO_DELAY + tr[(a, "eight", hero_seed)]["frames"] + END_HOLD
            n = max(n, min(longest, cap) + HOLD_AFTER)
        else:
            n = min(longest, cap) + HOLD_AFTER
        if i == 0:
            n = max(n, OPEN_HOLD + PULL + 60)
        phases.append({"arm": a, "start": start, "len": n})
        start += n
    frames = start - 1
    phases[-1]["delay"] = {hero: HERO_DELAY}

    rings: dict = {t["id"]: [] for t in tiles}
    tally = []  # per phase, per tile: contact frames shown, end frame, clean
    for ph in phases:
        nxt = ph["start"] + ph["len"]
        cut = ph["start"] + cap
        per = {}
        for t in tiles:
            e = dict(ev[(ph["arm"], t["layout"], t["seed"])])
            d = ph.get("delay", {}).get(t["id"], 0)
            s0 = ph["start"] + d
            if s0 + e["end"] >= min(nxt, cut + d):  # still going when the replay stops
                stop = min(nxt, cut + d) - s0 - 1
                # its later contacts are real, so the tally takes them at the cut
                e = {
                    **e,
                    "end": stop,
                    "clean": False,
                    "contacts": [min(c, stop) for c in e["contacts"]],
                }
                print(f"cut: {ph['arm']} {t['id']} ends at {e['end']} frames", file=sys.stderr)
            rings[t["id"]].append(ring_frames(e, s0, nxt) if d == 0 else _late(e, ph, s0, nxt))
            per[t["id"]] = {
                "contacts": [s0 + c for c in e["contacts"] if s0 + c < nxt],
                "end": s0 + e["end"],
                "clean": e["clean"],
            }
        shown_clean = sum(p["clean"] for p in per.values())
        shown_contacts = sum(len(p["contacts"]) for p in per.values())
        led = [ledger[(ph["arm"], t["layout"], t["seed"])] for t in tiles]
        led_clean = sum(1 for r in led if r["success"] and r["collisions"] == 0)
        led_contacts = sum(r["collisions"] for r in led)
        print(
            f"{ph['arm']}: frames {ph['start']}-{nxt - 1}, shown {shown_clean}/60 clean, "
            f"{shown_contacts} contacts; ledger {led_clean}/60, {led_contacts}",
            file=sys.stderr,
        )
        assert (shown_clean, shown_contacts) == (led_clean, led_contacts), ph["arm"]
        ph["caption"] = CAPTIONS[ph["arm"]]
        ph["clean"], ph["contacts"] = led_clean, led_contacts
        tally.append(per)

    last = phases[-1]
    camera = [
        [1, "close"],
        [OPEN_HOLD, "close"],
        [OPEN_HOLD + PULL, "wide"],
        [last["start"] + HERO_DELAY - 20, "wide"],
        [last["start"] + HERO_DELAY - 20 + PUSH, "close"],
        [frames, "close"],
    ]
    out = {
        "frames": frames,
        "cols": COLS,
        "rows": ROWS,
        "hero": hero,
        "template": {"eight": f"I0-eight-{hero_seed}", "diag4": "I0-diag4-1000"},
        "tiles": tiles,
        "phases": phases,
        "camera": camera,
        "rings": rings,
        "tally": tally,
    }
    (grid / "timeline.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"hero {hero}, {frames} frames ({frames / FPS:.1f} s)", file=sys.stderr)


def _late(e, ph, s0, nxt):
    """Rings for a tile that starts late: dark until it starts."""
    rows = [[f, [*RED, 0.0]] for f in range(ph["start"], s0)]
    return rows + ring_frames(e, s0, nxt)


# ---------------------------------------------------------------- finish


def font(size, bold=False):
    f = ImageFont.truetype(FONT, size)
    if bold:
        try:
            f.set_variation_by_name("Semibold")
        except (OSError, ValueError):
            pass
    return f


def card(size, alpha=236):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1), radius=14, fill=(245, 245, 247, alpha)
    )
    return img


def counter(ph: dict, per: dict, f: int) -> Image.Image:
    done = sum(1 for p in per.values() if p["clean"] and p["end"] <= f)
    touches = sum(1 for p in per.values() for c in p["contacts"] if c <= f)
    img = card((468, 142))
    d = ImageDraw.Draw(img)
    d.text(
        (20, 14),
        f"{len(per)} parallel simulated walks  ·  iteration {ph['arm']}",
        font=font(16),
        fill=(60, 64, 72),
    )
    d.ellipse((22, 58, 36, 72), fill=(64, 200, 110))
    d.text((48, 44), f"{done} / {len(per)}", font=font(34, True), fill=(30, 32, 36))
    tw = d.textlength(f"{done} / {len(per)}", font=font(34, True))
    d.text(
        (56 + tw, 58),
        f"seated, no contact  ({100 * done / len(per):.0f}%)",
        font=font(16),
        fill=(60, 64, 72),
    )
    d.ellipse((22, 104, 36, 118), fill=(232, 72, 60))
    d.text((48, 96), f"{touches / len(per):.2f}", font=font(24, True), fill=(30, 32, 36))
    tw = d.textlength(f"{touches / len(per):.2f}", font=font(24, True))
    d.text((56 + tw, 102), "contacts per walk", font=font(16), fill=(60, 64, 72))
    return img


def caption(ph: dict) -> Image.Image:
    f1, f2 = font(30, True), font(22)
    w = int(max(ImageDraw.Draw(Image.new("L", (1, 1))).textlength(ph["caption"], font=f2), 200))
    img = card((w + 130, 70))
    d = ImageDraw.Draw(img)
    d.text((22, 16), ph["arm"], font=f1, fill=(30, 32, 36))
    d.text((100, 22), ph["caption"], font=f2, fill=(60, 64, 72))
    return img


def project(m, x, y, z=0.0):
    p = np.array(m) @ np.array([x, y, z, 1.0])
    return (p[0] / p[3] * 0.5 + 0.5) * 1920, (0.5 - p[1] / p[3] * 0.5) * 1080


def finish(grid: Path, frames: Path, belt_dir: Path | None, out: Path, only: int | None) -> None:
    tl = json.loads((grid / "timeline.json").read_text())
    cams = {}
    for p in sorted(frames.glob("*cams*.json")):
        meta = json.loads(p.read_text())
        cams.update({int(k): v for k, v in meta["ndc_from_world"].items()})
        boxes = meta["boxes"]
    hero_tr = json.loads((grid / "tracks" / f"I0-{tl['hero']}.json").read_text())
    comp = frames / "comp"
    comp.mkdir(exist_ok=True)
    starts = [p["start"] for p in tl["phases"]]
    wide_from = tl["camera"][1][0] + 0.7 * (tl["camera"][2][0] - tl["camera"][1][0])
    push_from = tl["camera"][3][0]
    eight_rows = [b for t, b in boxes.items() if t.startswith("eight")]
    four_rows = [b for t, b in boxes.items() if t.startswith("diag4")]
    for f in [only] if only else range(1, tl["frames"] + 1):
        src = frames / f"{f:05d}.png"
        img = Image.open(src).convert("RGBA")
        pi = max(i for i, s in enumerate(starts) if s <= f)
        ph, per = tl["phases"][pi], tl["tally"][pi]
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        # the counter and the row labels while the field is in view
        vis = min(max((f - wide_from) / 20, 0), 1) * min(max((push_from + 40 - f) / 20, 0), 1)
        if pi > 0:
            vis = min(max((push_from + 40 - f) / 20, 0), 1)
        if vis > 0:
            c = counter(ph, per, f)
            ov.alpha_composite(_fade(c, vis), (28, 28))
            m = cams[f]
            for rows, text in ((eight_rows, "8 motors"), (four_rows, "4 motors")):
                x0 = min(b[0] for b in rows)
                ys = [b[2] for b in rows] + [b[3] for b in rows]
                px, py = project(m, x0 - 0.6, (min(ys) + max(ys)) / 2)
                lab = Image.new("RGBA", (160, 40), (0, 0, 0, 0))
                ImageDraw.Draw(lab).text((0, 0), text, font=font(22, True), fill=(235, 238, 242))
                if 0 < px < 1920 and 0 < py < 1080:
                    ov.alpha_composite(_fade(lab, vis), (max(int(px) - 150, 8), int(py) - 14))
        # the caption for each iteration, for its first three seconds
        k = f - ph["start"]
        if pi > 0 or f > wide_from:
            k0 = 0 if pi > 0 else int(wide_from) - ph["start"]
            a = min(max((k - k0) / 10, 0), 1) * min(max((k0 + 100 - k) / 12, 0), 1)
            if a > 0:
                cap = caption(ph)
                ov.alpha_composite(_fade(cap, a), (28, 1080 - cap.height - 56))
        # the belt panel over the opening, gone as the camera leaves the hero room
        if belt_dir is not None and f < wide_from:
            bp = belt_dir / f"{f - 1:05d}.png"
            if bp.exists() and f - 1 < hero_tr["frames"]:
                levels = hero_tr["felt"][f - 1]
                bc = C.belt_card(C.crop_belt(Image.open(bp)), _cells8(levels), False)
                a = min(max((wide_from - 30 - f) / 30, 0), 1)
                ov.alpha_composite(_fade(bc, a), (1920 - bc.width - 24, 24))
        d = ImageDraw.Draw(ov)
        lf = font(13)
        tw = d.textlength(C.LABEL, font=lf)
        d.text((1920 - tw - 22, 1080 - 30), C.LABEL, font=lf, fill=(235, 238, 242, 190))
        img = Image.alpha_composite(img, ov)
        # a quick dip between iterations, as every walker goes back to the start
        dip = min((abs(f - s + 0.5) for s in starts[1:]), default=99)
        if dip < 8:
            k = 1 - 0.8 * (1 - dip / 8)
            img = Image.eval(img.convert("RGB"), lambda v, k=k: int(v * k)).convert("RGBA")
        img.convert("RGB").save(out if only else comp / f"{f:05d}.png")
    if only:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(comp / "%05d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )


def _cells8(felt: list[float]) -> list[float]:
    """Belt cells clockwise from ahead, as composite_sim_walk.felt8 reads them."""
    if len(felt) == 8:
        return list(felt)
    out = [0.0] * 8
    for k, v in enumerate(felt):
        out[2 * k + 1] = v
    return out


def _fade(img: Image.Image, a: float) -> Image.Image:
    if a >= 1:
        return img
    out = img.copy()
    out.putalpha(Image.eval(img.getchannel("A"), lambda v: int(v * a)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("plan", "finish"))
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--frames", type=Path)
    ap.add_argument("--belt-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", type=int, help="composite this one frame to --out")
    a = ap.parse_args()
    if a.step == "plan":
        plan(a.grid)
    else:
        finish(a.grid, a.frames, a.belt_dir, a.out, a.only)


if __name__ == "__main__":
    main()
