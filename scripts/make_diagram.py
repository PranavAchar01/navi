"""Draw docs/architecture.svg from code (no image model). Run: python scripts/make_diagram.py

Every latency on the diagram is a measured median with its source in docs/ARCHITECTURE.md.
Colours are CSS classes, so the same file reads on a light or a dark page.
"""

from html import escape
from pathlib import Path

W, H = 1100, 690
LX, X0, BW, GAP, LH = 25, 185, 205, 26, 104  # lane label x, first box x, box width, gap, lane
BH = 86
out: list[str] = []


def box(x, y, title, model, line, ms, accent=False):
    out.append(
        f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="10" '
        f'class="{"bx ac" if accent else "bx"}"/>'
    )
    out.append(f'<text x="{x + 12}" y="{y + 22}" class="t">{escape(title)}</text>')
    out.append(f'<text x="{x + 12}" y="{y + 40}" class="m">{escape(model)}</text>')
    out.append(f'<text x="{x + 12}" y="{y + 57}" class="s">{escape(line)}</text>')
    out.append(f'<text x="{x + 12}" y="{y + BH - 11}" class="n">{escape(ms)}</text>')


def arrow(x1, y1, x2, y2, label="", dx=0, dy=-7):
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln" marker-end="url(#a)"/>')
    if label:
        out.append(
            f'<text x="{(x1 + x2) / 2 + dx}" y="{(y1 + y2) / 2 + dy}" class="s">'
            f"{escape(label)}</text>"
        )


def col(i):
    return X0 + i * (BW + GAP)


def lane(k, name, when, boxes, accent=False, chain=99):
    y = 34 + k * (LH + 22)
    out.append(f'<text x="{LX}" y="{y + 34}" class="t">{escape(name)}</text>')
    for i, w in enumerate(when):
        out.append(f'<text x="{LX}" y="{y + 54 + 16 * i}" class="s">{escape(w)}</text>')
    for i, b in enumerate(boxes):
        box(col(i), y, *b, accent=accent)
        if 0 < i < chain:
            arrow(col(i - 1) + BW, y + BH / 2, col(i), y + BH / 2)
    return y


y1 = lane(
    0,
    "1  Reflex",
    ["every frame", "under 250 ms", "no language model"],
    [
        (
            "Live depth",
            "Depth Anything + YOLO26n",
            "floor, steps, 8 sectors",
            "frame to cue 186 ms",
        ),
        (
            "Map checks",
            "scale, out of view, new",
            "the map feeds the reflex",
            "distance error 34% to 16%",
        ),
        ("STOP buzz", "belt stop layer", "beats everything else", "server to motor 38-56 ms"),
    ],
    True,
)
y2 = lane(
    1,
    "2  See, remember",
    ["every few seconds", "in the background"],
    [
        (
            "Frame picker",
            "sharpest frame per slot",
            "1.5 a second at home",
            "next: novelty, change",
        ),
        ("MapAnything", "open, Meta", "3D map with real scale", "19.6 s a window"),
        ("DINOv2", "open, Meta", "knows a place seen before", "revisit 1.15 m to 0.46 m"),
        ("Memory bank", "pins + SAM 3.1 check", "what, where, when, near", "SAM 3.1: 0.6 to 1.5 s"),
    ],
)
y3 = lane(
    2,
    "3  Talk",
    ["when you ask", "about 0.9 s"],
    [
        ("Transcribe", "Muse Voice Transcribe", "mic open, finds turn end", "turn end 164 ms"),
        (
            "Muse Spark 1.3",
            "memory-bank questions",
            "tools for people, changes",
            "first word 0.7-1.0 s",
        ),
        ("Answer", "voice in the glasses", "pin lit, belt points", "last word to answer 871 ms"),
        ("Offline", "Muse Glimmer, open", "stands in for Spark", "one sentence 1.2-2.6 s"),
    ],
    True,
    chain=3,
)
y4 = lane(
    3,
    "Navigate",
    ["three layers", "the reflex wins"],
    [
        ("Goal", "from Muse", '"take me to my keys"', "a pin"),
        ("Path planner", "nav.py, no model", "through doorways", "milliseconds; wiring"),
        ("GUIDE pulses", "belt guide layer", "soft turn-by-turn", "STOP overrides it"),
    ],
)
y5 = lane(
    4,
    "4  Deep",
    ["every few minutes", "nobody waiting"],
    [
        ("Map review", "Muse Wonder", "which changes matter", "9-15 s, about $0.15/hour"),
        ("Plans", "not built", '"get ready to leave"', "goals for Navigate"),
        ("Nudges", "not built", '"closet: 3 days unseen"', "from last-seen times"),
    ],
)
cx = col(1) + BW / 2
arrow(cx, y2, cx, y1 + BH, "map scale", dx=8, dy=4)
mx = col(3) + BW / 2
arrow(mx, y2 + BH, col(1) + BW / 2 + 40, y3, "pins", dx=10, dy=4)
arrow(col(1) + BW / 2, y5, col(0) + BW / 2 + 40, y4 + BH, "goals", dx=12, dy=4)

out.append(
    f'<text x="{LX}" y="{H - 34}" class="s">Measured on the Mac mini (M5 Pro): medians or the '
    "range seen. Blue boxes are on the cue or answer path; the others run beside them. "
    "Map review marks new obstacles for the reflex.</text>"
)
out.append(
    f'<text x="{LX}" y="{H - 14}" class="s">Sources for every number: docs/ARCHITECTURE.md.'
    " STOP never waits on an AI model.</text>"
)

STYLE = """
:root{--ink:#161616;--paper:#f6f6f4;--card:#fff;--mute:#5d5d57;--blue:#1a4fff}
@media (prefers-color-scheme:dark){:root{--ink:#ecece8;--paper:#151515;--card:#1f1f1f;
--mute:#a8a8a0;--blue:#7d9cff}}
.bx{fill:var(--card);stroke:var(--ink);stroke-width:1.2}.ac{stroke:var(--blue);stroke-width:2}
.t{font:600 15px "Averia Serif Libre",Georgia,serif;fill:var(--ink)}
.m{font:600 12.5px "Atkinson Hyperlegible",Helvetica,Arial,sans-serif;fill:var(--ink)}
.s{font:12.5px "Atkinson Hyperlegible",Helvetica,Arial,sans-serif;fill:var(--mute)}
.n{font:700 13px "Atkinson Hyperlegible",Helvetica,Arial,sans-serif;fill:var(--blue)}
.lane{font:700 11px "Atkinson Hyperlegible",Helvetica,Arial,sans-serif;fill:var(--mute);
letter-spacing:.12em}
.ln{stroke:var(--ink);stroke-width:1.4}.hd{fill:var(--ink)}.bg{fill:var(--paper)}
"""
svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
    f"<style>{STYLE.strip()}</style>"
    '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto"><path d="M0 0L10 5L0 10z" class="hd"/></marker></defs>'
    f'<rect width="{W}" height="{H}" class="bg"/>' + "".join(out) + "</svg>"
)
Path(__file__).resolve().parent.parent.joinpath("docs", "architecture.svg").write_text(svg)
