"""Draw docs/architecture.svg from code (no image model). Run: python scripts/make_diagram.py

Every latency on the diagram is a measured median with its source in docs/ARCHITECTURE.md.
Colours are CSS classes, so the same file reads on a light or a dark page.
"""

from html import escape
from pathlib import Path

W, H = 1100, 600
BW, BH, GAP, X0 = 190, 158, 25, 25  # box width and height, gap, left margin
out: list[str] = []


def box(x, y, w, title, model, lines, ms, accent=False):
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="10" '
        f'class="{"bx ac" if accent else "bx"}"/>'
    )
    out.append(f'<text x="{x + 14}" y="{y + 26}" class="t">{escape(title)}</text>')
    out.append(f'<text x="{x + 14}" y="{y + 46}" class="m">{escape(model)}</text>')
    for i, line in enumerate(lines):
        out.append(f'<text x="{x + 14}" y="{y + 66 + 17 * i}" class="s">{escape(line)}</text>')
    out.append(f'<text x="{x + 14}" y="{y + BH - 14}" class="n">{escape(ms)}</text>')


def arrow(x1, y1, x2, y2, label="", dx=0, dy=-7):
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln" marker-end="url(#a)"/>')
    if label:
        out.append(
            f'<text x="{(x1 + x2) / 2 + dx}" y="{(y1 + y2) / 2 + dy}" class="s" '
            f'text-anchor="middle">{escape(label)}</text>'
        )


def col(i):
    return X0 + i * (BW + GAP)


R1, R2 = 60, 330  # row tops
out.append(f'<text x="{X0}" y="{R1 - 16}" class="lane">ALWAYS RUNNING, EVERY FRAME</text>')
out.append(f'<text x="{X0}" y="{R2 - 16}" class="lane">WHEN YOU ASK</text>')

box(
    col(0),
    R1,
    BW,
    "1  Capture",
    "Ray-Ban Meta or phone",
    ["camera frames, motion,", "microphone audio", "glasses: Wearables DAT"],
    "17 frames a second",
)
box(
    col(1),
    R1,
    BW,
    "2  Transport",
    "WebSocket to the Mac mini",
    ["one socket carries", "frames, motion and", "16 kHz mic audio"],
    "inside the 186 ms below",
)
box(
    col(2),
    R1,
    BW,
    "3  Safety loop",
    "Depth Anything + YOLO26n",
    ["floor, obstacles and", "8 direction sectors", "on the Mac's GPU"],
    "frame to cue 186 ms",
    True,
)
box(
    col(3),
    R1,
    BW,
    "4  Map and memory",
    "MapAnything",
    ["3D map of the house,", "pins: what, where,", "when, what was near"],
    "map window 19.6 s",
)
box(
    col(4),
    R1,
    BW,
    "5  Finder",
    "YOLOE + Muse SAM 3.1",
    ["names what is in the", "frame; SAM 3.1 checks", "small things (keys)"],
    "SAM 3.1: 0.6 to 1.5 s",
)

AW = 2 * BW + GAP  # the answer box spans two columns
box(
    col(0),
    R2,
    AW,
    "8  Answer",
    "speech, lens, haptics",
    [
        "one sentence read out in the glasses or phone,",
        "the pin lit up on the screen, and the belt",
        "buzzes toward obstacles (server to motor 38-56 ms)",
    ],
    "last word to answer 871 ms",
    True,
)
box(
    col(2) + 50,
    R2,
    BW + 40,
    "7  Muse Spark 1.3",
    "phrases from the facts",
    ["starts on the first", "partial words; Glimmer", "30B on the Mac if offline"],
    "first word 0.7 to 1.0 s",
    True,
)
box(
    col(4),
    R2,
    BW,
    "6  Question",
    "Muse Voice Transcribe",
    ["mic stays open; the", "model hears where the", "question ends"],
    "turn end 164 ms",
    True,
)

mid = R1 + BH / 2
for i in range(4):
    arrow(col(i) + BW, mid, col(i + 1), mid)
arrow(col(2) + BW / 2, R1 + BH, col(2) + BW / 2 - 60, R2, "obstacle cues", dx=-62, dy=4)
arrow(col(3) + BW / 2 + 30, R1 + BH, col(3) + BW / 2 - 10, R2, "pins and map", dx=44, dy=4)
arrow(col(4), R2 + BH / 2, col(2) + 50 + BW + 40, R2 + BH / 2, "words")
arrow(col(2) + 50, R2 + BH / 2, col(0) + AW, R2 + BH / 2, "")

out.append(
    f'<text x="{X0}" y="{H - 34}" class="s">Measured on the Mac mini (M5 Pro): medians, '
    "or the range seen. Boxes in blue are on the answer or cue path; "
    "the map and finder run beside them.</text>"
)
out.append(
    f'<text x="{X0}" y="{H - 14}" class="s">Sources for every number: docs/ARCHITECTURE.md.</text>'
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
