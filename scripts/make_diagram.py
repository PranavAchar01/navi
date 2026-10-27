"""Draw docs/architecture.svg from code (no image model). Run: python scripts/make_diagram.py"""

from html import escape
from pathlib import Path

INK, PAPER, LINE, BLUE, MUTE = "#161616", "#f6f6f4", "#cfcfca", "#1a4fff", "#666660"
W, H = 1100, 560
out: list[str] = []


def box(x, y, w, h, title, lines=(), accent=False):
    stroke = BLUE if accent else INK
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#fff" stroke="{stroke}" '
        f'stroke-width="{2 if accent else 1.2}"/>'
    )
    out.append(f'<text x="{x + 14}" y="{y + 26}" class="t">{escape(title)}</text>')
    for i, line in enumerate(lines):
        out.append(f'<text x="{x + 14}" y="{y + 48 + 18 * i}" class="s">{escape(line)}</text>')


def arrow(x1, y1, x2, y2, label="", dy=-6):
    out.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" stroke-width="1.4" '
        'marker-end="url(#a)"/>'
    )
    if label:
        out.append(
            f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 + dy}" class="s" text-anchor="middle">'
            f"{escape(label)}</text>"
        )


box(
    30,
    40,
    200,
    110,
    "Phone browser",
    ["camera + DeviceMotion", "later: glasses via the", "iOS Wearables DAT app"],
)
box(
    30,
    200,
    200,
    110,
    "Web client",
    ["live view + proximity", "ring and belt cells", "3D map, pins, fly-to"],
    True,
)
box(
    30,
    360,
    200,
    150,
    "Cues",
    ["audio tones and speech", "haptic pulses", "off > 2 m, slow 2-1 m,", "fast 1-0.5 m, solid"],
)

box(
    330,
    40,
    330,
    200,
    "Mac mini server (Python)",
    [
        "WebSocket ingest: JPEG frames + pose/motion",
        "Safety loop, 10-20 Hz:",
        "  metric depth + YOLO26 -> 3 live sectors",
        "  5 more sectors from the map",
        "Fall check-in + shaking notice (motion)",
        "Query: 'where are my keys' -> pin + sentence",
    ],
    True,
)
box(
    330,
    290,
    330,
    220,
    "Memory loop (own process)",
    [
        "keyframes -> MapAnything windows (8 views)",
        "Sim3 stitch on overlap pixels, 5 cm voxels",
        "scale hint re-anchors live depth",
        "object finder watch-list -> pins with",
        "label, position, time, thumbnail, landmarks",
        "saved to sessions/<name>.npz",
    ],
)

box(
    760,
    40,
    310,
    130,
    "Language step (one interface)",
    [
        "template  |  openai  |  meta (Muse Spark)",
        "turns facts into one short sentence",
        "key from env or Keychain, never a file",
    ],
    False,
)
box(
    760,
    210,
    310,
    130,
    "Small-object finder (one interface)",
    [
        "yoloe (local, default)  |  sam31 (Meta)",
        "one switch in metasight.toml",
        "language = 'meta', finder = 'sam31'",
    ],
    False,
)
box(
    760,
    380,
    310,
    130,
    "Models on the Mac",
    [
        "Depth Anything V2 metric (fp16, MPS)",
        "YOLO26n, YOLOE-26s-seg",
        "MapAnything (Apache-2.0 weights)",
    ],
)

arrow(130, 150, 130, 200, "frames, motion")
arrow(230, 255, 330, 120, "ws", -8)
arrow(330, 160, 230, 280, "overlay, map, answers", 14)
arrow(130, 310, 130, 360, "")
arrow(495, 240, 495, 290, "keyframes / map")
arrow(660, 100, 760, 100, "facts")
arrow(660, 400, 760, 260, "")
arrow(660, 450, 760, 445, "")

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
    f'<style>.t{{font:600 15px "Averia Serif Libre",Georgia,serif;fill:{INK}}}'
    f'.s{{font:12.5px "Atkinson Hyperlegible",Helvetica,Arial,sans-serif;fill:{MUTE}}}</style>'
    f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker></defs>'
    f'<rect width="{W}" height="{H}" fill="{PAPER}"/>' + "".join(out) + "</svg>"
)
Path(__file__).resolve().parent.parent.joinpath("docs", "architecture.svg").write_text(svg)
