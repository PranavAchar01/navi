"""The Navi vs plain YOLO card for the demo cut, drawn from bench/results/navi-vs-yolo.json.

    .venv/bin/python bench/compare/navi_vs_yolo_card.py

1920x1080 PNG in the rough cut's white glass style (fonts, colours and panels come from
showcase/rough/build_rough.py). Every number on it is read from the results file.
Writes data/demo-takes/bench/navi-vs-yolo.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "showcase" / "rough"))

from build_rough import ACCENT, BG, INK, LINE, MUTED, H, W, font, glass, pill  # noqa: E402

RES = ROOT / "bench" / "results" / "navi-vs-yolo.json"
OUT = ROOT / "data" / "demo-takes" / "bench" / "navi-vs-yolo.png"
GREY_BAR = (176, 176, 186)

ROWS = [
    ("recall, wall/door", "Walls and doors", "no YOLO class"),
    ("recall, chair/bed", "Chairs and the bed", "YOLO has a class"),
    ("recall, desk/table", "Desks and tables", "closest YOLO class: dining table"),
]


def cell(d: ImageDraw.ImageDraw, x: int, y: int, r: dict, colour: tuple) -> None:
    d.text((x, y), f"{r['hit']} of {r['n']}", font=font(52, "Semibold"), fill=INK)
    bw = 300
    d.rounded_rectangle((x, y + 74, x + bw, y + 86), 6, fill=LINE)
    if r["hit"]:
        d.rounded_rectangle((x, y + 74, x + max(12, int(bw * r["rate"])), y + 86), 6, fill=colour)


def main() -> None:
    res = json.loads(RES.read_text())
    summ = {s["metric"]: s for s in res["summary"]}
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((140, 92), "Navi vs plain YOLO", font=font(64, "Semibold"), fill=INK)
    d.text(
        (140, 178),
        "Did it warn when something was within 1.3 m ahead? Frames from the bedroom take.",
        font=font(30),
        fill=MUTED,
    )

    box = (140, 250, W - 140, 860)
    glass(img, box, 32)
    d = ImageDraw.Draw(img)
    x_kind, x_navi, x_yolo = 210, 920, 1340
    pill(d, (x_navi, 290), "Navi", 26, fg=(255, 255, 255), bg=ACCENT)
    pill(d, (x_yolo, 290), "YOLO alone", 26)
    d.text((x_navi, 340), "depth + floor + path", font=font(22), fill=MUTED)
    d.text((x_yolo, 340), "same YOLO26n, distance from box size", font=font(22), fill=MUTED)

    y = 400
    for key, name, sub in ROWS:
        s = summ[key]
        d.line((x_kind, y - 18, box[2] - 70, y - 18), fill=LINE, width=2)
        d.text((x_kind, y), name, font=font(40, "Medium"), fill=INK)
        d.text((x_kind, y + 54), f"{sub}  ·  {s['n']} frames", font=font(24), fill=MUTED)
        cell(d, x_navi, y, s["navi"], ACCENT)
        cell(d, x_yolo, y, s["yolo"], GREY_BAR)
        y += 120

    c = summ["compute per frame, ms"]
    d.line((x_kind, y - 18, box[2] - 70, y - 18), fill=LINE, width=2)
    d.text((x_kind, y), "Compute per frame", font=font(40, "Medium"), fill=INK)
    d.text((x_kind, y + 54), "median on the Mac mini", font=font(24), fill=MUTED)
    d.text((x_navi, y), f"{c['navi']['p50']:.1f} ms", font=font(52, "Semibold"), fill=INK)
    d.text((x_yolo, y), f"{c['yolo']['p50']:.1f} ms", font=font(52, "Semibold"), fill=INK)

    on = res["navi_cue_on_share_all_replayed_frames"]
    data = res["data"]
    foot = [
        f"Navi buzzed on {round(100 * on['rate'])}% of all {on['n']} frames in this small room, "
        f"so false alarms are not rated: only {res['recall']['navi']['warned_on_clear']['n']} "
        "labelled frames had a clear path.",
        f"One 88 s take, {data['labelled']} frames labelled by eye ({data['unsure_dropped']} "
        "unsure dropped). Both timed at background priority.",
    ]
    for i, line in enumerate(foot):
        d.text((140, 900 + 44 * i), line, font=font(26), fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
