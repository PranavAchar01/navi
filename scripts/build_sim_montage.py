"""The montage of the simulated walker improving, cut from the Blender iteration clips.

SIMULATED USER MODEL, NOT A HUMAN STUDY. An opening card, then per iteration a short card with
that arm's numbers over all its seeds (bench/results/sim-walk.json, scripts/run_sim_walk.py;
"reached" means reached with no contact) and the clip of one seed rendered by
scripts/composite_sim_walk.py, with its binaural sound.

    PYTHONPATH=. .venv/bin/python scripts/build_sim_montage.py --scenario find_chair \
        --layout eight --clip I0=PATH.mp4@12 --clip I5=PATH.mp4 --out montage.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results" / "sim-walk.json"
FONT = "/System/Library/Fonts/SFNS.ttf"
W, H, FPS = 1920, 1080, 30
CARD_S, OPEN_S = 1.6, 2.4
LABEL = "SIMULATED USER MODEL, NOT A HUMAN STUDY"


def arm_numbers(res: dict, arm: str, scenario: str, layout: str) -> dict:
    rows = [
        r
        for r in res["rows"]
        if r["arm"] == arm and r["scenario"] == scenario and r["layout"] == layout
    ]
    clean = [bool(r["success"] and r["collisions"] == 0) for r in rows]
    return {
        "walks": len(rows),
        "reached": int(sum(clean)),
        "collisions": float(np.mean([r["collisions"] for r in rows])),
        "min_clear_m": float(np.mean([r["min_clear_m"] for r in rows])),
    }


def card(lines: list[tuple[str, int, tuple]], out: Path) -> None:
    img = Image.new("RGB", (W, H), (22, 24, 28))
    d = ImageDraw.Draw(img)
    y = H // 2 - sum(s + 22 for _, s, _ in lines) // 2
    for text, size, col in lines:
        f = ImageFont.truetype(FONT, size)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=col)
        y += size + 22
    f = ImageFont.truetype(FONT, 18)
    d.text((W - d.textlength(LABEL, font=f) - 36, H - 52), LABEL, font=f, fill=(150, 156, 166))
    img.save(out)


def still_clip(png: Path, secs: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", str(secs), "-i", str(png)]
        + ["-f", "lavfi", "-t", str(secs), "-i", "anullsrc=r=48000:cl=stereo"]
        + ["-vf", "format=yuv420p,fade=in:0:8,fade=out:st=%.2f:d=0.25" % (secs - 0.25)]
        + ["-r", str(FPS), "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )


def norm_clip(src: Path, out: Path, secs: float | None = None) -> None:
    cut = ["-t", str(secs)] if secs else []
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), *cut]
        + ["-vf", f"scale={W}:{H},format=yuv420p"]
        + ["-r", str(FPS), "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-ar", "48000"]
        + ["-ac", "2", str(out)],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="find_chair")
    ap.add_argument("--layout", default="eight")
    ap.add_argument("--clip", action="append", required=True, help="ARM=PATH.mp4[@SECS], in order")
    ap.add_argument("--title", default="Find the chair and sit, blindfolded")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = json.loads(RESULTS.read_text())
    motors = "8 motors" if a.layout == "eight" else "4 motors"
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        parts = []
        card(
            [
                (a.title, 64, (255, 255, 255)),
                (f"the walker model, {motors}, one iteration at a time", 30, (200, 205, 212)),
                ("same room, same seed; numbers over all seeds", 26, (150, 156, 166)),
            ],
            t / "open.png",
        )
        still_clip(t / "open.png", OPEN_S, t / "p00.mp4")
        parts.append(t / "p00.mp4")
        for k, spec in enumerate(a.clip, 1):
            arm, path = spec.split("=", 1)
            path, _, secs = path.partition("@")  # @SECONDS keeps only the start of the clip
            n = arm_numbers(res, arm, a.scenario, a.layout)
            card(
                [
                    (f"{arm}  ·  {res['arms'][arm]['hypothesis']}", 48, (255, 255, 255)),
                    (
                        f"reached with no contact {n['reached']} / {n['walks']}  ·  "
                        f"{n['collisions']:.2f} contacts a walk  ·  "
                        f"min clearance {n['min_clear_m'] * 100:.0f} cm",
                        30,
                        (200, 205, 212),
                    ),
                ],
                t / f"c{k}.png",
            )
            still_clip(t / f"c{k}.png", CARD_S, t / f"p{k}a.mp4")
            norm_clip(Path(path), t / f"p{k}b.mp4", float(secs) if secs else None)
            parts += [t / f"p{k}a.mp4", t / f"p{k}b.mp4"]
        (t / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(t / "list.txt")]
            + ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-c:a", "aac", "-b:a", "192k"]
            + ["-movflags", "+faststart", str(a.out)],
            check=True,
        )
    print(a.out)


if __name__ == "__main__":
    main()
