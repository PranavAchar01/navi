"""One clip per walker iteration, then the montage of the runs improving.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Reads bench/results/sim-walk.json (scripts/run_sim_walk.py)
and, for each arm in `CHAIN`, renders the same scenario and seed with that arm's settings, with the
held-out numbers for the arm in a card on the frame. The montage cuts the first seconds of each
clip together behind a short title card per iteration.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/render_sim_iterations.py \
        --chain I0 I1 I2 --scenario avoid_chair --seed 1003 --layout diag4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_sim_walk as R  # noqa: E402
import run_sim_walk as E  # noqa: E402

from metasight import simwalk as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "demo-takes" / "sim"


def card_lines(arm: str, res: dict, layout: str) -> list[str]:
    s = res["arms"][arm]["summary"][layout]
    n = len(range(*res["seeds"])) + 1
    t = s["time_to_goal_s"]
    return [
        f"{arm} held-out, {s['walks']} walks ({n} seeds x 6)",
        f"success        {s['success']:.2f}",
        f"collisions     {s['collisions']:.2f} /walk",
        f"min clearance  {s['min_clear_m']:.2f} m",
        f"time to goal   {t:.1f} s" if t else "time to goal   -",
        f"false buzzes   {s['false_per_min']:.1f} /min",
    ]


def title_card(path: Path, lines: list[str], secs: float = 1.6) -> None:
    img = Image.new("RGB", (R.W, R.H), R.BG)
    dr = ImageDraw.Draw(img)
    y = R.H // 2 - 30 * len(lines) // 2 - 20
    for k, ln in enumerate(lines):
        f = R.F_L if k == 0 else R.F_M
        dr.text((R.W // 2, y), ln, font=f, fill=R.HOT if k == 0 else R.INK, anchor="mm")
        y += 44 if k == 0 else 30
    dr.text(
        (R.W // 2, R.H - 40),
        "SIMULATED USER MODEL, NOT A HUMAN STUDY",
        font=R.F_S,
        fill=(255, 200, 90),
        anchor="mm",
    )
    frame = np.asarray(img).tobytes()
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{R.W}x{R.H}",
            "-r",
            str(R.FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    for _ in range(int(secs * R.FPS)):
        p.stdin.write(frame)
    p.stdin.close()
    p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", nargs="+", required=True)
    ap.add_argument("--scenario", default="avoid_chair")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--layout", default="diag4")
    ap.add_argument("--clip-s", type=float, default=7.0)
    ap.add_argument(
        "--extra", nargs="*", default=[], help="scenario:seed:layout walks for the last arm"
    )
    a = ap.parse_args()
    res = json.loads(E.OUT.read_text())
    it_dir = OUT / "iterations"
    parts: list[Path] = []
    tmp = OUT / "parts"
    tmp.mkdir(parents=True, exist_ok=True)
    for k, arm in enumerate(a.chain):
        _, belt_kw, walk_kw, hyp = E.ARMS[arm]
        sc = S.scenario(a.scenario)
        tr = S.walk(sc, E.belt_for(arm, a.layout), seed=a.seed, **walk_kw)
        m = tr.metrics()
        out = it_dir / f"{k:02d}-{arm}-{a.scenario}-{a.layout}.mp4"
        label = f"Iteration {k} ({arm}): {hyp}"
        R.render(sc, tr, a.layout, out, label, card=card_lines(arm, res, a.layout))
        print(out, m)
        s = res["arms"][arm]["summary"][a.layout]
        tc = tmp / f"title-{k:02d}.mp4"
        title_card(
            tc,
            [
                f"Iteration {k}",
                hyp,
                f"held-out: {s['collisions']:.2f} collisions/walk, success {s['success']:.2f}, "
                f"{s['false_per_min']:.1f} false buzzes/min",
            ],
        )
        cut = tmp / f"cut-{k:02d}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(out),
                "-t",
                str(a.clip_s),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(cut),
            ],
            check=True,
        )
        parts += [tc, cut]
    last = a.chain[-1]
    for j, spec in enumerate(a.extra):
        name, seed, lay = spec.split(":")
        sc = S.scenario(name)
        tr = S.walk(sc, E.belt_for(last, lay), seed=int(seed), **E.ARMS[last][2])
        out = it_dir / f"final-{last}-{name}-{lay}.mp4"
        motors = "8 motors" if lay == "eight" else "4 built motors"
        R.render(
            sc,
            tr,
            lay,
            out,
            f"Final ({last}), {motors}: {sc.title}",
            card=card_lines(last, res, lay),
        )
        print(out, tr.metrics())
        cut = tmp / f"extra-{j:02d}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(out),
                "-t",
                str(a.clip_s),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(cut),
            ],
            check=True,
        )
        parts.append(cut)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c",
            "copy",
            str(OUT / "montage.mp4"),
        ],
        check=True,
    )
    print(OUT / "montage.mp4")


if __name__ == "__main__":
    main()
