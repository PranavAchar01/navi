"""Before/after clip of the field-of-view memory in the rendered bedroom, one scenario and seed.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Left: the walker without memory (I1). Right: with
field-of-view memory (I2), on the current `simwalk` wiring. Same scenario, same seed, same room.
Each side is the Blender render (scripts/blender_sim_room.py) with the overlay of
scripts/composite_sim_walk.py and the belt panel of web/simbelt.html, and a caption with that
walk's own collisions and minimum clearance. `--note` adds a line such as the 100-seed aggregate.
Stages are skipped when their output exists, so a crash resumes where it stopped.

    PYTHONPATH=. nice -n 19 taskpolicy -b .venv/bin/python scripts/render_fov_before_after.py \
        desk_lookup --seed 1000 --out data/demo-takes/fov/desk_lookup-before-after-v2.mp4 \
        --note "100 seeds: 21 collisions without memory, 8 with"

Take a ~/helloworld/.heavy-lock* slot before running; the Blender stage is the heavy part.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "data" / "demo-takes" / "sim"
CUT = ROOT / "web" / "public" / "_cut" / "sim"
FONT = "/System/Library/Fonts/SFNS.ttf"
LABEL = "SIMULATED USER MODEL, NOT A HUMAN STUDY"
ARMS = (("I1", "NO MEMORY", (255, 200, 80)), ("I2", "WITH FIELD-OF-VIEW MEMORY", (110, 214, 255)))
PANEL = (960, 540)
W, H = 1920, 720
PORT = 5187


def run(cmd: list[str], **kw) -> None:
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def stage_scene(name: str, arm: str, seed: int, layout: str) -> tuple[Path, dict]:
    tag = f"fov-{name}-{arm}-{seed}"
    scene = SIM / "scenes" / f"{tag}.json"
    run(
        [sys.executable, "scripts/export_sim_scene.py", name, "--layout", layout,
         "--seed", seed, "--arm", arm, "--out", scene],
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )  # fmt: skip
    doc = json.loads(scene.read_text())
    return scene, doc


def n_png(d: Path) -> int:
    return len(list(d.glob("*.png"))) if d.exists() else 0


def stage_blender(scene: Path, frames: Path) -> None:
    base = frames / "base"
    if (base / "cam.json").exists() and n_png(base) >= len(json.loads(scene.read_text())["frames"]):
        return
    base.mkdir(parents=True, exist_ok=True)
    log = frames.with_name(frames.name + ".blender.log")
    with log.open("w") as f:
        run(
            ["blender", "-b", "--factory-startup", "--python", "scripts/blender_sim_room.py",
             "--", "--scene", scene, "--frames", base],
            stdout=f, stderr=subprocess.STDOUT,
        )  # fmt: skip


def stage_belt(scene: Path, frames: Path, layout: str) -> None:
    belt = frames / "belt"
    if belt.exists() and len(list(belt.glob("*.png"))) >= len(
        list((frames / "base").glob("*.png"))
    ):
        return
    CUT.mkdir(parents=True, exist_ok=True)
    cues = CUT / f"{scene.stem}.belt.json"
    shutil.copy(scene.with_suffix(".belt.json"), cues)
    n = 8 if layout == "eight" else 4
    url = f"http://127.0.0.1:{PORT}/simbelt.html?cues=/_cut/sim/{cues.name}&layout={n}"
    dev = subprocess.Popen(
        ["pnpm", "exec", "vite", "--port", str(PORT), "--strictPort"],
        cwd=ROOT / "web", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/simbelt.html", timeout=2)
                break
            except OSError:
                time.sleep(1)
        run(["node", "web/simbelt-shot.mjs", url, "all", belt])
    finally:
        dev.terminate()
        dev.wait(timeout=20)


def stage_comp(scene: Path, frames: Path) -> None:
    comp = frames / "comp"
    if comp.exists() and len(list(comp.glob("*.png"))) >= len(
        list((frames / "base").glob("*.png"))
    ):
        return
    run(
        [sys.executable, "scripts/composite_sim_walk.py", scene, frames / "base",
         "--belt-dir", frames / "belt", "--out", comp],
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )  # fmt: skip


def caption(img: Image.Image, x: int, text: str, col) -> None:
    ImageDraw.Draw(img).text((x, 650), text, font=ImageFont.truetype(FONT, 30), fill=col)


def stack(name: str, title: str, comps: list[Path], notes: list[str], out: Path, note: str) -> None:
    tmp = out.with_suffix(".frames")
    tmp.mkdir(parents=True, exist_ok=True)
    lists = [sorted(c.glob("*.png")) for c in comps]
    total = max(len(x) for x in lists)
    font_t, font_s = ImageFont.truetype(FONT, 40), ImageFont.truetype(FONT, 26)
    for k in range(total):
        img = Image.new("RGB", (W, H), (12, 12, 16))
        ImageDraw.Draw(img).text((24, 20), title, font=font_t, fill=(240, 240, 240))
        for j, files in enumerate(lists):
            fr = Image.open(files[min(k, len(files) - 1)]).convert("RGB").resize(PANEL)
            img.paste(fr, (j * PANEL[0], 90))
            caption(img, 24 + j * PANEL[0], notes[j], ARMS[j][2])
        d = ImageDraw.Draw(img)
        d.text((24, 692), LABEL + (f"   {note}" if note else ""), font=font_s, fill=(210, 210, 210))
        img.save(tmp / f"{k:05d}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", tmp / "%05d.png",
         "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", out],
    )  # fmt: skip
    shutil.rmtree(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenario", choices=("desk_lookup", "shelf_head", "low_box", "chair_legs"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--layout", default="eight", choices=("eight", "diag4"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="", help="a line after the label, e.g. the 100-seed numbers")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from metasight import simwalk as S

    title = S.scenario(a.scenario).title
    comps, notes = [], []
    for arm, label, _ in ARMS:
        scene, doc = stage_scene(a.scenario, arm, a.seed, a.layout)
        frames = SIM / "frames" / scene.stem
        stage_blender(scene, frames)
        stage_belt(scene, frames, a.layout)
        stage_comp(scene, frames)
        m = doc["metrics"]
        coll, clear = int(m["collisions"]), float(m["min_clear_m"])
        notes.append(f"{label}: {plural(coll, 'collision')}, min clearance {round(clear * 100)} cm")
        comps.append(frames / "comp")
    stack(a.scenario, f"{title} (seed {a.seed})", comps, notes, a.out, a.note)
    print(a.out)


if __name__ == "__main__":
    main()
