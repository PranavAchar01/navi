"""Play a recorded walk into a second server for the judge view, without touching the live one.

    .venv/bin/python scripts/judge_replay.py data/walks/<id> --ask-at 45 --keep

Starts `metasight.server` on --port (8450) with a copy of the saved house, so the live
service's memory is never written. Then streams the walk's frames at their recorded timing,
says "Simulated walk" to every viewer through a device message, and at --ask-at seconds
speaks the question as a transcript and asks it. Open http://127.0.0.1:<port>/judge.html.
The walk is not recorded again (the camera hello carries no walk id).
"""

import argparse
import asyncio
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
LOCKS = ROOT.parent  # ~/helloworld, where the heavy-job slots live


def take_slot() -> Path:
    """The first free ~/helloworld/.heavy-lock[-N] slot; a dead owner's slot is taken over."""
    n = 1
    while True:
        slot = LOCKS / (".heavy-lock" if n == 1 else f".heavy-lock-{n}")
        try:
            slot.mkdir()
        except FileExistsError:
            owner = slot / "owner"
            try:
                pid = int(owner.read_text().split()[0])
                os.kill(pid, 0)
            except (OSError, ValueError, IndexError):
                if owner.exists() and time.time() - owner.stat().st_mtime > 5:
                    shutil.rmtree(slot, ignore_errors=True)
                    continue
            n += 1
            continue
        (slot / "owner").write_text(f"{os.getpid()} judge-replay server\n")
        return slot


async def connect(url: str, timeout: float):
    t0 = time.time()
    while True:
        try:
            # No keepalive: the server runs at background priority and can miss a ping under load;
            # on a loopback link a dropped socket shows up as an error anyway.
            return await websockets.connect(url, max_size=None, ping_interval=None)
        except OSError:
            if time.time() - t0 > timeout:
                raise
            await asyncio.sleep(1)


async def play(a, url: str) -> None:
    walk = Path(a.walk)
    cam = json.loads((walk / "camera.json").read_text())
    frames = [json.loads(line) for line in (walk / "frames.jsonl").read_text().splitlines() if line]
    ws = await connect(url, 240)
    ready = asyncio.Event()
    guides = {"any": 0, "dist": 0, "bearing": 0}
    # Frame sent -> its safety message back (the obstacle cue), and question -> answer, in ms.
    cue_ms: list[float] = []
    asked_at: list[float] = []
    answer_ms: list[float] = []

    async def reader():
        async for m in ws:
            if isinstance(m, str):
                d = json.loads(m)
                if d["type"] == "status" and d.get("memory") == "ready":
                    ready.set()
                elif d["type"] == "safety":
                    if d.get("ts"):
                        cue_ms.append(time.time() * 1000 - d["ts"])
                    if not d.get("guide"):
                        continue
                    g = d["guide"]
                    guides["any"] += 1
                    guides["dist"] += g.get("dist") is not None
                    guides["bearing"] += g.get("bearing") is not None
                elif d["type"] == "braille" and d["phase"] != "cell":
                    print(f"braille {d['phase']}: {d.get('cues', '')}", flush=True)
                elif d["type"] == "answer":
                    if asked_at:
                        answer_ms.append((time.time() - asked_at.pop(0)) * 1000)
                    print(f"answer: {d['text']}", flush=True)

    rd = asyncio.create_task(reader())
    try:
        await asyncio.wait_for(ready.wait(), 240)
    except TimeoutError:
        print("memory not ready after 240 s; playing anyway", flush=True)
    device = {
        "type": "device",
        "name": "Recorded walk",
        "state": "replay",
        "res": [cam["width"], cam["height"]],
        "fps": round(len(frames) / ((frames[-1]["ts"] - frames[0]["ts"]) / 1000), 1),
        "imu_hz": 0,
        "kbps": 0,
        "budget_kbps": 0,
    }
    for lap in range(a.laps):
        await ws.send(json.dumps(device))
        await ws.send(
            json.dumps(
                {
                    "type": "camera",
                    "hfov": cam["hfov"],
                    "width": cam["width"],
                    "height": cam["height"],
                    "source": "judge-replay",
                }
            )
        )
        t0, start = time.time(), frames[0]["ts"]
        print(f"lap {lap + 1} start", flush=True)
        asked = False
        for i, f in enumerate(frames):
            at = (f["ts"] - start) / 1000
            await asyncio.sleep(max(0, t0 + at - time.time()))
            jpg = (walk / "frames" / f["file"]).read_bytes()
            await ws.send(struct.pack("<dI", time.time() * 1000, lap * 100000 + i) + jpg)
            if not asked and a.ask and at >= a.ask_at:
                asked = True
                asyncio.create_task(ask(ws, a.ask, asked_at))
        print(f"lap {lap + 1}: {len(frames)} frames in {time.time() - t0:.1f} s", flush=True)
    await asyncio.sleep(a.settle)
    print(f"safety messages with a guide: {guides}", flush=True)
    if cue_ms:
        q = sorted(cue_ms)
        p = lambda f: round(q[min(len(q) - 1, int(f * len(q)))])  # noqa: E731
        print(f"cue latency ms: n={len(q)} p50={p(0.5)} p95={p(0.95)} max={round(q[-1])}")
    print(f"answer latency ms (ask sent to answer back): {[round(x) for x in answer_ms]}")
    rd.cancel()
    await ws.close()


async def ask(ws, text: str, asked_at: list[float]) -> None:
    """The question as the glasses would hear it: a few partial captions, then the final one."""
    words = text.split()
    for k in range(1, len(words)):
        await ws.send(
            json.dumps({"type": "transcript", "text": " ".join(words[:k]), "final": False})
        )
        await asyncio.sleep(0.25)
    await ws.send(json.dumps({"type": "transcript", "text": text, "final": True}))
    await asyncio.sleep(0.4)
    await ws.send(json.dumps({"type": "ask", "text": text}))
    asked_at.append(time.time())


def interrupted(*_) -> None:
    raise KeyboardInterrupt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("--port", type=int, default=8450)
    ap.add_argument(
        "--belt-port", type=int, default=4250, help="UDP port of this server's belt emulator"
    )
    ap.add_argument("--ask", default="where is my laptop")
    ap.add_argument("--ask-at", type=float, default=45)
    ap.add_argument("--braille", default="emulator", help="off, emulator, wrist or belt")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument(
        "--settle", type=float, default=8, help="seconds to keep feeding nothing after the walk"
    )
    ap.add_argument(
        "--keep", action="store_true", help="leave the server up until Ctrl-C after the walk"
    )
    ap.add_argument(
        "--no-server", action="store_true", help="feed a judge server that is already running"
    )
    a = ap.parse_args()
    url = f"ws://127.0.0.1:{a.port}/ws"
    if a.no_server:
        asyncio.run(play(a, url))
        return

    slot = take_slot()
    home = Path(tempfile.mkdtemp(prefix="ms-judge-home-"))
    shutil.copytree(ROOT / "data" / "home", home, dirs_exist_ok=True)
    # The glasses profile with the belt on its own port (the live server's belt emulator listens
    # on 4210, and a second server sending there would drive it) and the answer also played as
    # Braille cell events, for the judge view's touch panel.
    config = home / "judge.toml"
    config.write_text(
        (ROOT / "metasight.glasses.toml")
        .read_text()
        .replace(
            "[safety]\n",
            f'[safety]\nbelt_addr = "127.0.0.1:{a.belt_port}"\nbraille = "{a.braille}"\n',
            1,
        )
    )
    env = {
        **os.environ,
        "METASIGHT_CONFIG": str(config),
        "METASIGHT_FINDER": os.environ.get("METASIGHT_FINDER", "yoloe"),
        "METASIGHT_HOME": str(home),  # a copy: the live house is never written
    }
    log = open(tempfile.gettempdir() + f"/ms-judge-{a.port}.log", "w")
    server = subprocess.Popen(
        [
            "nice",
            "-n",
            "19",
            "taskpolicy",
            "-b",
            sys.executable,
            "-m",
            "metasight.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(a.port),
            "--session",
            "judge",
        ],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(
        f"judge server pid {server.pid} on :{a.port}, log {log.name}, slot {slot.name}", flush=True
    )
    print(f"open http://127.0.0.1:{a.port}/judge.html", flush=True)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        asyncio.run(play(a, url))
        if a.keep:
            print("walk done; server stays up until Ctrl-C", flush=True)
            server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        # Only the group this script started.
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            os.killpg(server.pid, signal.SIGKILL)
        shutil.rmtree(slot, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    main()
