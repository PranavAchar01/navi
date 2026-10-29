"""Frame-to-cue latency on a private server at normal priority, the way the live service runs.

    .venv/bin/python bench/cue_latency.py data/walks/<id> --label before

Starts `metasight.server` on --port (8460) with a copy of the saved house (the live house and
the :8445 service are never touched), waits for the memory worker, then streams the walk at its
recorded timing and records, for every safety message: frame sent to message back (the obstacle
cue), the server's own stage times, and the box age. Map windows come from /api/state at the end.
Unlike judge_replay the server is not started in the background band: this is a short
measurement of the live path, under 5 minutes. Writes bench/results/perf/cue-<label>.json.
"""

from __future__ import annotations

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
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from judge_replay import connect  # noqa: E402


def pct(xs, q):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1) if s else None


def state(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as r:
        return json.loads(r.read())


async def windows(port: int, out: dict) -> None:
    """Every map window's stats, polled from /api/state."""
    while True:
        m = (await asyncio.to_thread(state, port)).get("memory") or {}
        if "window" in m:
            out[m["window"]] = {k: m.get(k) for k in ("views", "infer_ms", "total_ms")}
        await asyncio.sleep(0.5)


def demote_children(pid: int, flags: str) -> None:
    kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    for k in kids:
        if flags == "pause":  # no map work at all: the reflex loop's floor
            os.kill(int(k), signal.SIGSTOP)
        else:
            subprocess.run(["taskpolicy", *flags.split(), "-p", k], check=True)
    print(f"taskpolicy {flags} on worker pids {kids}", flush=True)


async def play(walk: Path, url: str, seconds: float, port: int, policy: str = "", pid: int = 0):
    cam = json.loads((walk / "camera.json").read_text())
    frames = [json.loads(x) for x in (walk / "frames.jsonl").read_text().splitlines() if x]
    ws = await connect(url, 240)
    ready = asyncio.Event()
    cue, buzz, stages, det_age, sent = [], [], {}, [], [0]
    map_age, lead, pose_at = [], [], []  # map freshness: pose age per cue, pose lead per window

    async def reader():
        async for m in ws:
            if not isinstance(m, str):
                continue
            d = json.loads(m)
            if d["type"] == "status" and d.get("memory") == "ready":
                ready.set()
            elif d["type"] == "safety" and d.get("ts") and sent[0]:
                cue.append(time.time() * 1000 - d["ts"])
                # the belt command left before the message when the label layer runs after it
                buzz.append(cue[-1] - (d.get("after_buzz_ms") or 0))
                for k, v in (d.get("ms") or {}).items():
                    stages.setdefault(k, []).append(v)
                if d.get("det_age_ms") is not None:
                    det_age.append(d["det_age_ms"])
                if d.get("map_age") is not None:
                    map_age.append(1000 * d["map_age"])
            elif d["type"] == "pose" and sent[0]:
                pose_at.append(time.time())
            elif d["type"] == "memory" and pose_at:
                # how long before the window's end its first pose reached the client
                first = [t for t in pose_at if t > (lead[-1][0] if lead else 0)]
                lead.append((time.time(), 1000 * (time.time() - first[0]) if first else 0.0))

    rd = asyncio.create_task(reader())
    await asyncio.wait_for(ready.wait(), 240)
    if policy:
        demote_children(pid, policy)
    wins: dict = {}
    poll = asyncio.create_task(windows(port, wins))
    await ws.send(
        json.dumps(
            {
                "type": "camera",
                "hfov": cam["hfov"],
                "width": cam["width"],
                "height": cam["height"],
                "source": "cue-bench",
            }
        )
    )
    t0, start = time.time(), frames[0]["ts"]
    for i, f in enumerate(frames):
        at = (f["ts"] - start) / 1000
        if at > seconds:
            break
        await asyncio.sleep(max(0, t0 + at - time.time()))
        jpg = (walk / "frames" / f["file"]).read_bytes()
        await ws.send(struct.pack("<dI", time.time() * 1000, i) + jpg)
        sent[0] += 1
    played = time.time() - t0
    await asyncio.sleep(3)
    rd.cancel()
    poll.cancel()
    tot = [w["total_ms"] for w in wins.values()]
    inf = [w["infer_ms"] for w in wins.values()]
    await ws.close()
    return {
        "frames_sent": sent[0],
        "played_s": round(played, 1),
        "cues": len(cue),
        "cue_hz": round(len(cue) / played, 1),
        "cue_ms": {"p50": pct(cue, 0.5), "p95": pct(cue, 0.95), "max": pct(cue, 1.0)},
        "buzz_ms": {"p50": pct(buzz, 0.5), "p95": pct(buzz, 0.95), "max": pct(buzz, 1.0)},
        "server_ms": {k: {"p50": pct(v, 0.5), "p95": pct(v, 0.95)} for k, v in stages.items()},
        "det_age_ms": {"p50": pct(det_age, 0.5), "p95": pct(det_age, 0.95)} if det_age else None,
        "map_windows": len(wins),
        "map_age_ms": {"p50": pct(map_age, 0.5), "p95": pct(map_age, 0.95)} if map_age else None,
        "pose_lead_ms": {"p50": pct([x[1] for x in lead], 0.5)} if lead else None,
        "window_total_ms": {"p50": pct(tot, 0.5), "max": pct(tot, 1.0)},
        "window_infer_ms": {"p50": pct(inf, 0.5), "max": pct(inf, 1.0)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=8460)
    ap.add_argument("--belt-port", type=int, default=4260)
    ap.add_argument("--seconds", type=float, default=70)
    ap.add_argument(
        "--worker-policy",
        default="",
        help="taskpolicy flags for the memory worker once it is up ('-b'), or 'pause'",
    )
    a = ap.parse_args()

    home = Path(tempfile.mkdtemp(prefix="ms-cue-home-"))
    shutil.copytree(ROOT / "data" / "home", home, dirs_exist_ok=True)
    config = home / "cue.toml"
    config.write_text(
        (ROOT / "metasight.glasses.toml")
        .read_text()
        .replace("[safety]\n", f'[safety]\nbelt_addr = "127.0.0.1:{a.belt_port}"\n', 1)
    )
    env = {**os.environ, "METASIGHT_CONFIG": str(config), "METASIGHT_HOME": str(home)}
    log = open(tempfile.gettempdir() + f"/ms-cue-{a.port}.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "metasight.server", "--host", "127.0.0.1", "--port", str(a.port),
         "--session", "cue-bench"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )  # fmt: skip
    print(f"server pid {server.pid} on :{a.port}, log {log.name}", flush=True)
    try:
        out = asyncio.run(
            play(
                Path(a.walk),
                f"ws://127.0.0.1:{a.port}/ws",
                a.seconds,
                a.port,
                a.worker_policy,
                server.pid,
            )
        )
    finally:
        if a.worker_policy == "pause":
            os.killpg(server.pid, signal.SIGCONT)
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            os.killpg(server.pid, signal.SIGKILL)
        shutil.rmtree(home, ignore_errors=True)
    out["walk"] = Path(a.walk).name
    path = ROOT / "bench" / "results" / "perf" / f"cue-{a.label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
