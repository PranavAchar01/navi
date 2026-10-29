"""Replay a recorded walk through a private server with the map wired into the reflex and the guide.

    .venv/bin/python bench/nav/replay_walk.py data/walks/<id> --ask "where are my keys"
    .venv/bin/python bench/nav/replay_walk.py data/walks/<id> --again   # re-analyse the last run

Starts `metasight.server` on --port (8461) with a copy of the saved house, as bench/cue_latency.py
does (the live house and :8445 are never touched), plays the walk at its recorded timing with its
ARKit poses, asks the question once memory is ready and again whenever a guide ends, and keeps
every safety message, answer and map chunk. Writes bench/results/nav/replay-<walk>.json and, with
--ask, bench/results/nav-replay-<item>.png (the route the planner gave and what the belt did).

No ground truth on a real walk, so the reflex numbers are proxies, with and without the priors
(the server logs the command before and after them): motor onsets per minute, buzzes shorter than
0.5 s per minute, and for each cue the priors raised, whether the camera then read something
nearer than 1 m in that direction within 3 s.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from judge_replay import ask, connect

from metasight import nav
from metasight.belt import MODES, NAMES
from metasight.home import cells

OUT = ROOT / "bench" / "results" / "nav"
ON = {NAMES[MODES[m]] for m in ("slow", "fast", "solid")}


async def play(walk: Path, url: str, question: str | None, ask_at: float) -> dict:
    cam = json.loads((walk / "camera.json").read_text())
    frames = [json.loads(x) for x in (walk / "frames.jsonl").read_text().splitlines() if x]
    poses = {}
    if (walk / "poses.jsonl").exists():
        for line in (walk / "poses.jsonl").read_text().splitlines():
            if line:
                p = json.loads(line)
                poses[p["seq"]] = p
    ws = await connect(url, 240)
    ready = asyncio.Event()
    rec: dict = {"safety": [], "answers": [], "pins": [], "map": [], "asked": []}
    t0 = [0.0]

    async def reader():
        async for m in ws:
            if isinstance(m, bytes):
                kind, n = m[:1], struct.unpack("<I", m[1:5])[0]
                pts = np.frombuffer(m[5 : 5 + 12 * n], np.float32).reshape(-1, 3)
                if kind == b"R":
                    rec["map"] = []
                rec["map"].append(cells(pts))
                continue
            d = json.loads(m)
            t = time.time() - t0[0] if t0[0] else None
            if d["type"] == "status" and d.get("memory") == "ready":
                ready.set()
            elif d["type"] == "safety" and t is not None:
                rec["safety"].append({"t": round(t, 3), **{k: d.get(k) for k in KEEP}})
            elif d["type"] == "answer":
                rec["answers"].append({"t": t, "text": d["text"]})
                print("answer:", d["text"], flush=True)
            elif d["type"] == "pins":
                rec["pins"] = d["pins"]

    rd = asyncio.create_task(reader())
    await asyncio.wait_for(ready.wait(), 240)
    await ws.send(
        json.dumps(
            {"type": "camera", "hfov": cam.get("hfov", 63.0), "width": cam["width"],
             "height": cam["height"], "source": "nav-replay"}
        )
    )  # fmt: skip
    t0[0], start = time.time(), frames[0]["ts"]
    next_ask, guiding = ask_at, False
    for i, f in enumerate(frames):
        at = (f["ts"] - start) / 1000
        await asyncio.sleep(max(0, t0[0] + at - time.time()))
        if (p := poses.get(f.get("seq"))) is not None:
            await ws.send(json.dumps({**p, "seq": i, "ts": time.time() * 1000}))
        jpg = (walk / "frames" / f["file"]).read_bytes()
        await ws.send(struct.pack("<dI", time.time() * 1000, i) + jpg)
        # Ask once the map is up, and again 3 s after each guide ends (found or timed out).
        last = rec["safety"][-1] if rec["safety"] else {}
        if guiding and not last.get("guide"):
            guiding, next_ask = False, at + 3.0
        elif last.get("guide"):
            guiding = True
        if question and next_ask is not None and at >= next_ask:
            next_ask = None
            rec["asked"].append(round(at, 2))
            asyncio.create_task(ask(ws, question, []))
            guiding = True
    await asyncio.sleep(3)
    rd.cancel()
    await ws.close()
    rec["map"] = np.unique(np.concatenate(rec["map"]), axis=0).tolist() if rec["map"] else []
    rec["hfov"] = cam.get("hfov", 63.0)
    return rec


KEEP = ("seq", "yaw", "pos", "live", "cues", "guide", "priors", "found", "map_age")


def serve(walk: Path, port: int, belt_port: int, question: str | None, ask_at: float) -> dict:
    home = Path(tempfile.mkdtemp(prefix="ms-nav-home-"))
    shutil.copytree(ROOT / "data" / "home", home, dirs_exist_ok=True)
    config = home / "nav.toml"
    config.write_text(
        (ROOT / "metasight.glasses.toml")
        .read_text()
        .replace("[safety]\n", f'[safety]\nbelt_addr = "127.0.0.1:{belt_port}"\n', 1)
    )
    env = {**os.environ, "METASIGHT_CONFIG": str(config), "METASIGHT_HOME": str(home)}
    log = open(tempfile.gettempdir() + f"/ms-nav-{port}.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "metasight.server", "--host", "127.0.0.1", "--port", str(port),
         "--session", "nav-replay"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )  # fmt: skip
    print(f"server pid {server.pid} on :{port}, log {log.name}", flush=True)
    try:
        return asyncio.run(play(walk, f"ws://127.0.0.1:{port}/ws", question, ask_at))
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            os.killpg(server.pid, signal.SIGKILL)
        shutil.rmtree(home, ignore_errors=True)


# ------------------------------------------------------------------ analysis
def belts(rec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame and motor: obstacle cue on, with the priors and without them."""
    ts, on_with, on_without = [], [], []
    for s in rec["safety"]:
        cues = [c["mode"] for c in s["cues"]]
        with_ = [m in ON for m in cues]
        without = list(with_)
        p = s.get("priors") or {}
        if p.get("raw"):
            for k, (r, a) in enumerate(zip(p["raw"], p["after"], strict=True)):
                if r != a and cues[k] == NAMES[a[0]]:  # nothing later in the loop changed it
                    without[k] = NAMES[r[0]] in ON and r[1] > 0
        ts.append(s["t"])
        on_with.append(with_)
        on_without.append(without)
    return np.array(ts), np.array(on_with), np.array(on_without)


def runs(on: np.ndarray, ts: np.ndarray, max_s: float = 0.5) -> tuple[int, int]:
    """(onsets, onsets lasting under max_s), all motors."""
    onsets = short = 0
    for m in range(on.shape[1]):
        col = on[:, m]
        starts = np.flatnonzero(col[1:] & ~col[:-1]) + 1
        for i in starts:
            onsets += 1
            j = i
            while j < len(col) and col[j]:
                j += 1
            short += j < len(col) and ts[j] - ts[i] < max_s
    return onsets, short


def confirmed(rec: dict, hfov: float) -> dict:
    """For each sector the priors raised (hidden, new or contact), did the camera read under 1 m
    in that world direction within 3 s?"""
    sf = rec["safety"]
    out = {"raised": 0, "seen_within_3s": 0}
    for n, s in enumerate(sf):
        p = s.get("priors") or {}
        if not p.get("raw") or s.get("yaw") is None:
            continue
        for k, (r, a) in enumerate(zip(p["raw"], p["after"], strict=True)):
            if a[0] <= r[0] and a[1] <= r[1]:
                continue  # not a raise
            out["raised"] += 1
            world = s["yaw"] - math.radians(45 * (k if k <= 4 else k - 8))
            for later in sf[n:]:
                if later["t"] - s["t"] > 3.0:
                    break
                if later.get("yaw") is None:
                    continue
                for third, d in enumerate(later["live"]):
                    b = later["yaw"] - math.radians((third - 1) * hfov / 3)
                    off = abs(math.degrees(math.atan2(math.sin(b - world), math.cos(b - world))))
                    if d is not None and d < 1.0 and off < 25:
                        out["seen_within_3s"] += 1
                        break
                else:
                    continue
                break
    return out


def summary(rec: dict) -> dict:
    ts, w, wo = belts(rec)
    minutes = (ts[-1] - ts[0]) / 60 if len(ts) > 1 else 0.0
    ow, sw = runs(w, ts)
    oo, so = runs(wo, ts)
    pri = [s["priors"] for s in rec["safety"] if s.get("priors")]
    did = {k: sum(p[k] for p in pri) for k in ("hidden", "calm", "new", "contact")}
    ms = sorted(p["ms"] for p in pri)
    guide = [s for s in rec["safety"] if s.get("guide")]
    routed = [g for g in guide if g["guide"].get("route") is not None]
    stop = [s for s in guide if any(c["mode"] == NAMES[MODES["solid"]] for c in s["cues"])]
    shown = [s for s in guide if any(c["mode"] == NAMES[MODES["guide"]] for c in s["cues"])]
    return {
        "frames": len(rec["safety"]),
        "minutes": round(minutes, 2),
        "priors_active_frames": len(pri),
        "priors_changed_frames": sum(bool(p.get("raw")) for p in pri),
        "priors_did": did,
        "priors_ms": {"p50": ms[len(ms) // 2], "p95": ms[int(0.95 * len(ms))]} if ms else None,
        "with_priors": {
            "onsets_per_min": round(ow / minutes, 1) if minutes else None,
            "short_buzz_per_min": round(sw / minutes, 1) if minutes else None,
        },
        "without_priors": {
            "onsets_per_min": round(oo / minutes, 1) if minutes else None,
            "short_buzz_per_min": round(so / minutes, 1) if minutes else None,
        },
        "raised_then_seen": confirmed(rec, rec["hfov"]),
        "asked_at_s": rec["asked"],
        "answers": [a["text"] for a in rec["answers"]],
        "guide_frames": len(guide),
        "guide_frames_on_route": len(routed),
        "guide_frames_shown": len(shown),
        "guide_frames_with_stop": len(stop),
        "guide_shown_during_stop": sum(s in shown for s in stop),
        "found_frames": sum(bool(s.get("found")) for s in rec["safety"]),
    }


def picture(rec: dict, item: str, path: Path) -> dict:
    """Floor plan from the replay's map, the walk, the planned route from where the question was
    asked, and per frame what the belt did."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = nav.floor_plan(np.array(rec["map"]))
    # The server guides to a pin placed on this walk's map; the newest such one, as it does.
    named = [p for p in rec["pins"] if p["label"] in (item, item.rstrip("s"))]
    named.sort(key=lambda p: (bool(p.get("placed")), p.get("ts", 0)))
    pin = named[-1] if named else None
    route = []
    sf = [s for s in rec["safety"] if s.get("pos")]
    first = next((s for s in sf if s.get("guide")), None)
    if pin is not None and first is not None:
        nv = nav.Navigator(plan)
        if nv.set_goal(pin["pos"], pin["label"]):
            r = plan.blocked.shape[1]
            c = plan.snap(*plan.ij(first["pos"][0], first["pos"][2]))
            while c is not None:
                route.append((float(plan.xs[c[0]]), float(plan.zs[c[1]])))
                k = int(nv.next[c[0] * r + c[1]])
                c = None if k < 0 else divmod(k, r)
    ext = [plan.zs[0], plan.zs[-1], plan.xs[-1], plan.xs[0]]
    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    img = np.where(plan.blocked, 0.35, np.where(plan.seen, 0.96, 0.85))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    if sf:
        z, x = [s["pos"][2] for s in sf], [s["pos"][0] for s in sf]
        ax.plot(z, x, color="#8a8f98", lw=1.2, label="walk (head position)")
    if route:
        ax.plot([p[1] for p in route], [p[0] for p in route], color="#2a6fdb", lw=2.2,
                label="planned route")  # fmt: skip
    kinds = {"guide": ("#2a6fdb", "Guide pulse on"), "stop": ("#d1495b", "Stop (solid) on")}
    for kind, (color, label) in kinds.items():
        pts = []
        for s in sf:
            g, cues = s.get("guide"), [c["mode"] for c in s["cues"]]
            if not g:
                continue
            if kind == "stop" and NAMES[MODES["solid"]] in cues:
                pts.append(s["pos"])
            elif kind == "guide" and NAMES[MODES["guide"]] in cues:
                pts.append(s["pos"])
        if pts:
            ax.scatter([p[2] for p in pts], [p[0] for p in pts], s=10, color=color, label=label,
                       zorder=3)  # fmt: skip
    for s in sf[::15]:  # the direction the belt pointed, every ~1 s
        g = s.get("guide")
        if g and g.get("bearing") is not None and s.get("yaw") is not None:
            b = s["yaw"] - math.radians(g["bearing"])
            ax.arrow(s["pos"][2], s["pos"][0], 0.35 * math.cos(b), 0.35 * math.sin(b),
                     width=0.012, color="#2a6fdb", alpha=0.8, zorder=4)  # fmt: skip
    if pin is not None:
        ax.scatter([pin["pos"][2]], [pin["pos"][0]], marker="*", s=220, color="#e0a100",
                   edgecolor="black", zorder=5, label=pin["label"])  # fmt: skip
    if sf:  # the map, the walk and the item all in frame
        zs = [*z, *(p[1] for p in route), *([pin["pos"][2]] if pin else [])]
        xs = [*x, *(p[0] for p in route), *([pin["pos"][0]] if pin else [])]
        ax.set_xlim(min(ext[0], *zs) - 0.3, max(ext[1], *zs) + 0.3)
        ax.set_ylim(max(ext[2], *xs) + 0.3, min(ext[3], *xs) - 0.3)
    ax.set_xlabel("map z (m)")
    ax.set_ylabel("map x (m)")
    ax.set_title(f'"{rec["question"]}": replay of a recorded walk')
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "pin": pin,
        "route_m": round(sum(math.dist(a, b) for a, b in zip(route, route[1:], strict=False)), 2),
        "route": [[round(a, 2), round(b, 2)] for a, b in route],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("--ask", help='question, e.g. "where are my keys"')
    ap.add_argument("--ask-at", type=float, default=3.0)
    ap.add_argument("--port", type=int, default=8461)
    ap.add_argument("--belt-port", type=int, default=4261)
    ap.add_argument("--again", action="store_true", help="re-analyse the saved run")
    a = ap.parse_args()
    walk = Path(a.walk)
    raw = OUT / f"replay-{walk.name}.raw.json"
    if a.again:
        rec = json.loads(raw.read_text())
    else:
        rec = serve(walk, a.port, a.belt_port, a.ask, a.ask_at)
        rec["question"] = a.ask
        raw.write_text(json.dumps(rec))
    out = {"walk": walk.name, "question": rec.get("question"), **summary(rec)}
    if rec.get("question"):
        item = rec["question"].split()[-1]
        png = ROOT / "bench" / "results" / f"nav-replay-{item}.png"
        out["picture"] = str(png.relative_to(ROOT))
        out.update(picture(rec, item, png))
        out["frames_log"] = [
            {"t": s["t"], "pos": s["pos"], "yaw": s["yaw"], "guide": s["guide"],
             "stop": any(c["mode"] == NAMES[MODES["solid"]] for c in s["cues"]),
             "cues": [c["mode"] for c in s["cues"]]}
            for s in rec["safety"] if s.get("guide")
        ]  # fmt: skip
    (OUT / f"replay-{walk.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("route", "frames_log")}, indent=1))


if __name__ == "__main__":
    main()
