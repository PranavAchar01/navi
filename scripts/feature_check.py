"""Morning check: every feature in docs/NIGHT-PLAN-2026-10-29.md, run against a private server.

The server is a separate copy on a free port (never the live one on 8445), with its own notes,
sessions, walks and a copy of the home map, so nothing it does reaches the live service. It is
fed one of Pranav's recorded walks at real speed, synthetic speech (macOS `say`) and the
two-voice stereo track from bench/people/make_two_voices.py, and every check reads the server's
own events and APIs. Each feature gets PASS, FAIL, NOT-BUILT or SKIP (unmeasurable on this walk)
with one number or one reason.

.venv/bin/python scripts/feature_check.py [--walk 20261028-135040-iphone] [--work DIR]

Writes docs/MORNING-CHECK.md and bench/results/morning-check.json, and appends a line to the
Jarvis daily log.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if os.environ.get("FEATURE_CHECK_SESSIONS"):
    # The private server (and its spawned memory worker, which re-runs this module) keeps its
    # notes, watch list and session files away from the live service's sessions/ folder.
    import metasight.config as _cfg

    _cfg.SESSIONS = Path(os.environ["FEATURE_CHECK_SESSIONS"])

import argparse  # noqa: E402
import asyncio  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

SR = 16000
LIVE_PORT = 8445
DOC = ROOT / "docs" / "MORNING-CHECK.md"
EVENTS = ROOT / "bench" / "events"
VLOAD, VLOAD_NAME = "voiceload", "Spoken question before the map loads"
RESULTS = ROOT / "bench" / "results" / "morning-check.json"
JARVIS_LOG = ROOT.parent / "jarvis" / "log"
STOP_MODES = ("solid", "stop")  # the belt's stop layer: about to touch
CAMERA_SECTORS = ("front-left", "front", "front-right")
# Cue scoring against bench/labels (lead's call 02:35: a raw on-rate cannot tell a right belt
# from a broken one on a cluttered walk).
FALSE_NAME, FALSE_M, FALSE_SCALE, FALSE_MIN = "False front cue vs labels", 2.0, 1.16, 20
URGENT_NAME, URGENT_M, URGENT_SCALE, URGENT_MAX_PCT = (
    "Over-urgent solid cue vs labels",
    1.5,
    1.16,
    10,
)
MISS_NAME, MISS_M, MISS_SCALE = "Missed near obstacle vs labels", 1.0, 0.84
# Each row fails only when the belt is wrong even under the depth's 16% scale error.
LABEL_SCALE_NOTE = (
    "labels are at the live depth scale (~0.4) while an ARKit check puts the true scale near "
    "0.75, so they will be redone once the iPhone profile's scale is fixed and this may become "
    'measurable (docs/haptics-sim.md, "Correction 2026-10-29")'
)
SKIP_FIND = {"person", "wall", "floor", "window", "curtain", "rug"}


def free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def speech(text: str, voice: str | None = None) -> bytes:
    """16 kHz mono int16 of `text` spoken by macOS `say`."""
    with tempfile.TemporaryDirectory() as d:
        aiff = Path(d) / "q.aiff"
        subprocess.run(
            ["say", *(["-v", voice] if voice else []), "-o", str(aiff), text], check=True
        )
        return subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(aiff),
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                str(SR),
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout


def two_voices(work: Path, secs: float = 20) -> bytes:
    """The stereo two-talker track (voice 1 at -50 deg from 1 s, voice 2 at +40 deg from 10 s)."""
    blank, mov = work / "blank.mp4", work / "two.mov"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=320x240:d={secs}"]
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(blank)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench/people/make_two_voices.py"),
            str(blank),
            str(mov),
            str(secs),
        ],
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(mov),
            "-vn",
            "-ar",
            str(SR),
            "-ac",
            "2",
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout


# ---------------------------------------------------------------- the private server


def serve(port: int) -> None:
    import uvicorn

    from metasight.server import create_app

    uvicorn.run(
        create_app(session="feature-check"), host="127.0.0.1", port=port, log_level="warning"
    )


def private_config(work: Path, belt_port: int) -> Path:
    """The live glasses profile with three changes: the map review every 2 min instead of 10 (so
    one happens in the run), SAM 3.1 crops every 10 s instead of 2 (cost), and its own belt
    emulator port."""
    text = (ROOT / "metasight.glasses.toml").read_text()
    extra = {
        "providers": "review_every_min = 2",
        "safety": f'belt_addr = "127.0.0.1:{belt_port}"',
        "memory": "meta_interval_s = 10.0",
    }
    for section, line in extra.items():
        key = line.split("=")[0].strip()
        text = re.sub(rf"(?m)^{key}\s*=.*\n", "", text)
        text = text.replace(f"[{section}]\n", f"[{section}]\n{line}\n", 1)
    out = work / "metasight.toml"
    out.write_text(text)
    return out


def heavy_slot() -> Path:
    """First free ~/helloworld/.heavy-lock slot, ours by mkdir; a slot whose owner died is stale."""
    i = 1
    while True:
        slot = ROOT.parent / (".heavy-lock" if i == 1 else f".heavy-lock-{i}")
        try:
            slot.mkdir()
        except FileExistsError:
            try:
                owner = int((slot / "owner").read_text().strip())
            except (FileNotFoundError, ValueError):
                # Just taken, owner not written yet: busy, unless it has sat ownerless ~10 s.
                try:
                    young = time.time() - slot.stat().st_mtime < 10
                except FileNotFoundError:
                    continue  # released meanwhile: try this slot again
                if not young:
                    shutil.rmtree(slot, ignore_errors=True)
                    continue
                i += 1
                continue
            try:
                os.kill(owner, 0)
            except ProcessLookupError:
                shutil.rmtree(slot, ignore_errors=True)
                continue
            except PermissionError:
                pass  # alive, someone else's process
            i += 1
            continue
        tmp = slot / f".owner.{os.getpid()}"
        tmp.write_text(f"{os.getpid()}\n")
        tmp.rename(slot / "owner")  # atomic: a reader sees no owner or the whole PID
        return slot


def preflight() -> str | None:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    if m and int(m.group(1)) < 10:
        return f"memory free {m.group(1)}% (< 10%)"
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    if free_gib < 15:
        return f"disk free {free_gib:.1f} GiB (< 15)"
    return None


# ---------------------------------------------------------------- the client


class Client:
    def __init__(self, url: str, http: str):
        self.url, self.http = url, http
        self.events: list[tuple[float, dict]] = []
        self.sent: dict[int, float] = {}  # frame seq -> wall time it left
        self.audio_sent: list[float] = []  # wall time of each audio chunk of the current stream
        self.stop_frames = False
        self.ws = None

    async def reader(self) -> None:
        async for m in self.ws:
            if isinstance(m, bytes):
                continue
            self.events.append((time.time(), json.loads(m)))

    def since(self, t: float, kind: str) -> list[dict]:
        return [e for at, e in self.events if at >= t and e.get("type") == kind]

    async def wait(self, t: float, kind: str, pred=lambda e: True, timeout: float = 20.0):
        """First `kind` event after wall time `t` that `pred` accepts, with its arrival time."""
        end = time.time() + timeout
        i = 0
        while time.time() < end:
            n = len(self.events)
            for at, e in self.events[i:n]:
                if at >= t and e.get("type") == kind and pred(e):
                    return e, at
            i = n
            await asyncio.sleep(0.05)
        return None, None

    async def ask(self, text: str) -> tuple[dict, float]:
        import httpx

        t0 = time.time()
        async with httpx.AsyncClient(timeout=60) as h:
            r = await h.post(f"{self.http}/api/ask", json={"text": text})
        return r.json(), 1000 * (time.time() - t0)

    async def state(self) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=10) as h:
            return (await h.get(f"{self.http}/api/state")).json()

    async def send(self, msg: dict) -> None:
        await self.ws.send(json.dumps(msg))

    async def frames(self, walk: Path, passes: list[int]) -> None:
        """Stream the walk at its recorded pace, looping until told to stop. `passes` counts
        finished passes."""
        rows = [json.loads(x) for x in (walk / "frames.jsonl").read_text().splitlines() if x]
        poses = {}
        if (walk / "poses.jsonl").exists():
            for x in (walk / "poses.jsonl").read_text().splitlines():
                if x:
                    p = json.loads(x)
                    poses[p.get("seq")] = p
        seq = 0
        while not self.stop_frames:
            t0, base = time.time(), rows[0]["ts"]
            for r in rows:
                if self.stop_frames:
                    return
                await asyncio.sleep(max(0.0, t0 + (r["ts"] - base) / 1000 - time.time()))
                if (p := poses.get(r["seq"])) is not None:
                    await self.send({**p, "seq": seq, "ts": time.time() * 1000})
                jpg = (walk / "frames" / r["file"]).read_bytes()
                self.sent[seq] = time.time()
                await self.ws.send(struct.pack("<dI", time.time() * 1000, seq) + jpg)
                seq += 1
            passes[0] += 1

    async def audio(self, pcm: bytes, ch: int, chunk: int = 1024, tail_s: float = 3.0) -> float:
        """Send int16 audio at real time, then silence; returns when the last sound left."""
        step = chunk * 2 * ch
        self.audio_sent = []
        t = time.time()
        msg = {"type": "audio", "sr": SR, "ch": ch, **({"spacing": 0.14} if ch == 2 else {})}
        for i in range(0, len(pcm), step):
            self.audio_sent.append(time.time())
            await self.send({**msg, "pcm": base64.b64encode(pcm[i : i + step]).decode()})
            t += chunk / SR
            await asyncio.sleep(max(0.0, t - time.time()))
        spoken = time.time()
        silence = base64.b64encode(b"\0" * step).decode()
        for _ in range(int(tail_s * SR / chunk)):
            await self.send({**msg, "pcm": silence})
            t += chunk / SR
            await asyncio.sleep(max(0.0, t - time.time()))
        return spoken


async def voice_before_map(c: Client) -> dict:
    """A spoken question while the home map is still loading must get the loading line, not
    silence: the voice path answers like the typed one does."""
    import websockets

    for _ in range(10):  # the voice service can come up a moment after HTTP
        st = await c.state()
        if st.get("memory_ready"):
            return row(VLOAD, VLOAD_NAME, "metasight-muse", "SKIP", "", "map loaded before the ask")
        if st.get("asr") is not None:
            break
        await asyncio.sleep(1)
    else:
        return row(VLOAD, VLOAD_NAME, "metasight-muse", "FAIL", "", "voice service never came up")
    async with websockets.connect(c.url, max_size=None, ping_interval=None) as ws:
        c.ws = ws
        rd = asyncio.create_task(c.reader())
        t0 = time.time()
        spoken = await c.audio(speech("where are my keys"), 1)
        a, at = await c.wait(t0, "answer", lambda e: True, 8)
        ready = (await c.state()).get("memory_ready")
        heard = [e["text"] for e in c.since(t0, "transcript") if e.get("final")]
        fin = [
            at_
            for at_, e in c.events
            if at_ >= t0 and e.get("type") == "transcript" and e.get("final")
        ]
        asr = (await c.state()).get("asr") or {}
        rd.cancel()
    c.ws = None
    if a is None:
        if ready:
            note = "map finished loading during the ask; "
        else:
            note = ""
        return row(
            VLOAD, VLOAD_NAME, "metasight-muse", "FAIL", "",
            f"{note}no answer in 8 s after the speech; finals heard {heard}",
        )  # fmt: skip
    ms = 1000 * (at - spoken)
    loading = "still loading" in (a.get("text") or "").lower()
    ok = loading and ms <= 2000
    return row(
        VLOAD, VLOAD_NAME, "metasight-muse", "PASS" if ok else "FAIL", f"{ms:.0f} ms",
        f"spoken 'where are my keys' before memory_ready -> \"{a.get('text')}\" "
        f"{ms:.0f} ms after the speech ended (limit 2000); final transcript at "
        f"{1000 * (fin[0] - spoken):.0f} ms; finals heard {heard}; "
        f"asr handshake {asr.get('handshake_ms')} ms, warmup {asr.get('warmup_ms')} ms"
        if fin
        else f"{ms:.0f} ms after the speech ended (limit 2000); no final transcript event",
    )  # fmt: skip


def row(key, feature, owner, status, number="", evidence=""):
    return {
        "key": key,
        "feature": feature,
        "owner": owner,
        "status": status,
        "number": number,
        "evidence": evidence,
    }


def src(path: str) -> str:
    p = ROOT / path
    return p.read_text() if p.exists() else ""


def cue_rows(safety: list, walk: Path) -> list:
    """Belt cues scored against the walk's ground-truth labels (make_sector_labels.py), not a raw
    on-rate: a cluttered walk needs a busy belt. A label frame is every 2nd recorded frame, so a
    safety event's seq maps to label seq % len(rows) // 2 on even rows. Labels hold the nearest
    obstacle per camera third; a cue counts as on at 26/255, the skin model's floor."""
    lab_path = ROOT / "bench" / "labels" / f"{walk.name}-sectors.json"
    if not safety or not lab_path.exists():
        why = "no safety frames" if not safety else f"no labels at {lab_path.relative_to(ROOT)}"
        return [
            row(k, n, "metasight-compare", "SKIP", "", why)
            for k, n in (("falsecue", FALSE_NAME), ("urgent", URGENT_NAME), ("missnear", MISS_NAME))
        ]
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    from replay_false_buzz import CACHE, commands

    labels = json.loads(lab_path.read_text())["frames"]
    n_rows = sum(1 for x in (walk / "frames.jsonl").read_text().splitlines() if x)
    d = np.load(CACHE)
    w = walk.name
    offline = commands(
        d[f"{w}/strips"], d[f"{w}/path"], d[f"{w}/ttc"], float(d[f"{w}/hfov"][0]), "sector", 2, 26
    ).any(axis=1)  # the sector scheme has only the three front motors, no priors

    def nearest(f: dict) -> float:
        return min((v for v in f["fov_thirds"] if v is not None), default=9.0)

    front_on = solid = 0
    far = far_on = 0
    far_idx: set[int] = set()
    solid_lab = solid_far = 0
    close = close_miss = 0
    roomy = 0  # labelled frames with nothing inside the over-urgent distance
    close_idx: set[int] = set()
    for _, e in safety:
        front = [q for n, q in zip(e["names"], e["cues"], strict=False) if n in CAMERA_SECTORS]
        f_on = any(q["mode"] != "off" and q["intensity"] >= 26 / 255 for q in front)
        s_on = any(q["mode"] in STOP_MODES and q["intensity"] > 0 for q in e["cues"])
        front_on += f_on
        solid += s_on
        r = e["seq"] % n_rows
        if r % 2 or r // 2 >= len(labels) or not labels[r // 2]["tracked"]:
            continue
        i = r // 2
        near = nearest(labels[i])
        if near * FALSE_SCALE >= FALSE_M:
            far += 1
            far_on += f_on
            far_idx.add(i)
        if s_on:
            solid_lab += 1
            solid_far += near * URGENT_SCALE > URGENT_M
        roomy += near * URGENT_SCALE > URGENT_M
        if near * MISS_SCALE < MISS_M:
            close += 1
            close_miss += not f_on
            close_idx.add(i)
    pri = [e["priors"] for _, e in safety if e.get("priors")]
    info = (
        f"info: front on {100 * front_on / len(safety):.0f}% and solid/stop "
        f"{100 * solid / len(safety):.0f}% of {len(safety)} frames; "
        + (
            f"priors in shadow mode, active in {len(pri)} frames, would have changed "
            f"{sum('after' in p for p in pri)}"
            if pri and all(p.get("shadow") for p in pri)
            else f"priors applied in {len(pri)} frames"
            if pri
            else "no priors in the frame events"
        )
    )
    out = []
    if far < FALSE_MIN:
        out.append(
            row(
                "falsecue",
                FALSE_NAME,
                "metasight-compare",
                "SKIP",
                f"{far} frames",
                f"unmeasurable on this walk ({far} frames with nothing labelled inside "
                f"{FALSE_M} m in view at labels x{FALSE_SCALE}, need {FALSE_MIN}); "
                f"{LABEL_SCALE_NOTE}; {info}",
            )
        )
    else:
        live_pct = 100 * far_on / far
        off_pct = 100 * float(offline[sorted(far_idx)].mean())
        out.append(
            row(
                "falsecue",
                FALSE_NAME,
                "metasight-compare",
                "PASS" if live_pct <= off_pct + 5 else "FAIL",
                f"{far_on} of {far} ({live_pct:.0f}%)",
                f"front cue on while the nearest labelled obstacle in view is {FALSE_M} m or "
                f"more (labels x{FALSE_SCALE}); offline sector scheme on the same "
                f"{len(far_idx)} label frames {off_pct:.0f}%, FAIL above {off_pct + 5:.0f}%; "
                + info,
            )
        )
    u_pct = 100 * solid_far / solid_lab if solid_lab else 0.0
    if roomy < FALSE_MIN:
        out.append(
            row(
                "urgent",
                URGENT_NAME,
                "metasight-compare",
                "SKIP",
                f"{roomy} frames",
                f"unmeasurable on this walk ({roomy} frames with nothing labelled within "
                f"{URGENT_M} m in view at labels x{URGENT_SCALE}, need {FALSE_MIN}); "
                + LABEL_SCALE_NOTE,
            )
        )
    else:
        out.append(
            row(
                "urgent",
                URGENT_NAME,
                "metasight-compare",
                "PASS" if u_pct <= URGENT_MAX_PCT else "FAIL",
                f"{solid_far} of {solid_lab} ({u_pct:.0f}%)",
                f"solid/stop cue while the nearest labelled obstacle in view is beyond "
                f"{URGENT_M} m (labels x{URGENT_SCALE}), over solid frames on labelled frames; "
                f"FAIL above {URGENT_MAX_PCT}%",
            )
        )
    if close < FALSE_MIN:
        out.append(
            row(
                "missnear",
                MISS_NAME,
                "metasight-compare",
                "SKIP",
                f"{close} frames",
                f"unmeasurable on this walk ({close} frames with an obstacle labelled under "
                f"{MISS_M} m in view at labels x{MISS_SCALE}, need {FALSE_MIN})",
            )
        )
    else:
        m_pct = 100 * close_miss / close
        off_miss = 100 * float((~offline[sorted(close_idx)]).mean())
        out.append(
            row(
                "missnear",
                MISS_NAME,
                "metasight-compare",
                "PASS" if m_pct <= off_miss + 5 else "FAIL",
                f"{close_miss} of {close} ({m_pct:.0f}%)",
                f"no front cue while the nearest labelled obstacle in view is under {MISS_M} m "
                f"(labels x{MISS_SCALE}); offline sector scheme misses {off_miss:.0f}% of the "
                f"same {len(close_idx)} label frames, FAIL above {off_miss + 5:.0f}%",
            )
        )
    return out


async def run_checks(c: Client, walk: Path, work: Path, results: list) -> dict:
    import websockets

    meta: dict = {}
    # no keepalive: a server at background priority can miss a ping and drop us
    async with websockets.connect(c.url, max_size=None, ping_interval=None) as ws:
        c.ws = ws
        rd = asyncio.create_task(c.reader())
        t_start = time.time()
        cam = json.loads((walk / "camera.json").read_text())
        await c.send(
            {
                "type": "camera",
                "source": cam.get("source", "iphone"),
                "hfov": cam.get("hfov"),
                "width": cam.get("width"),
                "height": cam.get("height"),
            }
        )
        await c.send({"type": "onboarding", "action": "start"})
        passes = [0]
        ft = asyncio.create_task(c.frames(walk, passes))
        n_frames = len((walk / "frames.jsonl").read_text().splitlines())
        print(f"streaming {walk.name} ({n_frames} frames), onboarding started", flush=True)
        while passes[0] < 1 and not ft.done():
            await asyncio.sleep(0.5)
        if ft.done() and ft.exception():
            raise ft.exception()
        t_walk = time.time()
        print(f"walk pass 1 done in {t_walk - t_start:.0f} s", flush=True)

        # ---- onboarding: walk -> name -> belt intro -> done
        walk_evs = [e for e in c.since(t_start, "onboarding") if e.get("step") == "walk"]
        area = max((e.get("area_m2", 0) for e in walk_evs), default=0)
        steps = []
        for line in ("that's all", "these are my keys", "that's all"):
            await c.ask(line)
            await asyncio.sleep(0.5)
        done, _ = await c.wait(t_walk, "onboarding", lambda e: e.get("step") == "done", 40)
        ob = c.since(t_start, "onboarding")
        steps = list(dict.fromkeys(e.get("step") for e in ob))
        demo_modes = [e["mode"] for e in ob if e.get("mode")]
        got_keys = any("Got it, keys" in (e.get("say") or "") for e in ob)
        if done and got_keys and "name" in steps and "belt" in steps:
            results.append(
                row(
                    "onboarding",
                    "Onboarding steps (walk, name, belt intro, done)",
                    "metasight-people",
                    "PASS",
                    f"{len(steps)} steps",
                    f"steps {' > '.join(steps)}; walked area {area} m2 before 'that's all'; "
                    f"'Got it, keys.'; belt intro played {len(demo_modes)} mode patterns",
                )
            )
        else:
            results.append(
                row(
                    "onboarding",
                    "Onboarding steps (walk, name, belt intro, done)",
                    "metasight-people",
                    "FAIL",
                    f"{len(steps)} steps",
                    f"steps seen {steps}, keys named {got_keys}, done {bool(done)}",
                )
            )
        belt_during = [
            e
            for at, e in c.events
            if e.get("type") == "belt" and at >= t_walk and any(m != "off" for m in e["modes"])
        ]
        pat_src = src("metasight/onboarding.py")
        stand_in = "stand-ins" in pat_src
        if len(set(demo_modes)) == 4 and belt_during:
            results.append(
                row(
                    "patterns",
                    "Per-mode belt patterns in the intro",
                    "metasight-braille",
                    "PASS",
                    f"{len(set(demo_modes))} of 4 modes",
                    f"intro named {', '.join(demo_modes)}; {len(belt_during)} belt states with "
                    "motors on"
                    + ("; PATTERNS still marked stand-ins in onboarding.py" if stand_in else ""),
                )
            )
        else:
            results.append(
                row(
                    "patterns",
                    "Per-mode belt patterns in the intro",
                    "metasight-braille",
                    "FAIL",
                    f"{len(set(demo_modes))} of 4 modes",
                    f"intro modes {demo_modes}, belt states with motors on {len(belt_during)}",
                )
            )

        # ---- let the map catch up on the walk
        settle_end = time.time() + 90
        st = await c.state()
        while time.time() < settle_end and (st["pending_kf"] > 0 or not st["pins"]):
            await asyncio.sleep(3)
            st = await c.state()
        meta["pins"] = sorted({p["label"] for p in st["pins"]})
        print(f"map settled: {len(st['pins'])} pins, pending {st['pending_kf']}", flush=True)

        # ---- mode switches
        t_m = time.time()
        switched = []
        for m in ("find", "talk", "explore", "guide"):
            t0 = time.time()
            await c.ask(f"{m} mode")
            e, at = await c.wait(t0, "activity", lambda e, m=m: e.get("mode") == m, 5)
            if e:
                switched.append((m, bool(e.get("say")), 1000 * (at - t0)))
        ok = len(switched) == 4 and all(s[1] for s in switched)
        results.append(
            row(
                "modes",
                "Mode switches by voice (find, talk, explore, guide)",
                "metasight-people",
                "PASS" if ok else "FAIL",
                f"{len(switched)} of 4",
                (
                    f"each switch sent an activity event with its spoken line; slowest "
                    f"{max(s[2] for s in switched):.0f} ms"
                    if switched
                    else "no activity events"
                )
                + ("" if ok else f"; got {[s[0] for s in switched]}"),
            )
        )
        del t_m

        # ---- find a seen item, then the guide bearings while frames keep flowing
        # Only items with a real, confirmed pin (score 0.3 or more, or SAM 3.1), never a known
        # false one: the backpack when it qualifies, else the most-seen such item.
        def real(p: dict) -> bool:
            return p["id"] not in KNOWN_FALSE_PINS and (
                (p.get("score") or 0) >= WEAK_PIN_SCORE or p.get("source") == "sam-3.1"
            )

        counts = {}
        for p in st["pins"]:
            if p["label"] not in SKIP_FIND and real(p):
                counts[p["label"]] = max(counts.get(p["label"], 0), p.get("seen", 0))
        # The guide ends as found inside 0.8 m, so ask for something not right beside the wearer:
        # the most-seen label with a placed pin at least 1.5 m from the latest position.
        pos = next(
            (e["pos"] for _, e in reversed(c.events) if e.get("type") == "safety" and e.get("pos")),
            None,
        )
        far = {
            lab
            for lab in counts
            for p in st["pins"]
            if p["label"] == lab
            and p.get("placed")
            and pos is not None
            and sum((a - b) ** 2 for a, b in zip(p["pos"], pos, strict=False)) ** 0.5 >= 1.5
        }
        pool = {k: v for k, v in counts.items() if k in far} or counts
        label = max(pool, key=pool.get) if pool else None
        why_label = "most-seen item with a confirmed pin" + (" 1.5 m away" if far else "")
        if "backpack" in counts:
            label, why_label = "backpack", "backpack has a confirmed pin"
        meta["find_far"] = bool(far)
        meta["find_choice"] = f"{label} ({why_label})"
        meta["find_label"] = label
        if label is None:
            results.append(
                row(
                    "find",
                    "Find a seen item",
                    "metasight-display",
                    "FAIL",
                    "0 pins",
                    "the walk left no pins to ask about",
                )
            )
        else:
            await c.send({"type": "activity", "mode": "guide"})  # hold guide mode for the planner
            t0 = time.time()
            res, ms = await c.ask(f"where is my {label}")
            good = res.get("label") == label and (res.get("pin") or res.get("live"))
            pnote, false_pin = pin_note(res)
            results.append(
                row(
                    "find",
                    "Find a seen item",
                    "metasight-display",
                    "FAIL" if false_pin else "PASS" if good else "FAIL",
                    f"{ms:.0f} ms",
                    ("known false pin; " if false_pin else "")
                    + f"'where is my {label}' -> \"{res.get('text')}\" "
                    + f"(via {res.get('language')}); {pnote}; chose {meta['find_choice']}",
                )
            )
            # Standing next to it ends the guide as found at once; then ask for the next most-seen
            # thing (up to 3 more) so the guide gets a target across the room.
            tried = [label]
            rest = [k for k in sorted(counts, key=counts.get, reverse=True) if k != label]
            while rest and len(tried) < 4:
                found = False
                for _ in range(12):  # up to 6 s for a guide frame or the found signal
                    await asyncio.sleep(0.5)
                    after = c.since(t0, "safety")
                    if any(e.get("guide") for e in after):
                        break
                    if found := any(e.get("found") for e in after):
                        break
                if not found:
                    break
                tried.append(rest.pop(0))
                t0 = time.time()
                res, _ = await c.ask(f"where is my {tried[-1]}")
            gnote, g_false = pin_note(res)  # the answer the guide points from
            meta["guide_tried"] = tried
            # At least 10 s of guide frames, longer (up to 30 s) until 20 frames carried a guide.
            while (dt := time.time() - t0) < 30 and (
                dt < 10 or sum(e.get("guide") is not None for e in c.since(t0, "safety")) < 20
            ):
                await asyncio.sleep(0.5)
            meta["guide_hold_s"] = round(time.time() - t0, 1)
            guides = [e.get("guide") for e in c.since(t0, "safety") if e.get("guide") is not None]
            found_at_once = any(e.get("found") for e in c.since(t0, "safety"))
            with_b = [g for g in guides if g.get("bearing") is not None]
            ages = [e["map_age"] for e in c.since(t0, "safety") if e.get("map_age") is not None]
            pin_ts = (res.get("pin") or {}).get("ts")
            why = (
                f"map pose age median {statistics.median(ages):.1f} s (fresh < 6 s)"
                if ages
                else "no map pose at all"
            ) + (
                "; the pin is from the saved home map (seen before this run)"
                if isinstance(pin_ts, (int, float)) and pin_ts < t_start
                else "; the pin was seen in this run"
            )
            nav_wired = bool(re.search(r"from \.nav import|nav\.", src("metasight/server.py")))
            if not guides:
                results.append(
                    row(
                        "guide",
                        "Guide bearings after a find",
                        "metasight-compare",
                        "FAIL",
                        "0 frames",
                        "no safety frame carried a guide after the answer"
                        + (
                            "; the belt signalled found at once (target under 0.8 m)"
                            if found_at_once
                            else ""
                        )
                        + ("" if meta["find_far"] else "; no pin was 1.5 m or more away")
                        + f"; asked for {', '.join(tried)}",
                    )
                )
            else:
                b = with_b[len(with_b) // 2] if with_b else None
                results.append(
                    row(
                        "guide",
                        "Guide bearings after a find",
                        "metasight-compare",
                        "PASS" if with_b and not g_false else "FAIL",
                        f"{len(with_b)} of {len(guides)} frames",
                        ("known false pin; " if g_false else "")
                        + (
                            f"median frame: {b['bearing']} deg, {b['dist']} m, in view "
                            f"{b['in_view']}; asked for {', '.join(tried)}"
                            if b
                            else f"guide set but bearing null, not in view; {why}"
                        )
                        + f"; {gnote}",
                    )
                )
            route = [g for g in guides if g.get("route") is not None]
            if not nav_wired:
                results.append(
                    row(
                        "nav",
                        "Route planner (nav.py) drives the guide",
                        "metasight-compare",
                        "NOT-BUILT",
                        "",
                        "server.py does not call nav.py; the guide points in a straight line",
                    )
                )
            else:
                results.append(
                    row(
                        "nav",
                        "Route planner (nav.py) drives the guide",
                        "metasight-compare",
                        "PASS" if route and not g_false else "FAIL",
                        f"{len(route)} frames with a route",
                        ("known false pin; " if g_false else "")
                        + (f"guide frames carried a route; {gnote}" if route else "")
                        if route
                        else (
                            f"nav is wired, but no guide frame had a route ({len(guides)} "
                            f"guide frames in {meta['guide_hold_s']} s of guide mode)"
                            + ("" if with_b else ": the guide had no bearing (see the row above)")
                        ),
                    )
                )

        # ---- a weak, unconfirmed pin is refused, not answered (display: skipped_pin_id)
        t_k = time.time()
        res_k, ms_k = await c.ask("where are my keys")
        await asyncio.sleep(3)
        keys_guide = [
            e["guide"]
            for e in c.since(t_k, "safety")
            if e.get("guide") and e["guide"].get("label") == "keys"
        ]
        used = res_k.get("pin_id", (res_k.get("pin") or {}).get("id"))
        text_k = res_k.get("text") or ""
        never = "haven't seen" in text_k.lower() or "have not seen" in text_k.lower()
        bad = used in KNOWN_FALSE_PINS or any(g.get("pin") in KNOWN_FALSE_PINS for g in keys_guide)
        ok = res_k.get("skipped_pin_id") is not None and never and not keys_guide and not bad
        results.append(
            row(
                "falsepin",
                "False keys pin refused",
                "metasight-display",
                "PASS" if ok else "FAIL",
                f"{ms_k:.0f} ms",
                f"'where are my keys' -> \"{text_k}\"; skipped_pin_id "
                f"{res_k.get('skipped_pin_id')}, pin used {used}, "
                f"{len(keys_guide)} guide frames toward keys in 3 s",
            )
        )

        # ---- Find My fallback for an item never seen
        # Something the walk never saw: a look-alike candidate first, else one with no look-alike.
        unseen = next(
            (x for x in ("wallet", "remote control", "glasses case") if x not in meta["pins"]),
            "umbrella",
        )
        findmy_src = any("Find My" in p.read_text() for p in (ROOT / "metasight").rglob("*.py"))
        if unseen:  # always set: umbrella is the last resort
            res, ms = await c.ask(f"where is my {unseen}")
            text = res.get("text", "")
            status = "PASS" if "Find My" in text else ("FAIL" if findmy_src else "NOT-BUILT")
            results.append(
                row(
                    "falsepin",
                    "findmy",
                    "Find My line for an unseen item",
                    "metasight-muse",
                    status,
                    f"{ms:.0f} ms",
                    f"'where is my {unseen}' -> \"{text}\""
                    + ("" if findmy_src else "; no Find My line in metasight/"),
                )
            )

        # ---- tool path: what changed
        res, ms = await c.ask("what changed in the last ten minutes?")
        results.append(
            row(
                "changed",
                "Tool path: what changed",
                "metasight-muse",
                "PASS" if res.get("tools") else "FAIL",
                f"{ms:.0f} ms",
                f'"{res.get("text")}" (tools {bool(res.get("tools"))})',
            )
        )

        # ---- speaker cue (Talk mode, two voices from the sides) + who is talking
        await c.send({"type": "activity", "mode": "talk"})
        await asyncio.sleep(0.5)
        pcm2 = two_voices(work)
        t_a = time.time()

        async def who():
            await asyncio.sleep(5.0)  # voice 1 is talking
            return await c.ask("who is talking?")

        wt = asyncio.create_task(who())
        await c.audio(pcm2, 2, chunk=1600, tail_s=1.0)
        who_res, who_ms = await wt
        cues = [(at, e) for at, e in c.events if at >= t_a and e.get("type") == "speaker_cue"]
        sent = c.audio_sent
        lat = {}
        for onset, side in ((1.0, "left"), (10.0, "right")):
            i = int(onset * SR / 1600)
            if i < len(sent):
                hit = [(at, e) for at, e in cues if at >= sent[i] and e.get("turn") == side]
                if hit:
                    lat[side] = (1000 * (hit[0][0] - sent[i]), hit[0][1]["bearing"])
        people = c.since(t_a, "people")
        doas = [e["doa"] for e in people if e.get("doa") is not None]
        ok = len(lat) == 2
        results.append(
            row(
                "speaker",
                "Speaker cue (two voices, -50 and +40 deg)",
                "metasight-people",
                "PASS" if ok else "FAIL",
                ", ".join(f"{s} {v[0]:.0f} ms" for s, v in lat.items()) or f"{len(cues)} cues",
                (
                    "first cue per voice after its onset: "
                    + "; ".join(f"turn {s} at {v[1]} deg" for s, v in lat.items())
                    if lat
                    else "no turn cue"
                )
                + f"; {len(cues)} cues, {len(people)} people events, "
                f"doa range {min(doas, default=None)} to {max(doas, default=None)}",
            )
        )
        results.append(
            row(
                "who",
                "Tool path: who is talking",
                "metasight-muse",
                "PASS" if who_res.get("tools") else "FAIL",
                f"{who_ms:.0f} ms",
                f'"{who_res.get("text")}" (tools {bool(who_res.get("tools"))})',
            )
        )
        await c.send({"type": "activity", "mode": "guide"})
        await asyncio.sleep(2)

        # ---- a spoken question through Transcribe
        st = await c.state()
        if st.get("asr") is None:
            results.append(
                row(
                    "voice",
                    "Spoken question to answer",
                    "metasight-muse",
                    "FAIL",
                    "",
                    "voice service off on the private server (no META_API_KEY?)",
                )
            )
        elif label:
            t0 = time.time()
            spoken = await c.audio(speech(f"where is my {label}"), 1)
            a, at = await c.wait(t0, "answer", lambda e: "voice" in e, 15)
            heard = [e["text"] for e in c.since(t0, "transcript") if e.get("final")]
            results.append(
                row(
                    "voice",
                    "Spoken question to answer",
                    "metasight-muse",
                    "PASS" if a else "FAIL",
                    f"{a['voice']['speech_end_to_answer_ms']} ms" if a else "",
                    (
                        f'heard {heard[:1]}, answered "{a["text"]}"; client saw it '
                        f"{1000 * (at - spoken):.0f} ms after the last word left"
                        if a
                        else f"no voice answer; finals heard {heard}"
                    ),
                )
            )

        # ---- conversation notes
        t0 = time.time()
        await c.send({"type": "notes", "action": "start"})
        rec, _ = await c.wait(t0, "notes", lambda e: e.get("state") == "recording", 5)
        await c.audio(speech("The doctor said to take two pills at nine in the morning."), 1)
        await asyncio.sleep(1.5)
        lines = [e for e in c.since(t0, "transcript") if e.get("source") == "notes"]
        t1 = time.time()
        await c.send({"type": "notes", "action": "stop"})
        ready, at = await c.wait(t1, "notes", lambda e: e.get("state") == "ready", 30)
        q, qms = (await c.ask("what did the doctor say?")) if ready else ({}, 0)
        q_ok = bool(q.get("tools")) and bool(re.search(r"\b(nine|9|two|2)\b", q.get("text", "")))
        ok = bool(rec and rec.get("say") and lines and ready and ready.get("title") and q_ok)
        results.append(
            row(
                "notes",
                "Notes start, stop, summary, question",
                "metasight-people",
                "PASS" if ok else "FAIL",
                f"summary {1000 * (at - t1):.0f} ms" if ready else "",
                (
                    f"start line said: {bool(rec and rec.get('say'))}; {len(lines)} transcript "
                    f"lines; title \"{ready.get('title')}\"; 'what did the doctor say?' -> "
                    f'"{q.get("text")}"'
                    if ready
                    else f"recording {bool(rec)}, lines {len(lines)}, no ready event in 30 s"
                ),
            )
        )

        # ---- Wonder map review (every 2 min on this server)
        review, _ = await c.wait(t_start, "map_review", lambda e: True, 0.1)
        if review is None:
            left = 2 * 60 * 2 + 30 - (time.time() - t_start)
            review, _ = await c.wait(t_start, "map_review", lambda e: True, max(1, left))
        if review is None:
            results.append(
                row(
                    "wonder",
                    "Wonder map review",
                    "metasight-muse",
                    "FAIL",
                    "",
                    f"no map_review event in {time.time() - t_start:.0f} s at 2 min intervals",
                )
            )
        else:
            usd = review.get("usd")
            results.append(
                row(
                    "wonder",
                    "Wonder map review",
                    "metasight-muse",
                    "PASS" if review.get("items") else "FAIL",
                    f"{review.get('ms')} ms"
                    + (f", ${usd:.3f}" if isinstance(usd, (int, float)) else ""),
                    f"{review.get('changes')} changes, {len(review.get('items', []))} items "
                    f"by {review.get('model')}"
                    + (
                        f'; e.g. "{said[0]}"'
                        if (said := [i.get("say") for i in review.get("items", []) if i.get("say")])
                        else ""
                    ),
                )
            )

        c.stop_frames = True
        await asyncio.sleep(1)
        ft.cancel()

        # ---- reflex: stop cues and latency over every frame of the run
        safety = [(at, e) for at, e in c.events if e.get("type") == "safety"]
        # For scripts/score_walk.py (priors on vs off against the labels): the walk's own frame
        # seq, since the feed's seq keeps counting across passes; the last pass wins per frame.
        walk_seq = [
            json.loads(x)["seq"] for x in (walk / "frames.jsonl").read_text().splitlines() if x
        ]
        EVENTS.mkdir(parents=True, exist_ok=True)
        with (EVENTS / f"{walk.name}-safety.jsonl").open("w") as fh:
            for _, e in safety:
                fh.write(
                    json.dumps(
                        {**e, "feed_seq": e["seq"], "seq": walk_seq[e["seq"] % len(walk_seq)]}
                    )
                    + "\n"
                )
        meta["safety_events"] = len(safety)
        lat_all, lat_stop = [], []
        for at, e in safety:
            t_sent = c.sent.get(e.get("seq"))
            if t_sent is None or e.get("after_buzz_ms") is None:
                continue
            ms = 1000 * (at - t_sent) - e["after_buzz_ms"]
            lat_all.append(ms)
            if any(q["mode"] in STOP_MODES and q["intensity"] > 0 for q in e["cues"]):
                lat_stop.append(ms)
        belt_lat = [
            e["latency_ms"] for e in c.since(t_start, "belt") if e.get("latency_ms") is not None
        ]
        if not lat_all:
            results.append(
                row("stop", "Stop cue and its latency", "metasight-rust", "FAIL", "", "no safety")
            )
        else:
            p95 = pct(lat_stop or lat_all, 0.95)
            ok = bool(lat_stop) and p95 < 250
            results.append(
                row(
                    "stop",
                    "Stop cue and its latency",
                    "metasight-rust",
                    "PASS" if ok else "FAIL",
                    f"p95 {p95:.0f} ms",
                    f"{len(lat_stop)} stop frames of {len(lat_all)}; frame sent to belt command "
                    f"median {statistics.median(lat_all):.0f} ms, p95 {pct(lat_all, 0.95):.0f} ms"
                    + (
                        f"; server to emulated motor median {statistics.median(belt_lat):.0f} ms"
                        if belt_lat
                        else ""
                    )
                    + "; server at background priority, target < 250 ms",
                )
            )
        # How often the belt buzzes ahead. Baseline: the sector scheme with no priors has its three
        # front motors on 27 to 38% of the time on his walks (docs/haptics-sim.md, Tracker
        # section); a cue counts as on at 26/255, the skin model's floor (scripts/score_labels.py).
        results += cue_rows(safety, walk)
        mapped = sum(
            1
            for _, e in safety
            # `live` covers only the camera's sectors; a sector past its end is never seen live
            if any(
                s is not None and e["known"][i] and (i >= len(e["live"]) or e["live"][i] is None)
                for i, s in enumerate(e["sectors"])
            )
        )
        results.append(
            row(
                "mapreflex",
                "Map fills sectors the camera cannot see",
                "metasight-compare",
                "PASS" if mapped else "FAIL",
                f"{mapped} of {len(safety)} frames",
                "frames where a fresh map pose gave a distance in a sector the camera had none for",
            )
        )

        # ---- map pose and map size
        poses = c.since(t_start, "pose")
        st = await c.state()
        mem = c.since(t_start, "memory")
        infer = [e["infer_ms"] for e in mem if isinstance(e.get("infer_ms"), (int, float))]
        ok = bool(poses) and st["map_points"] > 0 and poses[-1].get("pos")
        results.append(
            row(
                "pose",
                "Map pose non-empty, map not black",
                "metasight-display",
                "PASS" if ok else "FAIL",
                f"{len(poses)} poses",
                f"{st['map_points']} map points, {len(st['pins'])} pins, last pos "
                f"{[round(v, 2) for v in poses[-1]['pos']] if poses else None}; "
                f"{len(mem)} map windows"
                + (f", median {statistics.median(infer):.0f} ms" if infer else ""),
            )
        )
        meta["seconds"] = round(time.time() - t_start)
        meta["frames_sent"] = len(c.sent)
        rd.cancel()
    return meta


# ---------------------------------------------------------------- report

ORDER = [
    "stop",
    "belt-",
    "falsecue",
    "urgent",
    "missnear",
    "mapreflex",
    "guide",
    "nav",
    "patterns",
    "find",
    "findmy",
    "voice",
    "voiceload",
    "changed",
    "who",
    "pose",
    "wonder",
    "speaker",
    "modes",
    "onboarding",
    "notes",
]


# Rows an uncommitted file can change (the private server imports the working tree). A dirty
# file under metasight/ not listed here may change any row.
DIRTY_ROWS = {
    "metasight/nav.py": ("guide", "nav", "falsecue", "urgent", "missnear", "priors"),
    "metasight/pan.py": ("falsecue", "urgent", "missnear", "priors", "stop"),
    "metasight/belt.py": ("falsecue", "urgent", "missnear", "priors", "stop", "belt-", "patterns"),
    "metasight/providers/transcribe.py": ("voice", "notes"),
    "metasight/voice.py": ("voice", "notes"),
    "metasight/providers/finder.py": ("find", "falsepin", "findmy", "guide", "nav", "voice"),
    "metasight/people.py": ("speaker", "who"),
    "metasight/audio_doa.py": ("speaker",),
    "metasight/notes.py": ("notes",),
    "metasight/review.py": ("wonder",),
    "metasight/onboarding.py": ("onboarding", "patterns"),
    "metasight/modes.py": ("modes", "patterns"),
}


KNOWN_FALSE_PINS = {100015}  # the saved house's "keys" pin is the keyboard (compare, 085c47b)
WEAK_PIN_SCORE = 0.3


def pin_note(res: dict) -> tuple[str, bool]:
    """Which pin an answer rested on, and whether it is a known false one: a listed pin, or one
    the server marks unconfirmed (`confirmed`: score at least 0.3 or a SAM 3.1 source; display
    2a9dfcf) that the camera did not see live. Older servers without the field: score under 0.3."""
    pin = res.get("pin") or {}
    pid = res.get("pin_id", pin.get("id"))
    if pid is None:
        return ("live camera, no pin" if res.get("live") else "no pin"), False
    confirmed = res.get("confirmed")
    if confirmed is None:
        confirmed = (pin.get("score") or 0) >= WEAK_PIN_SCORE
    bad = pid in KNOWN_FALSE_PINS or (not confirmed and not res.get("live"))
    return (
        f"pin {pid} {pin.get('label') or res.get('label')} score {pin.get('score') or 0:.2f} "
        f"source {pin.get('source') or '?'}, confirmed {confirmed}"
        f"{', live' if res.get('live') else ''}"
    ), bad


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    dirty = [
        ln[3:] for ln in git("status", "--short").splitlines() if ln[3:].startswith("metasight/")
    ]
    return {"head": git("rev-parse", "--short", "HEAD"), "dirty": dirty}


def flag_dirty(results: list, dirty: list[str]) -> None:
    """Mark each row an uncommitted metasight/ file could have changed."""
    for r in results:
        hit = [
            Path(f).name
            for f in dirty
            if f.endswith(".py")
            and (
                f not in DIRTY_ROWS
                or any(
                    r["key"] == k or (k.endswith("-") and r["key"].startswith(k)) or k in r["key"]
                    for k in DIRTY_ROWS[f]
                )
            )
        ]
        if hit:
            r["dirty"] = hit
            r["evidence"] += f" [uncommitted at start: {', '.join(hit)}]"


def write_report(results: list, meta: dict, walk: str, stamp: str, note: str) -> str:
    def rank(r):
        k = "belt-" if r["key"].startswith("belt") else r["key"]
        return ORDER.index(k) if k in ORDER else 99

    results.sort(key=rank)
    n = {s: sum(r["status"] == s for r in results) for s in ("PASS", "FAIL", "NOT-BUILT", "SKIP")}
    summary = f"{n['PASS']} pass, {n['FAIL']} fail, {n['NOT-BUILT']} not built" + (
        f", {n['SKIP']} unmeasurable" if n["SKIP"] else ""
    )
    earlier = []
    if DOC.exists():
        old = DOC.read_text()
        m = re.search(r"^Run (.+?)\. Result: (.+?)\.$", old, re.M)
        if m:
            earlier.append(f"- {m.group(1)}: {m.group(2)}")
        if "## Earlier runs" in old:
            earlier += [
                x for x in old.split("## Earlier runs", 1)[1].splitlines() if x.startswith("- ")
            ]
    cell = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
    out = [
        "# Morning check",
        "",
        f"Run {stamp}. Result: {summary}.",
        "",
        f"Code at start: HEAD `{meta['git']['head']}`, "
        + (
            "uncommitted under metasight/: "
            + ", ".join(f"`{f}`" for f in meta["git"]["dirty"])
            + f" ({sum(bool(r.get('dirty')) for r in results)} rows flagged)"
            if meta["git"]["dirty"]
            else "no uncommitted changes under metasight/"
        )
        + ".",
        "",
        "`scripts/feature_check.py` starts a private server on a free port (never the live one on "
        f"{LIVE_PORT}), with its own notes, sessions and a copy of the home map, at background "
        f"priority. It streams the recorded walk `data/walks/{walk}` at real speed "
        f"({meta.get('frames_sent', 0)} frames, {meta.get('seconds', 0)} s in all), then speaks to "
        "it with macOS `say` and the two-voice track from `bench/people/make_two_voices.py`. "
        "Every result below comes from the server's own events and `/api/ask` / `/api/state`. "
        "Raw results: `bench/results/morning-check.json`.",
        "",
        "Differences from the live profile: map review every 2 min (not 10), SAM 3.1 crops every "
        "10 s (not 2), its own belt emulator port.",
        "",
        "The stop latency here is the private server launched at background priority while the "
        "live service and the other agents' jobs share the Mac. Since 6ab0b5a the server moves "
        "its safety and stop path out of the background band on start (memory, solver, model "
        "and review threads stay in it), but the reflex thread still runs at nice 19 here, "
        "unlike live. The live service at normal priority measured frame to belt command p95 "
        "73 ms on ab3877b (`docs/perf.md`, `bench/results/perf/cue-service-ab3877b.json`). The "
        "target is under 250 ms.",
        "",
        "Two clocks on the spoken question. The Number is the server's own "
        "`speech_end_to_answer_ms`: from where Muse Voice Transcribe puts the end of speech "
        "(`audioProcessedMs` mapped onto the Mac's clock) to the answer leaving. The evidence "
        "also gives the client's time, from the moment the last chunk of the `say` clip was sent "
        "to the answer arriving. The client number is the smaller one, because the clip's own "
        "trailing silence (about 0.2 s after the last word, measured) is already gone before "
        "its clock starts, and the server's audio-to-wall-clock mapping can place the end of "
        "speech a little earlier still. The server number is the conservative one and matches "
        "how `bench/results/voice-e2e.json` is reported.",
        "",
    ]
    if note:
        out += [note, ""]
    out += ["| Feature | Owner | Result | Number | Evidence |", "|---|---|---|---|---|"]
    for r in results:
        out.append(
            f"| {cell(r['feature'])} | {r['owner']} | **{r['status']}** | {cell(r['number'])} | "
            f"{cell(r['evidence'])} |"
        )
    if earlier:
        out += ["", "## Earlier runs", "", *earlier[:20]]
    DOC.write_text("\n".join(out) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--walk", default="20261028-135040-iphone")
    ap.add_argument("--work", default="", help="scratch folder (default: a new temp folder)")
    a = ap.parse_args()
    if a.serve:
        serve(a.serve)
        return
    walk = ROOT / "data" / "walks" / a.walk
    stamp = time.strftime("%Y-%m-%d %H:%M %Z")
    tree = git_state()  # the private server imports the working tree as it is now
    if why := preflight():
        print("skipped:", why)
        sys.exit(2)
    work = Path(a.work or tempfile.mkdtemp(prefix="feature-check-"))
    work = work / time.strftime("%Y%m%d-%H%M%S")
    for d in ("sessions", "walks"):
        (work / d).mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "data" / "home", work / "home")
    port, belt_port = free_port(socket.SOCK_STREAM), free_port(socket.SOCK_DGRAM)
    assert port != LIVE_PORT
    env = {
        **os.environ,
        "METASIGHT_CONFIG": str(private_config(work, belt_port)),
        "METASIGHT_FINDER": "hybrid",
        "METASIGHT_HOME": str(work / "home"),
        "METASIGHT_WALKS": str(work / "walks"),
        "FEATURE_CHECK_SESSIONS": str(work / "sessions"),
        "PATH": "/opt/homebrew/bin:" + os.environ.get("PATH", "/usr/bin:/bin"),
    }
    slot = heavy_slot()
    results: list = []
    meta: dict = {}
    note = ""
    log = open(work / "server.log", "w")
    proc = subprocess.Popen(
        ["nice", "-n", "19", "taskpolicy", "-b", sys.executable, __file__, "--serve", str(port)],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"private server pid {proc.pid} on :{port}, work {work}", flush=True)
    try:
        c = Client(f"ws://127.0.0.1:{port}/ws", f"http://127.0.0.1:{port}")
        t_up = time.time()
        up = t_up + 600  # at background priority the map models take minutes to load
        http_up = vload_tried = False
        while time.time() < up:
            if proc.poll() is not None:
                raise RuntimeError("server exited at start")
            try:
                ready = asyncio.run(c.state())["memory_ready"]
                http_up = True
                if not vload_tried and not ready:
                    vload_tried = True
                    try:
                        results.append(asyncio.run(asyncio.wait_for(voice_before_map(c), 60)))
                    except Exception as e:
                        results.append(
                            row(VLOAD, VLOAD_NAME, "metasight-muse", "FAIL", "", f"{e!r}")
                        )
                    print(results[-1]["status"], results[-1]["evidence"], flush=True)
                if ready:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f"memory not ready in 600 s (http answered: {http_up})")
        meta["ready_s"] = round(time.time() - t_up)
        print(f"server ready after {meta['ready_s']} s", flush=True)
        meta.update(asyncio.run(asyncio.wait_for(run_checks(c, walk, work, results), 900)))
    except Exception as e:
        tail = (work / "server.log").read_text()[-600:].replace("\n", " ")
        note = f"The run stopped early: {type(e).__name__}: {e}. Server log tail: `{tail}`"
        print(note, flush=True)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # uvicorn shuts down: belt emulator, memory worker
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        log.close()
        shutil.rmtree(slot, ignore_errors=True)
    have = {r["key"] for r in results}
    for key in ORDER:
        if key != "belt-" and key not in have and note:
            results.append(row(key, key, "metasight-verify", "FAIL", "", "check did not run"))
    events = EVENTS / f"{a.walk}-safety.jsonl"
    if events.exists():  # priors on vs off, read by belt_check's belt_priors_labels row
        sw = subprocess.run(
            ["nice", "-n", "19", "taskpolicy", "-b", sys.executable, "scripts/score_walk.py"]
            + [a.walk, "--events", str(events)],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=600,
        )
        meta["score_walk"] = (sw.stdout + sw.stderr).strip()[-400:]
    try:  # belt firmware checks (metasight-braille's scripts/belt_check.py), no server needed
        sys.path.insert(0, str(ROOT / "scripts"))
        from belt_check import run as belt_rows

        for key, ok, ev in belt_rows():
            name = ev.split(":")[0]
            if "SKIP" in ev:  # a check that could not run is not a pass
                results.append(
                    row(
                        f"belt-{key}",
                        name,
                        "metasight-braille",
                        "NOT-BUILT",
                        "",
                        ev[len(name) + 1 :].strip(),
                    )
                )
                continue
            results.append(
                row(
                    f"belt-{key}",
                    name,
                    "metasight-braille",
                    "PASS" if ok else "FAIL",
                    "",
                    ev[len(name) + 1 :].strip(),
                )
            )
    except ImportError as e:
        results.append(
            row(
                "belt",
                "Belt firmware checks",
                "metasight-braille",
                "NOT-BUILT",
                "",
                f"scripts/belt_check.py: {e}",
            )
        )
    meta["git"] = tree
    flag_dirty(results, tree["dirty"])
    summary = write_report(results, meta, a.walk, stamp, note)
    RESULTS.write_text(
        json.dumps(
            {"at": stamp, "walk": a.walk, "meta": meta, "results": results, "note": note}, indent=1
        )
    )
    JARVIS_LOG.mkdir(parents=True, exist_ok=True)
    with open(JARVIS_LOG / f"{time.strftime('%Y-%m-%d')}.md", "a") as f:
        f.write(
            f"- {time.strftime('%H:%M')} metasight-verify: morning check {summary} "
            "(meta-sight/docs/MORNING-CHECK.md)\n"
        )
    for r in results:
        print(f"{r['status']:9} {r['feature']}: {r['number']} | {r['evidence']}")
    print(summary)


if __name__ == "__main__":
    main()
