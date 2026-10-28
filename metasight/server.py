"""The Mac mini server: frames and device motion in over a WebSocket, safety cues and the house
map out. FastAPI for HTTP and the WebSocket, one worker process for the memory loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import multiprocessing as mp
import struct
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .align import map_resolution
from .belt import FLAG_FOUND, NAMES, BeltLink, guide, plan
from .config import ROOT, Settings
from .geometry import CAMERA_SECTORS, SECTORS, merge_sectors
from .heading import Heading, wrap
from .hub import Hub
from .memory_worker import run as memory_run
from .motion import FallDetector, Sample, ShakeDetector
from .pins import PinStore
from .providers import make_language
from .providers.finder import LOOKALIKES
from .query import (
    SYSTEM,
    facts_for_language,
    parse_query,
    template_answer,
    valid_sentence,
)
from .safety import SafetyLoop, hazards
from .walk import WalkRecorder

MAX_PENDING_KEYFRAMES = 12
SIGHT_AGREE_DEG = 50.0  # after a lock, a sighting further than this from it is another object
SIGHT_HOLD_S = 6.0  # how long a lock vets new sightings
POSE_FRESH_S = 6.0  # a map pose older than this gives no direction
FINDER_LONG_SIDE = 960  # small items vanish at the models' 518 px; the item finder looks here
log = logging.getLogger("metasight")


def _finite(x: float) -> float | None:
    return round(x, 2) if math.isfinite(x) else None


class Server:
    def __init__(self, settings: Settings, session: str):
        self.s = settings
        self.session = session
        self.hub = Hub()
        self.pins = PinStore()
        self.language = make_language(settings)
        self.safety: SafetyLoop | None = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="safety")
        self.llm_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")
        self.fall, self.shake = FallDetector(), ShakeDetector()
        self.motion_t = 0.0
        self.accel: deque[np.ndarray] = deque(maxlen=40)
        self.pose = None
        self.source_ws = None  # the client sending camera frames (phone, glasses app, browser)
        self.profile_hfov = settings.hfov_deg
        self.lenses: dict[WebSocket, float] = {}  # field of view a client measured on its camera
        self.walks: dict[WebSocket, WalkRecorder] = {}  # clients saving their walk on this Mac
        self.guide: dict | None = None  # the item the belt is pointing to, after a question
        self.found_until = 0.0
        self.heading = Heading(settings.hfov_deg)  # carries the head's turn between map poses
        self.map_sectors: list[float] = [math.inf] * 8
        self.map_pose_ts = 0.0
        self.map_yaw = 0.0  # heading the map sectors were measured along
        self.trail: list[list[float]] = []
        self.latest: tuple[float, int, bytes] | None = None
        self.latest_arrived = 0.0  # monotonic time the latest camera frame reached the server
        self.belt = BeltLink(settings.belt, settings.belt_addr, self._belt_state)
        self.frame_ready = asyncio.Event()
        self.pending_taps: list[tuple[str, float, float]] = []
        self.pending_kf = 0
        self.last_kf = 0.0
        self.kf_best: dict | None = None  # sharpest frame so far in this keyframe interval
        self.fps = 0.0
        self.mode = settings.mode
        self.last_done = 0.0
        self.memory_ready = False
        self.memory_stats: dict = {}
        self.ctx = mp.get_context("spawn")
        self.in_q = self.ctx.Queue()
        self.out_q = self.ctx.Queue()
        self.proc: mp.Process | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.safety = await self.loop.run_in_executor(self.pool, SafetyLoop, self.s)
        self.proc = self.ctx.Process(
            target=memory_run, args=(self.in_q, self.out_q, self.s, self.session), daemon=True
        )
        self.proc.start()
        await self.belt.start()
        threading.Thread(target=self._read_memory, daemon=True).start()
        self._frame_task = asyncio.create_task(self.frame_guard())

    def use_source(self, sock: WebSocket) -> None:
        """Frames now come from `sock`: use the lens it reported (an iPhone measures its own),
        else the profile's (the glasses report none)."""
        self.source_ws = sock
        hfov = self.lenses.get(sock, self.profile_hfov)
        if hfov == self.s.hfov_deg:
            return
        self.s.hfov_deg = hfov  # the safety loop reads it from these settings every frame
        self.heading.hfov_deg = hfov
        self.in_q.put({"cmd": "hfov", "hfov": hfov})
        log.info("camera lens now %.1f deg horizontal", hfov)

    def on_camera(self, sock: WebSocket, m: dict) -> None:
        """A client describes its camera, and whether to save its walk here."""
        hfov = m.get("hfov")
        if isinstance(hfov, (int, float)) and 20 <= hfov <= 150:
            self.lenses[sock] = round(float(hfov), 2)
        else:
            self.lenses.pop(sock, None)
        log.info(
            "camera from %s: hfov %s, %sx%s, walk %s",
            m.get("source", "?"),
            hfov,
            m.get("width"),
            m.get("height"),
            m.get("walk"),
        )
        if sock is self.source_ws:
            self.use_source(sock)
        old, walk = self.walks.pop(sock, None), m.get("walk")
        if old and old.id == walk:
            old.camera(m)
            self.walks[sock] = old
            return
        if old:
            old.close()
        if walk:
            try:
                self.walks[sock] = WalkRecorder(str(walk), m)
            except ValueError:
                log.warning("ignored walk id %r", walk)

    def drop_client(self, sock: WebSocket) -> None:
        self.hub.clients.discard(sock)
        self.lenses.pop(sock, None)
        if walk := self.walks.pop(sock, None):
            walk.close()

    def _belt_state(self, msg: dict) -> None:
        asyncio.ensure_future(self.hub.json(msg))

    def stop(self) -> None:
        self.belt.stop()
        self.in_q.put({"cmd": "stop"})
        if self.proc:
            self.proc.join(timeout=15)
            if self.proc.is_alive():
                self.proc.terminate()

    def _read_memory(self) -> None:
        while True:
            try:
                ev = self.out_q.get()
            except (EOFError, OSError):
                return
            asyncio.run_coroutine_threadsafe(self.on_memory_event(ev), self.loop)

    # ------------------------------------------------------------ memory events
    async def on_memory_event(self, ev: dict) -> None:
        t = ev["type"]
        if t == "ready":
            self.memory_ready = True
            await self.hub.json({"type": "status", "memory": "ready", "finder": ev["finder"]}, True)
        elif t == "map_replace":
            await self.hub.map_replace(ev["n"], ev["points"], ev["colors"], ev.get("sizes", b""))
            if ev["n"] == 0:
                self.trail = []
                await self.hub.json({"type": "trail", "points": []}, True)
        elif t == "mode":
            self.mode = ev["mode"]
            await self.hub.json({"type": "mode", "mode": self.mode}, True)
        elif t == "map_chunk":
            await self.hub.map_chunk(ev["n"], ev["points"], ev["colors"])
        elif t == "trail_replace":
            self.trail = ev["points"]
            await self.hub.json({"type": "trail", "points": self.trail}, True)
        elif t == "trail":
            self.trail += ev["points"]
            await self.hub.json({"type": "trail", "points": self.trail}, True)
        elif t == "pins":
            self.pins.replace_all(ev["pins"])
            await self.hub.json({"type": "pins", "pins": ev["pins"]}, True)
        elif t == "pose":
            pose = np.eye(4)
            pose[:3, 3], pose[:3, 2] = ev["pos"], ev["fwd"]
            pose[:3, 1] = -np.array(ev["up"])
            pose[:3, 0] = np.cross(pose[:3, 1], pose[:3, 2])
            self.pose = pose
            self.heading.anchor(pose[:3, 3], pose[:3, 2], ev.get("ts"))
            self.map_sectors = [math.inf if d is None else d for d in ev["map_sectors"]]
            self.map_yaw = math.atan2(ev["fwd"][0], ev["fwd"][2])
            self.map_pose_ts = time.time()
            out = {"type": "pose", "pos": ev["pos"], "fwd": ev["fwd"], "up": ev["up"]}
            out.update({k: ev[k] for k in ("hfov", "aspect") if k in ev})
            await self.hub.json(out, True)
        elif t == "scale_hint" and self.safety:
            self.safety.apply_scale_hint(ev["ratio"])
        elif t == "memory_stats":
            self.pending_kf = max(0, self.pending_kf - ev.get("consumed", 0))
            self.memory_stats = ev
            await self.hub.json({**ev, "type": "memory"}, True)
        elif t == "memory_error":
            self.pending_kf = max(0, self.pending_kf - ev.get("consumed", 0))
            await self.hub.json({"type": "status", "memory_error": ev["error"]}, True)

    # ------------------------------------------------------------ frames
    def _process(self, ts: float, seq: int, jpeg: bytes):
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        w, h = map_resolution(bgr.shape[1], bgr.shape[0])
        small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        gray = self.heading.shift_gray(small)
        sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        k = FINDER_LONG_SIDE / max(bgr.shape[:2])
        hi = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else bgr
        return small, self.safety.process(small), gray, sharp, hi

    async def frame_guard(self) -> None:
        """Keep the frame loop alive: log a failure, then carry on with the next frame."""
        while True:
            try:
                await self.frame_loop()
            except Exception:
                log.exception("frame loop failed")
                await asyncio.sleep(0.2)

    async def frame_loop(self) -> None:
        while True:
            await self.frame_ready.wait()
            self.frame_ready.clear()
            if self.latest is None:
                continue
            ts, seq, jpeg = self.latest
            arrived = self.latest_arrived
            done = await self.loop.run_in_executor(self.pool, self._process, ts, seq, jpeg)
            if done is None:
                continue
            small, res, gray, sharp, hi = done
            now = time.time()
            self.heading.image_turn(gray, now)
            if self.last_done:
                inst = 1.0 / max(1e-3, now - self.last_done)
                self.fps = inst if self.fps == 0 else 0.85 * self.fps + 0.15 * inst
            self.last_done = now
            live = res.sectors
            merged = merge_sectors(live, self._map_sectors_now())
            sc = res.scene
            # Side and back sectors come from the map; trust them only while the pose is fresh.
            map_fresh = self.memory_ready and self.map_pose_ts and now - self.map_pose_ts < 3.0
            known = [n in CAMERA_SECTORS or bool(map_fresh) for n in SECTORS]
            cmd, flags = plan(merged, known, sc.path_m, res.ttc_s, sc.drop_m)
            # The heading is carried between poses, so guidance tolerates an older position.
            guide_fresh = self.memory_ready and now - self.map_pose_ts < POSE_FRESH_S
            guiding = self._guide(cmd, now, bool(guide_fresh), res.obstacles)
            if now < self.found_until:
                flags |= FLAG_FOUND
            self.belt.send(cmd, flags, arrived)
            await self.hub.frame(seq, jpeg, self.source_ws)
            await self.hub.json(
                {
                    "type": "safety",
                    "seq": seq,
                    "yaw": self.heading.yaw,
                    "ts": ts,
                    "sectors": [_finite(d) for d in merged],
                    "names": SECTORS,
                    "live": [_finite(d) for d in live],
                    "known": known,
                    "cues": [{"mode": NAMES[m], "intensity": i / 255} for m, i in cmd],
                    "hazards": hazards(sc.grid, sc.corridor),
                    "path": _finite(sc.path_m),
                    "trip": _finite(sc.trip_m),
                    "drop": _finite(sc.drop_m),
                    "ttc": _finite(res.ttc_s),
                    "floor": sc.floor,
                    "floor_scale": res.floor_scale,
                    "eye_h": self.s.eye_height_m,
                    "guide": guiding,
                    "found": now < self.found_until,
                    "obstacles": [
                        {
                            "label": o.label,
                            "score": round(o.score, 2),
                            "box": [round(v, 4) for v in o.box],
                            "dist": _finite(o.dist_m),
                        }
                        for o in res.obstacles
                    ],
                    "objects": [
                        {
                            "label": o.label,
                            "score": round(o.score, 2),
                            "poly": o.poly,
                            "box": [round(v, 4) for v in o.box],
                            "dist": _finite(o.dist_m),
                        }
                        for o in res.objects
                    ],
                    "fps": round(self.fps, 1),
                    "ms": {k: round(v, 1) for k, v in res.timings_ms.items()},
                    "scale": round(self.safety.scale, 3),
                    "map_age": round(now - self.map_pose_ts, 1) if self.map_pose_ts else None,
                }
            )
            self._maybe_keyframe(now, seq, small, res, sharp, hi)

    def _map_sectors_now(self) -> list[float]:
        """The map's sectors turned with the head since their keyframe: turning left by 45
        degrees moves what was front-left to the front."""
        yaw = self.heading.yaw
        if yaw is None:
            return self.map_sectors
        k = round(math.degrees(wrap(yaw - self.map_yaw)) / 45.0) % 8
        return [self.map_sectors[(i - k) % 8] for i in range(8)]

    def _guide(
        self, cmd: list[tuple[int, int]], now: float, map_fresh: bool, seen: list
    ) -> dict | None:
        """After "where is my X", point the belt at X until you reach it or 90 s pass. When X is
        in view the camera gives its direction and distance (and corrects the map heading);
        otherwise the map does, but only with a fresh pose: without one the direction would be a
        guess, so the belt stays quiet."""
        g = self.guide
        if g is None or now > g["until"]:
            self.guide = None
            return None
        if "want" in g and self.pins.latest(g["want"]) is not None:
            # Memory confirmed the real item after the look-alike answer: say so, guide there.
            self.guide = None
            self.loop.create_task(self.answer(f"where is my {g['want']}"))
            return None
        pin = self.pins.pins.get(g["pin"])
        rel = self.heading.to(np.array(pin.pos)) if pin is not None and map_fresh else None
        live = [
            (self.heading.box_bearing(o.box), o.dist_m)
            for o in seen
            if (o.label == g["label"] or g["label"] in LOOKALIKES.get(o.label, ()))
            and math.isfinite(o.dist_m)
        ]
        if rel is not None and now - g.get("seen", -math.inf) < SIGHT_HOLD_S:
            # Locked on recently, so the heading agrees with the camera: another chair in view
            # is not the one asked about. Before the first lock the map's heading can be far off
            # and the camera wins.
            live = [b for b in live if abs(b[0] - rel[1]) < SIGHT_AGREE_DEG]
        if live:
            near = (lambda b: abs(b[0] - rel[1])) if rel else (lambda b: b[1])
            bearing, dist = min(live, key=near)
            g["seen"] = now
            if pin is not None:
                self.heading.sight(np.array(pin.pos), bearing)
            rel = dist, bearing
        if rel is None:
            return {"label": g["label"], "dist": None, "bearing": None}
        dist, bearing = rel
        if dist < 0.8:
            self.guide, self.found_until = None, now + 2.0
            return None
        guide(cmd, bearing, dist)
        return {"label": g["label"], "dist": round(dist, 2), "bearing": round(bearing, 1)}

    def _maybe_keyframe(self, now: float, seq: int, small, res, sharp: float, hi) -> None:
        """Send memory the sharpest frame of each keyframe interval. Walking and turning blur
        most frames; a bottle on a desk walked past can be sharp in only one of them, and the
        map stitches better from sharp views too."""
        if not self.memory_ready:
            return
        if self.kf_best is None or sharp > self.kf_best["sharp"]:
            g = None
            if self.accel:
                gd = np.mean(self.accel, axis=0)  # device axes: x right, y up, z out of screen
                g = np.array([gd[0], -gd[1], -gd[2]])  # -> OpenCV camera axes (rear camera)
            self.kf_best = {
                "sharp": sharp,
                "cmd": "kf",
                "seq": seq,
                "ts": now,
                "bgr": small,
                "hi": hi,
                "depth": res.depth,
                "depth_scale": self.safety.scale,
                "gravity": g,
            }
        hz = self.s.outside["keyframe_hz"] if self.mode == "outside" else self.s.keyframe_hz
        if now - self.last_kf < 1.0 / hz:
            return
        self.last_kf = now
        kf, self.kf_best = self.kf_best, None
        if self.pending_kf >= MAX_PENDING_KEYFRAMES:
            return
        kf.pop("sharp")
        kf["taps"], self.pending_taps = self.pending_taps, []
        self.pending_kf += 1
        self.in_q.put(kf)

    # ------------------------------------------------------------ motion
    async def on_motion(self, samples: list[list[float]]) -> None:
        now = time.time()
        for row in samples:
            t, ax, ay, az = row[:4]
            s = Sample(t, ax, ay, az)
            dt = t - self.motion_t
            self.motion_t = t
            self.accel.append(np.array([ax, ay, az]))
            if len(row) >= 7:  # gyro, rad/s, device axes: x right, y up, z toward the face
                up = np.mean(self.accel, axis=0)
                up = up / (np.linalg.norm(up) or 1.0)
                up = up if up[1] >= 0 else -up  # either accelerometer sign: up is the +y side
                self.heading.gyro(np.array(row[4:7]), up, dt, now)
            for ev in self.fall.update(s) + self.shake.update(s):
                await self.hub.json({"type": "alert", "kind": ev.kind, "message": ev.message})

    # ------------------------------------------------------------ questions
    def _answer_sync(self, text: str) -> dict:
        label = parse_query(text, self.pins.labels())
        if label is None and self.pins.labels() and self.language.name != "template":
            try:
                pick = (
                    self.language.complete(
                        "Reply with exactly one item from the list, or none.",
                        f"Items: {', '.join(self.pins.labels())}\nQuestion: {text}",
                        max_tokens=8,
                    )
                    .strip()
                    .lower()
                )
                label = pick if pick in self.pins.labels() else None
            except Exception:
                label = None
        if label is None:
            return {
                "type": "answer",
                "label": None,
                "text": "Which object do you mean?",
                "pin": None,
            }
        pin = self.pins.latest(label)
        stand_in = False
        if pin is None:
            # The careful detector can miss a motion-blurred item the quick one saw as its
            # look-alike; offer that, and say it is unconfirmed.
            alike = [
                p
                for p in self.pins.pins.values()
                if label in LOOKALIKES.get(p.label, ()) and (p.seen >= 2 or p.score >= 0.25)
            ]
            if alike:
                pin, stand_in = max(alike, key=lambda p: (p.seen, p.score)), True
        now = time.time()
        # A stale pose gives a confident wrong direction; without one the answer names the place.
        pose = self.pose if now - self.map_pose_ts < POSE_FRESH_S else None
        text_out = template_answer(label, pin, pose, now, stand_in)
        polished = False
        if pin is not None and not stand_in and self.language.name != "template":
            try:
                phrased = self.language.complete(SYSTEM, facts_for_language(label, pin, pose, now))
                if valid_sentence(phrased):
                    text_out, polished = phrased.strip(), True
            except Exception:
                pass
        if pin is not None:
            self.guide = {"pin": pin.id, "label": pin.label, "until": now + 90.0}
            if stand_in:
                self.guide["want"] = label
        return {
            "type": "answer",
            "label": label,
            "text": text_out,
            "pin": pin.to_json() if pin else None,
            "language": self.language.name if polished else "template",
        }

    async def answer(self, text: str) -> dict:
        res = await self.loop.run_in_executor(self.llm_pool, self._answer_sync, text)
        await self.hub.json(res)
        return res


class Ask(BaseModel):
    text: str


def create_app(settings: Settings | None = None, session: str | None = None) -> FastAPI:
    settings = settings or Settings.load()
    session = session or time.strftime("%Y%m%d-%H%M%S")
    srv = Server(settings, session)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await srv.start()
        yield
        srv.stop()

    app = FastAPI(lifespan=lifespan, title="meta-sight")
    app.state.srv = srv

    @app.get("/api/state")
    async def state():
        return JSONResponse(
            {
                "memory_ready": srv.memory_ready,
                "language": srv.language.name,
                "finder": settings.finder,
                "fps": round(srv.fps, 1),
                "pins": [p.to_json(False) for p in srv.pins.pins.values()],
                "mode": srv.mode,
                "map_points": srv.hub.map_total,
                "memory": srv.memory_stats,
            }
        )

    @app.post("/api/ask")
    async def ask(body: Ask):
        return await srv.answer(body.text)

    @app.post("/api/test_fall")
    async def test_fall():
        for ev in srv.fall.trigger_test(srv.motion_t):
            await srv.hub.json({"type": "alert", "kind": ev.kind, "message": ev.message})
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await srv.hub.join(sock)
        try:
            while True:
                msg = await sock.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    data = msg["bytes"]
                    ts, seq = struct.unpack("<dI", data[:12])
                    srv.latest = (ts, seq, data[12:])
                    srv.latest_arrived = time.monotonic()
                    if sock is not srv.source_ws:
                        srv.use_source(sock)
                    if walk := srv.walks.get(sock):
                        walk.frame(ts, seq, data[12:])
                    srv.frame_ready.set()
                elif msg.get("text"):
                    m = json.loads(msg["text"])
                    kind = m.get("type")
                    if kind == "motion":
                        await srv.on_motion(m["s"])
                    elif kind == "ask":
                        asyncio.create_task(srv.answer(m["text"]))
                    elif kind == "camera":
                        srv.on_camera(sock, m)
                    elif kind == "arpose":  # the phone's own tracked pose, kept with its walk
                        if walk := srv.walks.get(sock):
                            walk.pose(m)
                    elif kind == "mode" and m.get("mode") in ("home", "outside"):
                        srv.in_q.put({"cmd": "mode", "mode": m["mode"]})
                    elif kind == "tap":
                        srv.pending_taps.append((m["label"], float(m["x"]), float(m["y"])))
                    elif kind in ("device", "transcript"):  # glasses state, live captions
                        await srv.hub.json(m, kind == "device")
                    elif kind == "fall_ok":
                        for ev in srv.fall.respond_ok(srv.motion_t):
                            await srv.hub.json(
                                {"type": "alert", "kind": ev.kind, "message": ev.message}
                            )
        except WebSocketDisconnect:
            pass
        finally:
            srv.drop_client(sock)

    dist = ROOT / "web" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cert", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--session", default=None)
    a = ap.parse_args()
    cert = Path(a.cert) if a.cert else None
    uvicorn.run(
        create_app(session=a.session),
        host=a.host,
        port=a.port,
        ssl_certfile=str(cert) if cert else None,
        ssl_keyfile=a.key,
        log_level="info",
    )


if __name__ == "__main__":
    main()
