"""The Mac mini server: frames and device motion in over a WebSocket, safety cues and the house
map out. FastAPI for HTTP and the WebSocket, one worker process for the memory loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import multiprocessing as mp
import socket
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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import band, nav, roll_call
from . import home as homemod
from .align import map_resolution
from .belt import FLAG_FOUND, NAMES, BeltLink, RhythmHold, plan, strong
from .config import ROOT, SESSIONS, Settings, secret
from .conversation import Talk
from .fovmem import FovLoop
from .geometry import CAMERA_SECTORS, SECTORS, grid_strips, merge_sectors
from .haptics import make_output
from .heading import Heading, wrap
from .home import cells as occupied_cells
from .hub import Hub
from .memory_worker import run as memory_run
from .modes import ALL as ACTIVITIES
from .modes import Activity
from .modes import command as activity_command
from .modes import implied as implied_activity
from .motion import FallDetector, Sample, ShakeDetector
from .notes import Notes, asks_notes, caption, notes_command
from .onboarding import PATTERNS, Onboarding, pattern_length, wants_setup
from .pan import GUIDE_ALIGNED_DEG, Median, Params, Persist, plan_pan, scene_targets, with_guide
from .pins import PinStore, confirmed, layer_of
from .providers import make_language, make_reviewer
from .providers.finder import LOOKALIKES
from .providers.tools import Tools, agent_answer
from .query import (
    SYSTEM,
    asked_for,
    camera_answer,
    facts_for_language,
    glimpse_answer,
    is_errand,
    live_answer,
    needs_tools,
    parse_query,
    template_answer,
    unseen_answer,
    valid_sentence,
)
from .relocalize import RoomMatcher
from .review import MapReview
from .review import plan as errand_plan
from .roomprior import RoomPrior
from .safety import SafetyLoop, hazards
from .voice import VoiceAsk
from .walk import WALKS, WalkRecorder

MAX_PENDING_KEYFRAMES = 12
SIGHT_AGREE_DEG = 50.0  # after a lock, a sighting further than this from it is another object
SIGHT_HOLD_S = 6.0  # how long a lock vets new sightings
POSE_FRESH_S = 6.0  # a map pose older than this gives no direction
# The guide still points from the last pose this long. With the phone's ARKit the position
# itself is tracked, so a minute is fine; with only a memory window's pose (the glasses) the
# heading follows the gyro and the camera's turn, but the walker's steps since are missing.
GUIDE_POSE_S = 60.0
GUIDE_DR_S = 15.0
NO_DIRECTION = "Turn slowly, I'll buzz when you face it."
ANSWER_POSE_S = 12.0  # answers accept an older pose: windows take 5 to 11 s during a solve
LIVE_SEEN_S = 1.0  # a camera sighting this recent answers "where is my X" over memory
LIVE_HOLD_S = 3.0  # ...and one this recent still counts as in view (the detector blinks)
LIVE_OVER_PIN_S = 60.0  # an older camera sighting wins when memory's pin is this much older
MAP_WAIT_S = 3.0  # right after a restart, how long a question waits for the home map
MAP_LOADING = "I'm still loading your home map. Ask me again in a few seconds."
HOLDING = "One sec, putting that together."  # spoken at once for a plan, which takes 2 s or more
AR_POSE_HZ = 5.0  # ARKit-carried poses go out to viewers at most this often
ROOM_VIEW_HZ = 5.0  # frames placed in the fixed room by PnP, for the room overlay
# Fewer PnP inliers than this and a frame can land on a look-alike wall: on the bedroom replay a
# keyframe with 191 landed 1.6 m off, while single frames with 230 or more were within 16 cm.
ROOM_VIEW_MIN_INLIERS = 220
ROOM_PRIOR_AGE_S = 0.3  # an older room pose has turned with the head too far to vouch for a box
FLOOR_EVERY_S = 2.0  # the floor plan for routes and priors is rebuilt from the map this often
RETIE_M = 0.3  # the pose jumping this far in one frame is a re-tie: the priors sit out a second
# ARKit's camera is the landscape sensor (x along its long side, y up, z back). Frames go out
# turned upright, so the image's right is ARKit's +y and its down is ARKit's +x.
AR_TO_CV = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)


def arkit_pose(m: list[float]) -> np.ndarray:
    """The phone's column-major camera-to-world matrix, in the upright image's OpenCV axes."""
    return np.array(m, dtype=float).reshape(4, 4).T @ AR_TO_CV


FINDER_LONG_SIDE = 960  # small items vanish at the models' 518 px; the item finder looks here
log = logging.getLogger("metasight")


def _home_bgr(home: homemod.Home) -> list[np.ndarray]:
    return [cv2.imdecode(np.frombuffer(v.jpg, np.uint8), cv2.IMREAD_COLOR) for v in home.views]


def _finite(x: float) -> float | None:
    return round(x, 2) if math.isfinite(x) else None


def beside_in_frame(
    label: str, bearing: float, frame: list[tuple[str, float, float]]
) -> str | None:
    """The fixed thing seen nearest in bearing to `label` in the same frame (furniture before
    structure, never a carried item or a person): a place without a pose, "by the desk"."""
    fixed = [
        (layer_of(other, "object") != "furniture", abs(b - bearing), other)
        for other, b, _ in frame
        if other not in (label, "person") and layer_of(other, "object") != "carried"
    ]
    return min(fixed)[2] if fixed else None


def found_message(label: str) -> dict:
    """What the phone says when the guide arrives: for a seat, how to sit without looking."""
    if label in ("chair", "armchair", "sofa", "bench", "stool"):
        text = f"{label.capitalize()} right in front of you. Turn around, reach back for the seat."
    else:
        text = f"Your {label} is about one step ahead."
    return {"type": "answer", "label": label, "text": text, "pin": None, "language": "template"}


class Server:
    def __init__(self, settings: Settings, session: str):
        self.s = settings
        self.session = session
        self.hub = Hub()
        self.pins = PinStore(items=self.s.meta_items)
        self.language = make_language(settings)
        reviewer = make_reviewer(settings)
        self.review = (
            MapReview(reviewer, self.pins, 60 * settings.review_every_min) if reviewer else None
        )
        self.safety: SafetyLoop | None = None
        self.pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="safety", initializer=band.stop_path_thread
        )
        self.llm_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="llm", initializer=band.low_thread
        )
        self.fall, self.shake = FallDetector(), ShakeDetector()
        self.notes = Notes(SESSIONS / "notes")  # conversation notes (notes.py)
        self._notes_task: asyncio.Task | None = None
        self.onboarding = Onboarding()  # first run: walk, name things, feel the modes
        self._load_watch(settings)
        self.talk = Talk(settings.hfov_deg)
        self.rollcall_taps = roll_call.BeltTaps()
        self.persist = Persist(2)  # two frames near before a motor fires
        self.persist_strips = Median(3) if self.s.cue_median else Persist(2)  # panned cues' strips
        self.path_filter = Median(3) if self.s.cue_median else None
        self.cue_range = Params(
            off_m=self.s.cue_off_m, fast_m=self.s.cue_fast_m, solid_m=self.s.cue_solid_m
        )
        self.path_m = math.inf  # this frame's clear distance straight ahead
        self.rhythm = RhythmHold(self.s.cue_hold_s)
        self.activity = Activity()  # guide, find, talk or explore (modes.py)
        self.talk.cues_enabled = self.activity.profile.speaker_cues
        self.voice: VoiceAsk | None = None  # spoken questions through Muse Voice Transcribe
        if settings.asr == "meta" and (key := secret("META_API_KEY", "MODEL_API_KEY")):
            words = list(dict.fromkeys([*settings.watchlist, *settings.meta_items]))
            self.voice = VoiceAsk(self, key, words)
            self.voice.asr.on_heard = self.heard
        self.motion_t = 0.0
        self.phone_clock = math.inf  # server time minus phone time, least delayed sample
        self.accel: deque[np.ndarray] = deque(maxlen=40)
        self.pose = None
        self.source_ws = None  # the client sending camera frames (phone, glasses app, browser)
        self.profile_hfov = settings.hfov_deg
        self.lenses: dict[WebSocket, float] = {}  # field of view a client measured on its camera
        self.cameras: dict[WebSocket, str] = {}  # what a client streams from: glasses, iphone
        self.walks: dict[WebSocket, WalkRecorder] = {}  # clients saving their walk on this Mac
        self.guide: dict | None = None  # the item the belt is pointing to, after a question
        self.found_until = 0.0
        # (time, [(label, bearing, dist)]) of the last processed frame's detections.
        self.live_seen: tuple[float, list[tuple[str, float, float]]] = (0.0, [])
        # label -> when the camera last saw it this run, for answers memory has not caught up on.
        self.live_last: dict[str, float] = {}
        # label -> (time, bearing, metres, map point or None) of its nearest last sighting
        # label -> (ts, bearing, dist, map point or None, the fixed thing beside it or None)
        self.live_where: dict[str, tuple] = {}
        self.last_obstacles: list = []  # the label layer's last result, for the item guide
        self.heading = Heading(settings.hfov_deg)  # carries the head's turn between map poses
        self.fov = FovLoop(settings.hfov_deg, settings.eye_height_m, low_slope=0.02)
        self.map_sectors: list[float] = [math.inf] * 8
        self.map_pose_ts = 0.0
        # The map flattened to a floor plan (nav.py): routes for the item guide, priors for the
        # reflex. Rebuilt in a thread from the map points as they arrive.
        self.map_cells: list[np.ndarray] = []
        self.floor: nav.FloorPlan | None = None
        self.floor_at, self.floor_dirty, self.floor_job = 0.0, False, False
        self.route: nav.Navigator | None = None
        self.route_key: tuple | None = None
        self.motion: deque[tuple[float, np.ndarray, float]] = deque(maxlen=30)  # (t, pos, yaw)
        self.priors_off_until = 0.0
        self.ghosts = nav.Ghosts(need_seen=True)  # map cells live depth confirmed or saw through
        self.cam_aspect = 9 / 16  # width over height of the frames memory solves
        # The fixed room's matcher (built on first use, False without a fixed home and mesh).
        self.room_matcher: RoomMatcher | bool | None = None
        self.room_job, self.room_at = False, 0.0
        # The saved objects that vouch for weak boxes, and the last placed frame's pose.
        self.room_prior: RoomPrior | None = None
        self.room_seen: tuple[float, np.ndarray, np.ndarray] | None = None  # time, pose, K
        # Phone mode: ARKit tracks every frame, and each memory pose for a tracked frame ties
        # ARKit's world to the map's, so the map pose is current between windows.
        self.arposes: deque[tuple[int, np.ndarray]] = deque(maxlen=900)  # (frame seq, pose)
        self.ar_align: np.ndarray | None = None  # ARKit world -> map
        self.ar_pose_ts = 0.0
        self.ar_sent = 0.0
        self.map_yaw = 0.0  # heading the map sectors were measured along
        self.trail: list[list[float]] = []
        self.latest: tuple[float, int, bytes] | None = None
        self.latest_arrived = 0.0  # monotonic time the latest camera frame reached the server
        self.in_fps = 0.0  # frames a second arriving from the camera (DAT on the glasses)
        self.belt = BeltLink(
            settings.belt, settings.belt_addr, self._belt_state, relay=self._belt_relay
        )
        self.braille = make_output(settings, self.hub.json, self.belt)
        self.frame_ready = asyncio.Event()
        self.pending_taps: list[tuple[str, float, float]] = []
        self.pending_kf = 0
        self.current_walk: str | None = None
        self.last_kf = 0.0
        self.kf_best: dict | None = None  # sharpest frame so far in this keyframe interval
        self.fps = 0.0
        self.mode = settings.mode
        self.last_done = 0.0
        self.memory_ready = False
        self.map_loaded = asyncio.Event()
        self.memory_stats: dict = {}
        self.ctx = mp.get_context("spawn")
        self.in_q = self.ctx.Queue()
        self.out_q = self.ctx.Queue()
        self.reflex_busy = self.ctx.Event()  # set while the depth pass runs; the map yields
        self.proc: mp.Process | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        if band.split():  # started in the background band: only the stop path leaves it
            self.loop.set_default_executor(ThreadPoolExecutor(initializer=band.low_thread))
        self.safety = await self.loop.run_in_executor(self.pool, SafetyLoop, self.s)
        self.proc = self.ctx.Process(
            target=memory_run,
            args=(self.in_q, self.out_q, self.s, self.session),
            kwargs={"reflex_busy": self.reflex_busy},
            daemon=True,
        )
        self.proc.start()
        band.low_process(self.proc.pid)
        await self.belt.start()
        threading.Thread(target=self._read_memory, daemon=True).start()
        self._frame_task = asyncio.create_task(self.frame_guard())
        if self.review:
            self._review_task = asyncio.create_task(self.review_loop())

    def use_source(self, sock: WebSocket) -> None:
        """Frames now come from `sock`: use the lens it reported (an iPhone measures its own),
        else the profile's (the glasses report none)."""
        self.source_ws = sock
        self.talk.reset()
        hfov = self.lenses.get(sock, self.profile_hfov)
        if hfov == self.s.hfov_deg:
            return
        self.s.hfov_deg = hfov  # the safety loop reads it from these settings every frame
        self.heading.hfov_deg = hfov
        self.in_q.put({"cmd": "hfov", "hfov": hfov})
        log.info("camera lens now %.1f deg horizontal", hfov)

    def on_camera(self, sock: WebSocket, m: dict) -> None:
        """A client describes its camera, and whether to save its walk here."""
        self.cameras[sock] = str(m.get("source", ""))
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
        self._walk_started(str(m.get("walk") or f"conn-{id(sock)}"))
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

    def _walk_started(self, key: str) -> None:
        """A new walk starts with an empty pin set; the old walk's pins stay in its folder.
        The same walk id reconnecting (a phone that dropped) keeps its pins."""
        prev, self.current_walk = self.current_walk, key
        if prev is None or prev == key:
            return
        folder = WALKS / prev
        if self.pins.pins and folder.is_dir():
            (folder / "pins.json").write_text(
                json.dumps([p.to_json(False) for p in self.pins.pins.values()], indent=1)
            )
        self.pins = PinStore(items=self.s.meta_items)
        # The memory worker keeps the house across walks: it files the old walk and starts the
        # new one from the saved home map.
        self.in_q.put({"cmd": "walk", "prev": prev})

    def drop_client(self, sock: WebSocket) -> None:
        self.hub.clients.discard(sock)
        self.lenses.pop(sock, None)
        self.cameras.pop(sock, None)
        if walk := self.walks.pop(sock, None):
            walk.close()
        if not self.hub.clients:
            self.stop_notes_soon()
        if self.voice and not self.hub.clients:
            self.voice.asr.want(False)  # nobody left to speak; the session closes itself

    def _belt_state(self, msg: dict) -> None:
        asyncio.ensure_future(self.hub.json(msg))

    def _belt_relay(self, msg: dict) -> None:
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
        if t == "ready":  # the worker sends the home map's pins first
            self.memory_ready = True
            self.map_loaded.set()
            await self.hub.json({"type": "status", "memory": "ready", "finder": ev["finder"]}, True)
        elif t == "map_replace":
            await self.hub.map_replace(ev["n"], ev["points"], ev["colors"], ev.get("sizes", b""))
            self.map_cells = [self._cells(ev["points"])]
            self.floor_dirty = True
            if ev["n"] == 0:
                self.trail = []
                await self.hub.json({"type": "trail", "points": []}, True)
        elif t == "mode":
            self.mode = ev["mode"]
            await self.hub.json({"type": "mode", "mode": self.mode}, True)
        elif t == "map_chunk":
            await self.hub.map_chunk(ev["n"], ev["points"], ev["colors"])
            self.map_cells.append(self._cells(ev["points"]))
            self.floor_dirty = True
        elif t == "change":
            said = {
                "moved": f"The {ev.get('label')} has moved since you were last here.",
                "maybe_moved": f"The {ev.get('label')} may have moved. Have a look around here.",
            }.get(ev["change"])
            message = {"message": said} if said else {}
            await self.hub.json({"type": "alert", "kind": ev["change"], **ev, **message}, True)
        elif t == "trail_replace":
            self.trail = ev["points"]
            await self.onboard_trail(ev["points"])
            await self.hub.json({"type": "trail", "points": self.trail}, True)
        elif t == "trail":
            self.trail += ev["points"]
            await self.onboard_trail(ev["points"])
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
            self._align_arkit(ev.get("seq"), pose)
            for walk in self.walks.values():  # so a replay can join ARKit to the map exactly
                walk.map_pose(ev.get("seq"), ev.get("ts"), pose, self.ar_align)
            self.heading.anchor(pose[:3, 3], pose[:3, 2], ev.get("ts"))
            self.map_sectors = [math.inf if d is None else d for d in ev["map_sectors"]]
            self.map_yaw = math.atan2(ev["fwd"][0], ev["fwd"][2])
            self.map_pose_ts = time.time()
            self.cam_aspect = ev.get("aspect", self.cam_aspect)
            out = {"type": "pose", "pos": ev["pos"], "fwd": ev["fwd"], "up": ev["up"]}
            out.update({k: ev[k] for k in ("hfov", "aspect") if k in ev})
            await self.hub.json(out, True)
        elif t == "scale_hint" and self.safety:
            self.safety.apply_scale_hint(ev["ratio"])
        elif t == "memory_stats":
            self.pending_kf = max(0, self.pending_kf - ev.get("consumed", 0))
            self.memory_stats = ev
            await self.hub.json({**ev, "type": "memory"}, True)
        elif t == "global_stats":
            await self.hub.json(ev, True)
        elif t == "dropped":  # keyframes the worker threw away with the old walk's buffer
            self.pending_kf = max(0, self.pending_kf - ev["consumed"])
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
        return small, self.safety.reflex(small), gray, sharp, bgr

    def _label(self, small, res, bgr):
        """The label layer, after the buzz went out: detections, and the finder's larger frame."""
        k = FINDER_LONG_SIDE / max(bgr.shape[:2])
        hi = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else bgr
        res = self.safety.label(small, res)
        seen = self.room_seen
        if self.room_prior and seen and time.time() - seen[0] < ROOM_PRIOR_AGE_S:
            res.obstacles = self.room_prior.apply([*res.obstacles, *res.weak], seen[1], seen[2])
        return res, hi

    async def review_loop(self) -> None:
        """The deep speed: every few minutes the reviewer reads the map's change log. Off the
        event loop and on the llm pool, so it never holds up a frame or a spoken answer."""
        while True:
            await asyncio.sleep(self.review.every_s)
            try:
                res = await self.loop.run_in_executor(self.llm_pool, self.review.run)
            except Exception:
                log.exception("map review failed")
                continue
            if res["changes"]:
                log.info("map review: %d changes, %s", res["changes"], res.get("usd"))
                await self.hub.json(res, replay=True)

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
            self.reflex_busy.set()
            try:
                done = await self.loop.run_in_executor(self.pool, self._process, ts, seq, jpeg)
            finally:
                self.reflex_busy.clear()
            if done is None:
                continue
            small, res, gray, sharp, bgr = done
            now = time.time()
            self._room_view(now, seq, small)
            self.heading.image_turn(gray, now)
            if self.last_done:
                inst = 1.0 / max(1e-3, now - self.last_done)
                self.fps = inst if self.fps == 0 else 0.85 * self.fps + 0.15 * inst
            self.last_done = now
            live = self.persist(res.sectors)
            merged = merge_sectors(live, self._map_sectors_now())
            sc = res.scene
            # Side and back sectors come from the map; trust them only while the pose is fresh.
            map_fresh = self.memory_ready and self.map_pose_ts and now - self.map_pose_ts < 3.0
            sides = bool(map_fresh) and self.s.map_sides
            known = [n in CAMERA_SECTORS or sides for n in SECTORS]
            if self.s.belt_cues == "panned":
                strips = self.persist_strips(grid_strips(sc.grid))
                if self.path_filter is not None:
                    sc.path_m = self.path_filter([sc.path_m])[0]
                p = self.cue_range
                targets = scene_targets(strips, self.s.hfov_deg, merged, known, p)
                cmd, flags = plan_pan(targets, known, sc.path_m, res.ttc_s, sc.drop_m, p)
            else:
                cmd, flags = plan(merged, known, sc.path_m, res.ttc_s, sc.drop_m)
            # The map's priors may add or raise cues; a cue live depth made is never weakened.
            self._maybe_floor(now)
            pri = self._priors(now, live)
            priors_did = None
            if pri is not None and pri.active:
                # Shadow mode (lead, 02:05): priors are logged, not applied, until path-gating
                # and ghost clearing are scored against the ground-truth labels.
                shadow = [list(c) for c in cmd]
                did = nav.apply(shadow, pri, merged, off_path="none", hidden=False)
                changed = any(did.values())
                priors_did = {**did, "ms": round(pri.ms, 2), "shadow": True}
                if changed:  # the command the priors would have sent, for replays
                    priors_did.update(raw=[list(c) for c in cmd], after=shadow)
            # Shadow (fov-lab): what the out-of-view hazard memory would cue; logged, not applied.
            fov = self.fov.step(
                now, res.depth_m, sc.grid, sc.floor, res.obstacles, list(self.accel), self.heading
            )
            cmd = self.activity.calm(cmd, flags)
            if (demo := self.onboarding.demo_cmd(now)) is not None:
                cmd = demo  # the belt intro: a mode's pattern, nothing else
            # Two layers: the stop cue above comes from depth and the floor alone and goes out
            # now; the speaker turn and the item guide ride on the last frame's labels.
            cmd = self.talk.overlay(cmd, now)
            cmd = self.rollcall_taps.overlay(cmd, now)
            # The heading is carried between poses, so guidance tolerates an older position.
            guide_fresh = self.memory_ready and self.guide_pose_fresh(now)
            self.path_m = sc.path_m
            guiding = self._guide(cmd, now, bool(guide_fresh), self.last_obstacles)
            if now < self.found_until:
                flags |= FLAG_FOUND
            if self.s.belt_strength == "strong":
                cmd = strong(cmd)
            if self.s.cue_hold_s > 0:
                cmd = self.rhythm(cmd, now)
            if not (self.braille and self.braille.owns_belt):  # a Braille cue on the belt wins
                self.belt.send(cmd, flags | self.activity.flag_bits(), arrived)
            buzzed = time.perf_counter()
            label = band.low(self._label)  # after the buzz: no longer the stop path
            res, hi = await self.loop.run_in_executor(self.pool, label, small, res, bgr)
            self.last_obstacles = res.obstacles
            people = self.talk.frame(now, res.obstacles, small, self.heading)
            self.notes.vote(people)
            self.live_seen = (
                now,
                [
                    (o.label, self.heading.box_bearing(o.box), o.dist_m)
                    for o in [*res.obstacles, *res.objects]
                    if math.isfinite(o.dist_m)
                ],
            )
            placed = bool(self.memory_ready and self.guide_pose_fresh(now))
            for label, bearing, dist in sorted(self.live_seen[1], key=lambda x: -x[2]):
                self.live_last[label] = now
                point = self.heading.ahead(dist, bearing) if placed else None
                beside = beside_in_frame(label, bearing, self.live_seen[1])
                self.live_where[label] = (now, bearing, dist, point, beside)
            await self.hub.frame(seq, jpeg, self.source_ws)
            await self.hub.json(people, True)
            if people["cue"]:
                await self.hub.json({"type": "speaker_cue", **people["cue"]})
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
                    "priors": priors_did,
                    "fov": fov,
                    "pos": None if self.heading.pos is None else self.heading.pos.round(3).tolist(),
                    "found": now < self.found_until,
                    "sound": self.s.sound,  # the phone plays the find beacon and hazard beeps
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
                    "in_fps": round(self.in_fps, 1),
                    "camera": self.cameras.get(self.source_ws, ""),
                    "ms": {k: round(v, 1) for k, v in res.timings_ms.items()},
                    # how long after the belt command this message left
                    "after_buzz_ms": round(1000 * (time.perf_counter() - buzzed), 1),
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
        pin = self._guide_pin(g)
        rel = (
            self.heading.to(np.array(pin.pos))
            if pin is not None and pin.placed and map_fresh
            else None
        )
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
            return {
                "label": g["label"],
                "pin_id": None if pin is None else pin.id,
                "bearing_deg": None,
                "dist_m": None,
                "turn": None,
                "dist": None,
                "bearing": None,
                "in_view": False,
            }
        dist, bearing = rel
        # Out of view, the route around walls and furniture gives the direction, not the
        # straight line to the item.
        step = None
        if not live and pin is not None and (nv := self._navigator(pin.id, pin)) is not None:
            step = nv.step(self.heading.pos, self.heading.yaw)
        # The detector's distance for a chair or table takes in the floor seen between its legs
        # and reads long; in view and dead ahead, the clear path straight ahead ends at it.
        reach = min(dist, self.path_m) if live and abs(bearing) < GUIDE_ALIGNED_DEG else dist
        if reach < self.s.found_m or (step is not None and step.arrived):
            self.guide, self.found_until = None, now + 2.0
            self.loop.create_task(self.hub.json(found_message(g["label"]), False))
            return None
        if step is not None:
            bearing = step.bearing_deg
        # In view and nothing nearer ahead, the fast cue in front is the item itself: let the
        # guide taps through it. Solid still silences the guide.
        target_ahead = bool(live) and self.path_m > dist - 0.4
        cmd[:] = with_guide(cmd, bearing, take_fast=target_ahead)[0]
        aligned = abs(wrap(math.radians(bearing))) <= math.radians(GUIDE_ALIGNED_DEG)
        return {
            "label": g["label"],
            "pin_id": None if pin is None else pin.id,
            "bearing_deg": round(bearing, 1),  # the way to turn now: the route's, or straight
            "dist_m": round(dist, 2),
            "turn": "ahead" if aligned else ("right" if bearing > 0 else "left"),
            "dist": round(dist, 2),
            "bearing": round(bearing, 1),
            "in_view": bool(live),  # the camera sees it now, not just the map
            "route": None if step is None else _finite(step.route_m),
            "narrow": None if step is None else _finite(step.narrow_m),
        }

    def guide_to(self, pin_id) -> bool:
        """Point the belt at a pin for 90 s, as after "where is my X"; None stops the guide.
        Only a pin the store has and would answer with (a weak personal-item hit is not one)."""
        if pin_id is None:
            self.guide = None
            return True
        pin = self.pins.pins.get(pin_id) if isinstance(pin_id, int) else None
        if pin is None or not self.pins.stands(pin):
            return False
        self.guide = {"pin": pin.id, "label": pin.label, "until": time.time() + 90.0}
        return True

    def _guide_pin(self, g: dict):
        """The pin to point at: the one answered, until memory places a newer pin of the same
        thing (this walk's sighting of a saved home pin not placed yet, or the item seen again
        elsewhere). Only a strictly newer placed pin takes over, so the target never flips back
        and forth; the switch sticks, and the route and camera lock start again from it."""
        pin = self.pins.pins.get(g["pin"]) if g["pin"] is not None else None
        here = [  # never onto a personal item's pin only a weak local hit placed
            p
            for p in self.pins.pins.values()
            if p.label == g["label"] and p.placed and self.pins.stands(p)
        ]
        newest = max(here, key=lambda p: p.ts, default=None)
        if newest is None or (pin is not None and pin.placed and newest.ts <= pin.ts):
            return pin
        if newest.id != g["pin"]:
            g["pin"] = newest.id
            g.pop("seen", None)  # the lock vetted the camera against the old pin's bearing
        return newest

    def can_point(self, g: dict, now: float) -> bool:
        """Whether the belt has a direction for the guide: in view now, or a placed pin and a
        pose recent enough to point from."""
        t, seen = getattr(self, "live_seen", (0.0, []))
        if now - t < 1.0 and any(label == g["label"] for label, _b, _d in seen):
            return True
        pin = self._guide_pin(g)
        return (
            pin is not None
            and pin.placed
            and self.heading.to(np.array(pin.pos)) is not None
            and self.guide_pose_fresh(now)
        )

    def guide_pose_fresh(self, now: float) -> bool:
        """An ARKit-carried pose points for a minute; a memory pose dead-reckoned on heading
        alone for 15 s."""
        ar = getattr(self, "ar_pose_ts", 0.0)
        return now - ar < GUIDE_POSE_S or now - self.map_pose_ts < GUIDE_DR_S

    @staticmethod
    def _cells(points: bytes) -> np.ndarray:
        return occupied_cells(np.frombuffer(points, np.float32).reshape(-1, 3))

    def _maybe_floor(self, now: float) -> None:
        """Rebuild the floor plan from the map points in a thread, at most every 2 s."""
        if not self.floor_dirty or self.floor_job or now - self.floor_at < FLOOR_EVERY_S:
            return
        keys = [c for c in self.map_cells if len(c)]
        self.floor_dirty = False
        if not keys:
            return
        self.map_cells = [np.unique(np.concatenate(keys), axis=0)]
        self.floor_at, self.floor_job = now, True

        def done(f) -> None:
            self.floor_job = False
            self.floor = f.result()

        self.loop.run_in_executor(None, nav.floor_plan, self.map_cells[0]).add_done_callback(done)

    def _navigator(self, key: str, pin) -> nav.Navigator | None:
        """The route to a pin on the current floor plan. A new plan or pin routes again in a
        thread (one Dijkstra, tens of ms); the old route serves until it lands."""
        want = (key, tuple(np.round(pin.pos, 2)), id(self.floor))
        route = self.route
        if route is not None and not np.allclose(route.goal, pin.pos, atol=0.05):
            route = None  # a route to somewhere else
        if self.floor is None or want == self.route_key:
            return route
        floor, self.route_key = self.floor, want

        def solve() -> nav.Navigator | None:
            nv = nav.Navigator(floor)
            return nv if nv.set_goal(pin.pos, pin.label) else None

        def done(f) -> None:
            if self.route_key == want:
                self.route = f.result()

        self.loop.run_in_executor(None, solve).add_done_callback(done)
        return route

    def _priors(self, now: float, live: list[float]) -> nav.Priors | None:
        """What the map adds to this frame's cues (nav.priors), or None without a usable pose.
        The position error grows with the pose's age and the walking speed; a re-tie (the pose
        jumping more than 0.3 m in a frame) turns them off for a second."""
        pos, yaw = self.heading.pos, self.heading.yaw
        if self.floor is None or pos is None or yaw is None or not self.memory_ready:
            self.motion.clear()
            return None
        if self.motion and float(np.linalg.norm(pos - self.motion[-1][1])) > RETIE_M:
            self.motion.clear()
            self.priors_off_until = now + 1.0
            self.ghosts = nav.Ghosts(need_seen=True)  # the map moved under us: start over
        self.motion.append((now, pos.copy(), yaw))
        if now < self.priors_off_until:
            return None
        t0, p0, y0 = self.motion[0]
        dt = now - t0
        speed = math.hypot(pos[0] - p0[0], pos[2] - p0[2]) / dt if dt > 0.2 else 0.0
        turn = wrap(yaw - y0) / dt if dt > 0.2 else 0.0
        hfov = self.s.hfov_deg
        vfov = 2 * math.degrees(math.atan(math.tan(math.radians(hfov / 2)) / self.cam_aspect))
        pitch = 0.0  # degrees the camera looks below level
        if self.pose is not None:
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(self.pose[1, 2])))))
        return nav.priors(
            self.floor,
            pos,
            yaw,
            live,
            hfov,
            speed,
            turn,
            0.05 + speed * (now - self.pose_ts()),
            self.s.eye_height_m,
            min(80.0, max(10.0, vfov / 2 + pitch)),
            self.ghosts,
        )

    def pose_ts(self) -> float:
        """When the map pose was last current: a memory window, or ARKit carrying it since."""
        return max(self.map_pose_ts, getattr(self, "ar_pose_ts", 0.0))

    def _align_arkit(self, seq: int | None, pose: np.ndarray) -> None:
        """A map pose for a frame the phone also tracked ties ARKit's world to the map's."""
        a = next((a for s, a in reversed(self.arposes) if s == seq), None)
        if a is not None:
            self.ar_align = pose @ np.linalg.inv(a)

    async def on_arpose(self, m: dict) -> None:
        """Between memory windows the phone's own tracking carries the map pose, frame by
        frame; MapAnything still corrects it at every window."""
        if m.get("track") != "tracking":
            return
        try:
            a = arkit_pose(m["m"])
        except (KeyError, TypeError, ValueError):
            return
        self.arposes.append((m.get("seq"), a))
        if self.ar_align is None:
            return
        pose = self.ar_align @ a
        self.pose = pose
        self.heading.track(pose[:3, 3], pose[:3, 2])
        now = time.time()
        self.ar_pose_ts = now
        if now - self.ar_sent >= 1.0 / AR_POSE_HZ:
            self.ar_sent = now
            fwd, up = pose[:3, 2].tolist(), (-pose[:3, 1]).tolist()
            await self.hub.json(
                {"type": "pose", "pos": pose[:3, 3].tolist(), "fwd": fwd, "up": up}, True
            )

    def _room_view(self, now: float, seq: int, small: np.ndarray) -> None:
        """The room overlay's pose: this frame placed in the fixed home's room by PnP on the
        saved views, a few times a second. The map pose arrives a window late and only its
        heading is carried in between, which slid the room's outlines off the picture."""
        if self.room_job or self.room_matcher is False or now - self.room_at < 1 / ROOM_VIEW_HZ:
            return
        self.room_job, self.room_at = True, now

        def run():
            if self.room_matcher is None:
                home = homemod.load()
                ok = home and home.fixed and (homemod.home_dir() / "room.ply").is_file()
                self.room_matcher = RoomMatcher(_home_bgr(home), home.views) if ok else False
                objs = homemod.home_dir() / "objects.json"
                if ok and objs.is_file():
                    self.room_prior = RoomPrior(
                        json.loads(objs.read_text()), homemod.home_dir() / "room.ply"
                    )
            m = self.room_matcher
            return m.locate(small) if m and small.shape[:2] == m.shape else None

        def done(f) -> None:
            self.room_job = False
            hit = None if f.exception() else f.result()
            if f.exception():
                log.warning("room view failed: %s", f.exception())
            if hit is None or hit[1] < ROOM_VIEW_MIN_INLIERS:
                return
            m = self.room_matcher
            self.room_seen = (now, hit[0], m.k / np.array([[m.shape[1]], [m.shape[0]], [1]]))
            msg = {"type": "room_pose", "seq": seq, "pose": hit[0].tolist(), "inliers": hit[1]}
            msg.update(hfov=self.room_matcher.hfov, aspect=small.shape[1] / small.shape[0])
            asyncio.ensure_future(self.hub.json(msg))

        self.loop.run_in_executor(None, run).add_done_callback(done)

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
        if samples and len(samples[-1]) >= 7:
            # Samples carry the phone's clock and keyframes the server's. The smallest gap seen
            # is the one with the least network delay in it, so it maps one clock onto the other.
            self.phone_clock = min(self.phone_clock, now - samples[-1][0])
            if self.memory_ready:
                rows = [r for r in samples if len(r) >= 7]
                self.in_q.put(
                    {
                        "cmd": "gyro",
                        "t": [r[0] + self.phone_clock for r in rows],
                        "w": [r[4:7] for r in rows],
                    }
                )
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
    def _planner(self):
        """Errand plans, grounded in the bank. Someone is waiting on these, so Spark makes them:
        on the live bank it took 1.5 to 1.7 s where Wonder took 2.9 to 13.9 s for the same
        grounded steps. Wonder keeps the background review, where nobody waits."""
        return lambda goal: errand_plan(self.language, self.pins, goal, time.time())

    def _answer_sync(self, text: str) -> dict:
        if (needs_tools(text) or asks_notes(text)) and hasattr(self.language, "_client"):
            # People, voices, changes: Spark asks the pins, the map and the people tracks itself.
            # One to three calls, so only these questions take this path.
            try:
                tools = Tools(
                    self.pins,
                    self.heading,
                    lambda: self.hub.last.get("people"),
                    notes=self.notes,
                    planner=self._planner(),
                )
                said = agent_answer(self.language, text, tools)
            except Exception:
                said = ""
            if valid_sentence(said):
                if asks_notes(text) and self.notes.check(said):
                    said = self.notes.fallback(text)  # a quote or number the notes never held
                return {
                    "type": "answer",
                    "label": None,
                    "text": said.strip(),
                    "pin": None,
                    "language": self.language.name,
                    "tools": True,
                    **({"steps": tools.steps} if tools.steps else {}),
                }
        seen_at, seen = self.live_seen
        if time.time() - seen_at > LIVE_SEEN_S:
            seen = []
        labels = sorted({*self.pins.labels(), *self.live_last})
        label = parse_query(text, labels)
        guess = None
        if label is None and labels and self.language.name != "template":
            try:
                pick = (
                    self.language.complete(
                        "Reply with the one item from the list that is the very object asked "
                        "about under another name (sneakers: shoe). If none is, reply none.",
                        f"Items: {', '.join(labels)}\nQuestion: {text}",
                        max_tokens=8,
                    )
                    .strip()
                    .lower()
                )
                guess = pick if pick in labels else None
            except Exception:
                guess = None
        if label is None:
            noun = asked_for(text)  # "where is my umbrella": named, but never seen
            pin = self.pins.latest(guess) if guess and noun else None
            if pin is not None:  # the model's other name for it is only ever offered as a maybe
                now = time.time()
                pose = self.pose if now - Server.pose_ts(self) < ANSWER_POSE_S else None
                pose = pose if pin.placed else None
                self.guide = {"pin": pin.id, "label": pin.label, "want": noun, "until": now + 90}
                return {
                    "type": "answer",
                    "label": None,
                    "text": template_answer(noun, pin, pose, now, stand_in=True),
                    "pin": pin.to_json(),
                    "language": "template",
                    "want": noun,
                }
            return {
                "type": "answer",
                "label": None,
                "text": unseen_answer(noun) if noun else "Which object do you mean?",
                "pin": None,
                **({"want": noun} if noun else {}),  # never seen: the camera looks for it now
            }
        pin = self.pins.latest(label)
        live = [s for s in seen if s[0] == label]
        where = getattr(self, "live_where", {}).get(label)
        if not live and where is not None and time.time() - where[0] < LIVE_HOLD_S:
            live = [(label, where[1], where[2])]  # out of the last frame, not out of view
        if live:
            # In frame right now: the live sighting wins over memory, however old the pin.
            _, bearing, dist = min(live, key=lambda s: s[2])
            pin_id = pin.id if pin else None
            self.guide = {"pin": pin_id, "label": label, "until": time.time() + 90.0}
            return {
                "type": "answer",
                "label": label,
                "text": live_answer(label, dist, bearing),
                "pin": pin.to_json() if pin else None,
                "language": "template",
                "live": True,
            }
        last = self.live_last.get(label)
        if last is not None and (pin is None or pin.ts < last - LIVE_OVER_PIN_S):
            # The camera saw it this run, well after memory last placed it: the sighting is the
            # answer, never the older pin. Where the pose placed the sighting, say where from here.
            now = time.time()
            self.guide = {"pin": None, "label": label, "until": now + 90.0}
            rel, beside = None, None
            if where is not None and where[3] is not None and self.guide_pose_fresh(now):
                rel = self.heading.to(where[3])
            elif where is not None and len(where) > 4:
                beside = where[4]  # no pose to place it: what was in view with it says where
            # Seen beside what memory already has it near: the same place, so the pin is good
            # to fly to; anywhere else the old pin is not where it is.
            same = pin is not None and beside is not None and beside in pin.near
            if same:
                self.guide["pin"] = pin.id
            return {
                "type": "answer",
                "label": label,
                "text": camera_answer(label, now - last, rel, beside),
                "pin": pin.to_json() if same else None,
                "language": "template",
            }
        glimpsed = self.pins.glimpse(label, pin.ts) if pin is not None else None
        if glimpsed is not None:
            # Too unsure to move the pin, but far fresher than it: offer it, hedged.
            self.guide = {"pin": glimpsed.id, "label": label, "until": time.time() + 90.0}
            return {
                "type": "answer",
                "label": label,
                "text": glimpse_answer(label, glimpsed, time.time()),
                "pin": glimpsed.to_json(),
                "language": "template",
            }
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
        pose = self.pose if now - Server.pose_ts(self) < ANSWER_POSE_S else None
        if pin is not None and not pin.placed:  # a saved pin not yet placed in this walk's frame
            pose = None
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
            # "Not seen" over a pin the store passed by: say which, so a check can see why.
            **({"skipped_pin_id": w.id} if pin is None and (w := self.pins.skipped(label)) else {}),
        }

    def _load_watch(self, settings: Settings) -> None:
        """Things the wearer named in earlier runs join the watch list before the worker starts."""
        try:
            extra = json.loads((SESSIONS / "watch.json").read_text())
        except (OSError, ValueError):
            return
        settings.watchlist.extend(w for w in extra if w not in settings.watchlist)

    def add_watch(self, label: str) -> None:
        """Add a named thing to the watch list now, and keep it for the next run."""
        if label in self.s.watchlist:
            return
        self.s.watchlist.append(label)
        self.in_q.put({"cmd": "watch", "label": label})
        SESSIONS.mkdir(exist_ok=True)
        (SESSIONS / "watch.json").write_text(json.dumps(self.s.watchlist))

    async def onboard_trail(self, points: list) -> None:
        for ev in self.onboarding.trail(points):
            await self.hub.json(ev, True)

    async def start_onboarding(self) -> None:
        await self.hub.json(self.onboarding.start(), True)

    async def run_belt_intro(self) -> None:
        """Each mode in turn: its line, then its pattern on the belt."""
        for ev, mode in self.onboarding.belt_events():
            await self.hub.json(ev, True)
            await asyncio.sleep(2.5)  # the line is spoken first
            if mode:
                self.onboarding.demo_mode, self.onboarding.demo_t0 = mode, time.time()
                # The camera may not be streaming yet, so the pattern is sent from here too.
                t_end = time.time() + pattern_length(PATTERNS[mode]) + 0.5
                while time.time() < t_end:
                    demo = self.onboarding.demo_cmd(time.time())
                    self.belt.send(demo, self.activity.flag_bits(), time.monotonic())
                    await asyncio.sleep(0.05)
                self.onboarding.demo_mode = None
                self.belt.send([(0, 0)] * 8, self.activity.flag_bits(), time.monotonic())
        await self.hub.json(self.onboarding.finish(), True)

    # ------------------------------------------------------------ conversation notes
    async def heard(self, h) -> None:
        """Every transcript from Muse Voice Transcribe: notes while recording, else questions."""
        if h.final:
            cmd = notes_command(h.text)
            if cmd == "stop" and self.notes.recording:
                return await self.stop_notes()
            if cmd == "start" and not self.notes.recording:
                return await self.start_notes()
        if self.notes.recording:
            if not h.final:  # the words so far, for the large-text captions
                await self.hub.json(caption(h.text, None, False))
            elif ev := self.notes.hear(h.text, h.spoken_end):
                await self.hub.json(
                    {"type": "transcript", "text": h.text, "final": True, "source": "notes"}
                )
                await self.hub.json(caption(h.text, self.notes.cur.lines[-1].get("who"), True))
                if self.braille:
                    self.braille.say(h.text)  # the newest line wins; haptic reading is slow
                await self.hub.json(ev, True)
            return
        await self.voice.on_heard(h)

    async def start_notes(self) -> None:
        if self.voice is None:
            await self.hub.json(
                {
                    "type": "notes",
                    "state": "idle",
                    "say": "Notes need the voice service, which is off.",
                },
                True,
            )
            return
        await self.hub.json(self.notes.start(), True)
        self._notes_task = asyncio.create_task(self._notes_reminders())

    async def _notes_reminders(self) -> None:
        while self.notes.recording:
            await asyncio.sleep(5)
            line = self.notes.reminder()
            if line == "limit":
                await self.stop_notes()
            elif line:
                await self.hub.json(self.notes.event("recording", line), True)

    async def stop_notes(self) -> None:
        if self._notes_task and self._notes_task is not asyncio.current_task():
            self._notes_task.cancel()
        ev, note = self.notes.stop()
        await self.hub.json(ev, True)
        if note is not None:
            done = await self.loop.run_in_executor(
                self.llm_pool, self.notes.summarize, note, self.language
            )
            await self.hub.json(done, True)

    def stop_notes_soon(self) -> None:
        if self.notes.recording:  # nobody is left to hear the recording cue: end it, keep it
            asyncio.ensure_future(self.stop_notes())

    async def set_activity(self, mode: str, cause: str, speak: bool = True) -> None:
        """Switch guide/find/talk/explore: tell the app (which says the line) and the judge view."""
        ev = self.activity.set(mode, cause, speak)
        if ev is None:
            return
        self.talk.cues_enabled = self.activity.profile.speaker_cues
        await self.hub.json(ev, True)
        if mode == "explore":
            await self.roll_call()

    async def roll_call(self) -> None:
        """What is around me, from the map's pins and the room twin, clockwise from ahead: the
        phone says it and plucks each name from where the thing is; the belt taps toward each."""
        home = homemod.home_dir()
        room = roll_call.room_name(home)
        twin = roll_call.twin_objects(home)
        items = roll_call.pick(list(self.pins.pins.values()), self.heading.to, twin=twin, room=room)
        if not items:
            return
        msg = roll_call.message(items, room)
        await self.hub.json(msg, False)
        self.rollcall_taps.start(msg, time.time())

    def known_labels(self) -> list[str]:
        return sorted({*self.pins.labels(), *self.live_last})

    async def answer(
        self,
        text: str,
        pending: asyncio.Future | None = None,
        quiet_unknown: bool = False,
        spoken_end: float | None = None,
    ) -> dict | None:
        """`pending`: the same answer already started from a partial transcript.
        `quiet_unknown`: overheard speech that names no known thing gets no reply.
        `spoken_end`: wall time the wearer's last word arrived, for the voice timing."""
        t0 = time.perf_counter()
        if self.onboarding.active:
            handled, events, label = self.onboarding.hear(text)
            if label:
                self.add_watch(label)
            for ev in events:
                await self.hub.json(ev, True)
            if handled:
                if self.onboarding.step == "belt":
                    asyncio.create_task(self.run_belt_intro())
                return {"type": "onboarding", "text": text}
        if (cmd := notes_command(text)) is not None:
            if cmd == "start" and not self.notes.recording:
                await self.start_notes()
            elif cmd == "stop" and self.notes.recording:
                await self.stop_notes()
            return {"type": "notes", "text": text}
        if not self.onboarding.active and wants_setup(text):
            await self.start_onboarding()
            return {"type": "onboarding", "text": text}
        if (named := activity_command(text)) is not None:
            await self.set_activity(named, "voice")
            return {"type": "activity", "mode": named, "text": text}
        if not quiet_unknown:
            implied = implied_activity(text, parse_query(text, self.known_labels()) is not None)
            if implied is not None:
                await self.set_activity(implied, "question", speak=False)  # the answer speaks
        if is_errand(text) and (pending is None or not self.map_loaded.is_set()):
            # waiting for the plan (or for the map) would only leave the wearer in silence
            hold = {"type": "answer", "label": None, "text": HOLDING, "pin": None}
            await self.hub.json({**hold, "holding": True})
            if self.braille:
                self.braille.say(HOLDING)
        if not self.map_loaded.is_set():
            # Right after a restart the bank is empty until the worker loads the home map, and
            # "I haven't seen your keys" would be a wrong answer. Wait a moment, then say so; a
            # spoken question is told at once, since seconds of silence after it sound broken.
            try:
                await asyncio.wait_for(self.map_loaded.wait(), 0 if quiet_unknown else MAP_WAIT_S)
                pending = None  # it was answered from the empty bank
            except TimeoutError:
                pending = asyncio.get_running_loop().create_future()
                loading = {"type": "answer", "label": None, "text": MAP_LOADING, "pin": None}
                pending.set_result({**loading, "loading": True})
        if pending is not None:
            res = await pending
        else:
            res = await self.loop.run_in_executor(self.llm_pool, self._answer_sync, text)
        unknown = res.get("label") is None and not res.get("tools")
        if quiet_unknown and unknown and not res.get("loading"):  # "still loading" is still said
            return None
        res["ms"] = round(1000 * (time.perf_counter() - t0))
        if spoken_end is not None:
            res["voice"] = {
                "early": pending is not None,
                "speech_end_to_answer_ms": round(1000 * (time.time() - spoken_end)),
            }
        if (p := res.get("pin")) is not None:  # the pin the answer rests on, and if it stands
            res["pin_id"], res["confirmed"] = p["id"], confirmed(p["score"], p["source"])
        g = getattr(self, "guide", None)
        if g and res.get("label") and res["label"] in (g["label"], g.get("want")):
            if not self.can_point(g, time.time()):  # never a silent belt after a find
                res["text"] = f"{res['text'].rstrip()} {NO_DIRECTION}"
        await self.hub.json(res)
        if res.get("label"):  # SAM 3.1 looks for it only now, while the wearer is after it
            self.in_q.put({"cmd": "want", "label": res["label"]})
        elif res.get("want"):  # a thing the house never saw: the camera learns the word too
            self.in_q.put({"cmd": "want", "label": res["want"], "new": True})
        if self.braille:
            res["cues"] = self.braille.say(res["text"])
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
                "activity": srv.activity.mode,
                "onboarding": srv.onboarding.step,
                "recording": srv.notes.recording,
                "map_points": srv.hub.map_total,
                "memory": srv.memory_stats,
                "pending_kf": srv.pending_kf,
                "asr": (
                    {"model": "muse-voice-transcribe-1.0", **srv.voice.asr.stats, **srv.voice.last}
                    if srv.voice
                    else None
                ),
            }
        )

    @app.get("/api/room.ply")
    async def room_mesh():
        """The room mesh, thinned for the browser (room_home.py writes it beside the home)."""
        f = homemod.home_dir() / "room.ply"
        if not f.is_file():
            return JSONResponse({"error": "no room mesh"}, status_code=404)
        return FileResponse(f, media_type="application/octet-stream")

    @app.get("/api/splat.ply")
    async def room_splat():
        """The room as Gaussian splats (room_splat.py then Brush), in room.ply's frame."""
        f = homemod.home_dir() / "splat.ply"
        if not f.is_file():
            return JSONResponse({"error": "no room splat"}, status_code=404)
        return FileResponse(f, media_type="application/octet-stream")

    @app.post("/api/ask")
    async def ask(body: Ask):
        return await srv.answer(body.text)

    @app.post("/api/braille")
    async def braille_say(body: Ask):
        """Play any text as a haptic Braille cue (demo and wiring check)."""
        if srv.braille is None:
            return JSONResponse({"error": "braille is off; set safety.braille"}, status_code=409)
        return {"cues": srv.braille.say(body.text)}

    @app.post("/api/test_fall")
    async def test_fall():
        for ev in srv.fall.trigger_test(srv.motion_t):
            await srv.hub.json({"type": "alert", "kind": ev.kind, "message": ev.message})
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await srv.hub.join(sock)
        if srv.voice:
            srv.voice.asr.want(True)  # warm before anyone speaks
        try:
            while True:
                msg = await sock.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    data = msg["bytes"]
                    ts, seq = struct.unpack("<dI", data[:12])
                    srv.latest = (ts, seq, data[12:])
                    t = time.monotonic()
                    if srv.latest_arrived:
                        inst = 1.0 / max(1e-3, t - srv.latest_arrived)
                        srv.in_fps = inst if srv.in_fps == 0 else 0.9 * srv.in_fps + 0.1 * inst
                    srv.latest_arrived = t
                    if sock is not srv.source_ws:
                        srv.use_source(sock)
                    if walk := srv.walks.get(sock):
                        walk.frame(ts, seq, data[12:])
                    srv.frame_ready.set()
                elif msg.get("text"):
                    m = json.loads(msg["text"])
                    kind = m.get("type")
                    if kind == "motion":
                        if walk := srv.walks.get(sock):
                            walk.motion(m)
                        await srv.on_motion(m["s"])
                    elif kind == "ask":
                        asyncio.create_task(srv.answer(m["text"]))
                    elif kind == "audio":
                        srv.talk.audio(m, time.time())
                        if srv.voice:
                            srv.voice.audio(m)
                    elif kind == "camera":
                        srv.on_camera(sock, m)
                    elif kind == "arpose":  # the phone's own tracked pose, kept with its walk
                        if walk := srv.walks.get(sock):
                            walk.pose(m)
                        await srv.on_arpose(m)
                    elif kind == "notes":
                        if m.get("action") == "start" and not srv.notes.recording:
                            await srv.start_notes()
                        elif m.get("action") == "stop" and srv.notes.recording:
                            await srv.stop_notes()
                        elif m.get("action") == "delete":
                            srv.notes.delete(str(m.get("id", "")))
                    elif kind == "onboarding":
                        if m.get("action") == "start":
                            await srv.start_onboarding()
                        elif m.get("action") == "skip":
                            await srv.hub.json(srv.onboarding.skip(), True)
                        elif m.get("action") == "belt" and srv.onboarding.step in ("name", "walk"):
                            srv.onboarding.step = "belt"
                            asyncio.create_task(srv.run_belt_intro())
                    elif kind == "activity" and m.get("mode") in ACTIVITIES:
                        await srv.set_activity(m["mode"], "app")
                    elif kind == "mode" and m.get("mode") in ("home", "outside"):
                        srv.in_q.put({"cmd": "mode", "mode": m["mode"]})
                    elif kind == "guide":  # the wearer picked a pin on the screen
                        srv.guide_to(m.get("pin_id"))
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
    config = uvicorn.Config(
        create_app(session=a.session),
        host=a.host,
        port=a.port,
        ssl_certfile=str(cert) if cert else None,
        ssl_keyfile=a.key,
        log_level="info",
    )
    server = uvicorn.Server(config)
    if a.host != "::":
        server.run()
        return
    # iPhones resolve the Mac's .local name to IPv6 first, so listen on both stacks
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.bind(("::", a.port))
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()
