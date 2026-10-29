"""The memory loop in its own process: its own MPS context, so a 3 s window never stalls the
10-20 Hz safety loop in the server process. Talks over two queues of plain dicts."""

from __future__ import annotations

import os
import queue
import time
from pathlib import Path

import numpy as np

from .config import SESSIONS, Settings
from .loops import Describer
from .memory import Keyframe, MapAnythingRunner, MemoryEngine, SolverProcess
from .providers import make_finder
from .walk import WALKS


def yield_to(busy, fn, wait_s: float = 0.06):
    """`fn`, started only once the safety loop's depth pass is done (or after `wait_s`): the
    reflex loop gets the GPU first, the map waits a few ms."""

    def call(*a, **k):
        t = time.monotonic() + wait_s
        while busy.is_set() and time.monotonic() < t:
            time.sleep(0.002)
        return fn(*a, **k)

    return call


def run(
    in_q, out_q, settings: Settings, session: str, new_needed: int = 4, reflex_busy=None
) -> None:
    from ultralytics import YOLO

    default_every = 2 if settings.finder == "sam31" else 1
    sam_every = int(os.environ.get("METASIGHT_SAM_EVERY", default_every))
    engine = MemoryEngine(
        settings,
        MapAnythingRunner(hfov_deg=settings.hfov_deg if settings.map_intrinsics else None),
        make_finder(settings),
        YOLO(str(settings.weights_dir / "yolo26n.pt")),
        sam_every=sam_every,
        describer=Describer(),
        solver=SolverProcess(settings.hfov_deg if settings.map_intrinsics else None),
    )
    if reflex_busy is not None:
        engine.runner.infer = yield_to(reflex_busy, engine.runner.infer)
        engine.landmarks.predict = yield_to(reflex_busy, engine.landmarks.predict)
        engine.finder.find = yield_to(reflex_busy, engine.finder.find)
    engine.emit = out_q.put
    if settings.mode != "home":
        engine.set_mode(settings.mode)
    else:
        engine.load_home()
    SESSIONS.mkdir(exist_ok=True)
    save_path = Path(SESSIONS) / f"{session}.npz"
    if engine._home_prior:  # before "ready": the server answers questions from then on
        out_q.put({"type": "pins", "pins": engine._pins_out()})
    out_q.put({"type": "ready", "finder": engine.finder.name})
    out_q.put({"type": "mode", "mode": engine.mode})
    buf: list[Keyframe] = []
    arrived = 0.0
    while True:
        per = engine.s.window - engine.s.overlap
        need = max(2, per // 2 + 1) if engine.mode == "outside" else new_needed
        try:
            for e in engine.global_poll() or []:
                out_q.put(e)
        except Exception as exc:
            out_q.put({"type": "memory_error", "error": repr(exc)})
        try:
            msg = in_q.get(timeout=0.25)
        except queue.Empty:
            msg = None
        if msg is not None:
            if msg["cmd"] == "stop":
                engine.solver.stop()
                if engine.mode == "home":
                    engine.save(save_path)
                    engine.persist_pins()
                return
            if msg["cmd"] == "walk":  # a new walk: the saved house stays, the walk starts clean
                prev = WALKS / msg["prev"] if msg.get("prev") else None
                engine.new_walk(prev)
                out_q.put({"type": "dropped", "consumed": len(buf)})
                buf = []
                out_q.put({"type": "map_replace", "n": 0, "points": b"", "colors": b""})
                out_q.put({"type": "pins", "pins": engine._pins_out()})
                continue
            if msg["cmd"] == "watch":  # the wearer named a thing: look for it from now on
                if msg["label"] not in engine.s.watchlist:
                    engine.s.watchlist.append(msg["label"])
                continue
            if msg["cmd"] == "want":  # asked for: the billed finder looks for it for a while
                if want := getattr(engine.finder, "want", None):
                    want(msg["label"])
                continue
            if msg["cmd"] == "hfov":  # the frame source changed lens (an iPhone, the glasses)
                if settings.map_intrinsics:
                    engine.runner.hfov_deg = msg["hfov"]
                    engine.solver.set_hfov(msg["hfov"])
                continue
            if msg["cmd"] == "mode":
                if engine.mode == "home" and len(engine._map_pts):
                    engine.save(save_path)  # leaving home: the house map stays on disk
                engine.set_mode(msg["mode"])
                out_q.put({"type": "dropped", "consumed": len(buf)})
                buf = []
                out_q.put({"type": "map_replace", "n": 0, "points": b"", "colors": b""})
                out_q.put({"type": "pins", "pins": []})
                out_q.put({"type": "mode", "mode": engine.mode})
                continue
            if msg["cmd"] == "kf":
                if not buf:
                    arrived = time.time()
                buf.append(
                    Keyframe(
                        msg["seq"],
                        msg["ts"],
                        msg["bgr"],
                        msg["depth"],
                        msg["depth_scale"],
                        msg["gravity"],
                        msg["taps"],
                        msg.get("hi"),
                    )
                )
        stale = buf and len(buf) >= 2 and time.time() - arrived > 4.0
        if len(buf) < need and not stale:
            continue
        consumed = len(buf)
        if len(buf) > per:  # backlog: spread the window over the whole backlog, keep the newest
            idx = np.linspace(0, len(buf) - 1, per).round().astype(int)
            chunk = [buf[i] for i in idx]
        else:
            chunk = buf
        buf = []
        # A source switch (portrait phone, landscape glasses) leaves frames of another size in the
        # backlog; a window is one batch, so it keeps only the newest frame's size.
        chunk = [k for k in chunk if k.bgr.shape == chunk[-1].bgr.shape]
        try:
            events = engine.process_window(chunk)
        except Exception as exc:  # keep the loop alive; the server shows the error
            out_q.put({"type": "memory_error", "error": repr(exc), "consumed": consumed})
            continue
        for e in events:
            if e["type"] == "memory_stats":
                e["consumed"] = consumed
            out_q.put(e)
        if engine.global_due():
            engine.global_start()
        if engine.mode == "home" and engine._windows % 10 == 0:
            engine.save(save_path)
            engine.persist_pins()
