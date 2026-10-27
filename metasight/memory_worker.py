"""The memory loop in its own process: its own MPS context, so a 3 s window never stalls the
10-20 Hz safety loop in the server process. Talks over two queues of plain dicts."""

from __future__ import annotations

import queue
import time
from pathlib import Path

import numpy as np

from .config import SESSIONS, Settings
from .memory import Keyframe, MapAnythingRunner, MemoryEngine
from .providers import make_finder


def run(in_q, out_q, settings: Settings, session: str, new_needed: int = 4) -> None:
    from ultralytics import YOLO

    engine = MemoryEngine(
        settings,
        MapAnythingRunner(),
        make_finder(settings),
        YOLO(str(settings.weights_dir / "yolo26n.pt")),
        sam_every=2 if settings.finder == "sam31" else 1,
    )
    SESSIONS.mkdir(exist_ok=True)
    save_path = Path(SESSIONS) / f"{session}.npz"
    out_q.put({"type": "ready", "finder": engine.finder.name})
    buf: list[Keyframe] = []
    arrived = 0.0
    per = settings.window - settings.overlap
    while True:
        try:
            msg = in_q.get(timeout=0.25)
        except queue.Empty:
            msg = None
        if msg is not None:
            if msg["cmd"] == "stop":
                engine.save(save_path)
                return
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
                    )
                )
        stale = buf and len(buf) >= 2 and time.time() - arrived > 4.0
        if len(buf) < new_needed and not stale:
            continue
        consumed = len(buf)
        if len(buf) > per:  # backlog: spread the window over the whole backlog, keep the newest
            idx = np.linspace(0, len(buf) - 1, per).round().astype(int)
            chunk = [buf[i] for i in idx]
        else:
            chunk = buf
        buf = []
        try:
            events = engine.process_window(chunk)
        except Exception as exc:  # keep the loop alive; the server shows the error
            out_q.put({"type": "memory_error", "error": repr(exc)})
            continue
        for e in events:
            if e["type"] == "memory_stats":
                e["consumed"] = consumed
            out_q.put(e)
        if engine._windows % 10 == 0:
            engine.save(save_path)
