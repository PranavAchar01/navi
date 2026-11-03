"""Server and memory wiring that had only unit tests of the parts (docs/UNTESTED.md)."""

import asyncio
import queue
from types import SimpleNamespace

import numpy as np

from metasight.memory import MemoryEngine
from metasight.motion import FallDetector, ShakeDetector
from metasight.pins import Pin, PinStore


def test_a_tap_on_the_screen_pins_what_is_under_the_finger():
    """Tap-to-pin: the tapped point's map position becomes a sure pin with source "tap"."""
    eng = MemoryEngine.__new__(MemoryEngine)
    eng._kfs, eng._sims, eng._obs, eng.landmarks = {}, [(1.0, np.eye(3), np.zeros(3))], [], None
    eng.pins = PinStore()
    eng.pins.pins[7] = Pin(7, "door", "landmark", [1.0, 0.0, 2.2], 1.0, 1.0, 3, 0.9)
    h, w = 100, 200
    pts = np.zeros((h, w, 3), np.float32)
    pts[..., 0], pts[..., 2] = 1.2, 2.0  # everything in view sits at (1.2, 0, 2.0)
    kf = SimpleNamespace(
        seq=3, ts=50.0, bgr=np.zeros((h, w, 3), np.uint8), taps=[("keys", 0.5, 0.5)]
    )
    placed = SimpleNamespace(kf=kf, pts=pts, mask=np.ones((h, w), bool))
    assert MemoryEngine._pin_detections(eng, placed, run_finder=False) == 1
    pin = next(p for p in eng.pins.pins.values() if p.label == "keys")
    assert np.allclose(pin.pos, [1.2, 0.0, 2.0]) and pin.source == "tap" and pin.score == 1.0
    assert pin.near == ["door"]
    kf.taps = [("keys", 0.5, 0.5)]
    placed.mask[:] = False  # no map points under the finger: nothing is pinned
    assert MemoryEngine._pin_detections(eng, placed, run_finder=False) == 0


def _motion_server():
    from metasight.heading import Heading

    sent = []

    async def json(m, *_):
        sent.append(m)

    srv = SimpleNamespace(
        phone_clock=float("inf"),
        memory_ready=False,
        in_q=queue.Queue(),
        motion_t=0.0,
        accel=[],
        heading=Heading(60.0),
        fall=FallDetector(),
        shake=ShakeDetector(),
        hub=SimpleNamespace(json=json),
    )
    return srv, sent


def test_a_fall_reaches_the_screen_and_okay_cancels_it():
    """Fall check-in end to end through the server: motion rows in, an alert out, "fall_ok"."""
    from metasight.server import Server

    srv, sent = _motion_server()
    rows = []
    t = 0.0
    for _ in range(30):  # upright and still
        rows.append([t, 0.0, -9.8, 0.0])
        t += 0.02
    rows.append([t, 0.0, -40.0, 0.0])  # impact
    t += 0.02
    for _ in range(200):  # lying on its side, still
        rows.append([t, 9.8, 0.0, 0.0])
        t += 0.02
    asyncio.run(Server.on_motion(srv, rows))
    kinds = [m["kind"] for m in sent if m["type"] == "alert"]
    assert kinds and kinds[0].startswith("fall"), kinds
    n = len(sent)
    ok = srv.fall.respond_ok(srv.motion_t)
    assert ok and len(sent) == n  # the server sends these on a "fall_ok" message
