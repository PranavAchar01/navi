"""After a find the belt points at the item: from this walk's sighting when the answer named a
saved home pin not yet placed, at the home pin once the walk is relocalised onto the home map,
from an ARKit pose up to a minute old or a heading-only one up to 15 s, and never silently."""

import asyncio
import queue
import time
from types import SimpleNamespace

import numpy as np

from metasight.heading import Heading
from metasight.memory import HOME_ID
from metasight.pins import Pin, PinStore
from metasight.server import GUIDE_DR_S, GUIDE_POSE_S, NO_DIRECTION, Server


def server(pose_age_s: float, pins: list[Pin], ar_age_s: float = float("inf")):
    store = PinStore()
    store.pins = {p.id: p for p in pins}
    heading = Heading(60.0)
    heading.track(np.zeros(3), np.array([0.0, 0.0, 1.0]))  # at the origin, facing +z
    now = time.time()
    srv = SimpleNamespace(
        pins=store,
        heading=heading,
        map_pose_ts=now - pose_age_s,
        ar_pose_ts=now - ar_age_s,
        live_seen=(0.0, []),
        guide={"pin": pins[0].id, "label": "desk", "until": now + 90.0},
        found_until=0.0,
        _navigator=lambda *_: None,
    )
    srv._guide_pin = lambda g: Server._guide_pin(srv, g)
    srv.pose_ts = lambda: Server.pose_ts(srv)
    srv.guide_pose_fresh = lambda now: Server.guide_pose_fresh(srv, now)
    return srv, now


def desk(pid: int, pos, placed: bool, ts: float) -> Pin:
    return Pin(pid, "desk", "object", list(pos), ts, ts, seen=3, score=0.8, placed=placed)


def test_an_unplaced_home_pin_hands_over_to_this_walks_sighting():
    t = time.time()
    home = desk(1, [9.0, 0.0, 9.0], placed=False, ts=t - 3600)  # saved: other frame
    here = desk(2, [2.0, 0.0, 2.0], placed=True, ts=t - 20)  # this walk: 45 degrees off ahead
    srv, now = server(10.0, [home, here])
    g = guide(srv, now)
    assert g["bearing"] is not None and 40 < abs(g["bearing"]) < 50  # 45 degrees off ahead
    assert Server.can_point(srv, srv.guide, now)


def guide(srv, now):
    return Server._guide(srv, [(0, 0)] * 8, now, srv.guide_pose_fresh(now), [])


def test_an_arkit_pose_points_for_a_minute_heading_alone_for_15_s():
    t = time.time()
    here = [desk(2, [2.0, 0.0, 2.0], placed=True, ts=t - 20)]
    for pose_age, ar_age, points in [
        (GUIDE_DR_S - 5, float("inf"), True),  # glasses: memory pose, heading carried
        (GUIDE_DR_S + 5, float("inf"), False),  # too many steps since to trust it
        (GUIDE_POSE_S + 5, 30.0, True),  # phone: ARKit carried the pose
        (GUIDE_POSE_S + 5, GUIDE_POSE_S + 5, False),
    ]:
        srv, now = server(pose_age, here, ar_age)
        assert (guide(srv, now)["bearing"] is not None) is points, (pose_age, ar_age)
        assert Server.can_point(srv, srv.guide, now) is points


def test_a_relocalised_walk_guides_to_the_home_pin_with_no_sighting_this_walk():
    """Keys seen on an earlier walk: the answer names the saved pin, then the walk relocalises
    and memory sends the same pin moved into this walk's frame."""
    t = time.time()
    saved = desk(HOME_ID + 7, [9.0, 0.0, 9.0], placed=False, ts=t - 86400)
    srv, now = server(5.0, [saved])
    assert guide(srv, now)["bearing"] is None and not Server.can_point(srv, srv.guide, now)
    srv.pins.replace_all([{**saved.to_json(), "pos": [-2.0, 0.0, 2.0], "placed": True}])
    g = guide(srv, now)
    assert g["bearing"] is not None and 40 < abs(g["bearing"]) < 50  # at (-2, 2): 45 degrees off
    assert Server.can_point(srv, srv.guide, now)


def test_only_an_unplaced_pin_cannot_point_unless_the_camera_sees_it():
    t = time.time()
    srv, now = server(1.0, [desk(1, [9.0, 0.0, 9.0], placed=False, ts=t - 3600)])
    assert not Server.can_point(srv, srv.guide, now)
    srv.live_seen = (now, [("desk", 10.0, 1.5)])
    assert Server.can_point(srv, srv.guide, now)


def test_a_never_seen_answer_promises_no_buzz():
    t = time.time()
    srv, _ = server(GUIDE_POSE_S + 5, [desk(1, [9.0, 0.0, 9.0], placed=False, ts=t - 3600)])
    sent = []

    async def json(ev, *_):
        sent.append(ev)

    async def run(label, text):
        srv.loop = asyncio.get_running_loop()
        srv.map_loaded = asyncio.Event()
        srv.map_loaded.set()
        srv.hub, srv.in_q, srv.braille = SimpleNamespace(json=json), queue.Queue(), None
        srv.onboarding, srv.notes = SimpleNamespace(active=False), SimpleNamespace(recording=False)
        srv.known_labels, srv.can_point = lambda: ["desk"], lambda g, now: False
        done = srv.loop.create_future()
        done.set_result({"type": "answer", "label": label, "text": text, "pin": None})
        return await Server.answer(srv, "where is my umbrella", pending=done)

    never = asyncio.run(run(None, "I haven't seen your umbrella yet."))
    assert NO_DIRECTION not in never["text"]
    found = asyncio.run(run("desk", "I saw the desk 9 hours ago."))
    assert found["text"].endswith(NO_DIRECTION)
