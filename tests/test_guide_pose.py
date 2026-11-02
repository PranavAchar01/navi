"""After a find the belt points at the item: from this walk's sighting when the answer named a
saved home pin not yet placed, at the home pin once the walk is relocalised onto the home map,
from an ARKit pose up to a minute old or a heading-only one up to 15 s, and never silently."""

import asyncio
import math
import queue
import time
from types import SimpleNamespace

import numpy as np

from metasight.heading import Heading
from metasight.memory import HOME_ID
from metasight.pins import Pin, PinStore
from metasight.review import bank
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
        s=SimpleNamespace(found_m=0.8),
        path_m=math.inf,
    )
    srv._guide_pin = lambda g: Server._guide_pin(srv, g)
    srv.pose_ts = lambda: Server.pose_ts(srv)
    srv.guide_pose_fresh = lambda now: Server.guide_pose_fresh(srv, now)
    srv._lost_close = lambda g, now: Server._lost_close(srv, g, now)
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


def keys(pid: int, pos, placed: bool, ts: float) -> Pin:
    return Pin(pid, "keys", "object", list(pos), ts, ts, seen=2, score=0.6, placed=placed)


def test_guide_switches_mid_run_to_a_newer_placed_pin_and_never_back():
    """Compare's keys replay: the answer named an old keys pin memory had not placed, so the
    belt had no route; memory placed this walk's keys later and the guide must move to them."""
    t = time.time()
    old = keys(HOME_ID + 15, [9.0, 0.0, 9.0], placed=False, ts=t - 9 * 3600)
    srv, now = server(5.0, [old])
    srv.guide = {"pin": old.id, "label": "keys", "until": now + 90.0, "seen": now - 1}
    routed = []
    srv._navigator = lambda key, pin: routed.append(key)
    assert guide(srv, now)["bearing"] is None and routed == []

    found = keys(40, [2.0, 0.0, 2.0], placed=True, ts=t - 2)  # memory places them mid-guide
    srv.pins.replace_all([old.to_json(), found.to_json()])
    g = guide(srv, now)
    assert g["bearing"] is not None and srv.guide["pin"] == 40 and routed[-1] == 40
    assert "seen" not in srv.guide  # the lock on the old bearing is dropped

    other = keys(41, [-2.0, 0.0, 2.0], placed=True, ts=t - 30)  # an older placed sighting
    srv.pins.replace_all([old.to_json(), found.to_json(), other.to_json()])
    guide(srv, now)
    assert srv.guide["pin"] == 40  # not strictly newer: stays

    seen_again = {**other.to_json(), "ts": t}  # the other one seen again, now the newest
    srv.pins.replace_all([old.to_json(), found.to_json(), seen_again])
    guide(srv, now)
    assert srv.guide["pin"] == 41


def test_a_never_seen_thing_is_wanted_so_the_camera_looks_for_it():
    """'Where is my umbrella', never seen: the answer is unchanged, and the worker is told to
    look for an umbrella from now (SAM 3.1 sweeps it, the local model learns the word)."""
    t = time.time()
    srv, _ = server(GUIDE_POSE_S + 5, [desk(1, [9.0, 0.0, 9.0], placed=True, ts=t - 60)])
    srv.language, srv.live_last = SimpleNamespace(name="template"), {}
    res = Server._answer_sync(srv, "where is my umbrella")
    assert res["label"] is None and res["want"] == "umbrella"
    assert res["text"].startswith("I haven't seen your umbrella yet.")
    assert "want" not in Server._answer_sync(srv, "where is it")  # names nothing: nothing wanted

    sent = []

    async def json(ev, *_):
        sent.append(ev)

    async def run():
        srv.loop = asyncio.get_running_loop()
        srv.map_loaded = asyncio.Event()
        srv.map_loaded.set()
        srv.hub, srv.in_q, srv.braille = SimpleNamespace(json=json), queue.Queue(), None
        srv.onboarding, srv.notes = SimpleNamespace(active=False), SimpleNamespace(recording=False)
        srv.known_labels, srv.can_point = lambda: ["desk"], lambda g, now: False
        done = srv.loop.create_future()
        done.set_result(res)
        await Server.answer(srv, "where is my umbrella", pending=done)
        return srv.in_q.get_nowait()

    assert asyncio.run(run()) == {"cmd": "want", "label": "umbrella", "new": True}
    assert sent[-1]["text"] == res["text"]


def saved_keys(score: float, source: str = "yoloe-26l-seg") -> Pin:
    """Shaped like the saved house's keys pin 100015, which is the keyboard."""
    t = time.time() - 9 * 3600
    return Pin(HOME_ID + 15, "keys", "object", [2.0, 0.0, 2.0], t, t - 600, 13, score,
               ["sink", "TV"], source=source, layer="carried")  # fmt: skip


def ask_keys(srv):
    """The answer event as it goes out, and the guide it left."""
    srv.language, srv.live_last, srv.pose = SimpleNamespace(name="template"), {}, None
    sent = []

    async def json(ev, *_):
        sent.append(ev)

    async def set_activity(*_, **__):
        pass

    async def run():
        srv.loop = asyncio.get_running_loop()
        srv.map_loaded = asyncio.Event()
        srv.map_loaded.set()
        srv.hub, srv.in_q, srv.braille = SimpleNamespace(json=json), queue.Queue(), None
        srv.onboarding, srv.notes = SimpleNamespace(active=False), SimpleNamespace(recording=False)
        srv.known_labels, srv.set_activity = lambda: ["keys"], set_activity
        srv.can_point = lambda g, now: Server.can_point(srv, g, now)
        done = srv.loop.create_future()
        done.set_result(Server._answer_sync(srv, "where are my keys"))
        return await Server.answer(srv, "where are my keys", pending=done)

    asyncio.run(run())
    return sent[-1]


def test_a_weak_local_keys_pin_sam_never_saw_is_not_answered_or_routed_to():
    """Pin 100015: yoloe at 0.19, seen 13 times, never confirmed by SAM 3.1, and it is the
    keyboard. Find says the keys were not seen, Guide never routes to it, errand plans leave it
    out, and the saved pin itself is left exactly as it was."""
    pin = saved_keys(0.19)
    before = pin.to_json()
    srv, now = server(5.0, [pin])
    srv.pins.items, srv.guide = frozenset({"keys"}), None
    ev = ask_keys(srv)
    assert ev["text"].startswith("I haven't seen your keys")
    assert ev["pin"] is None and "pin_id" not in ev and srv.guide is None
    assert ev["skipped_pin_id"] == pin.id  # so verify can see why
    # A guide already running for keys does not switch onto it either.
    srv.guide = {"pin": None, "label": "keys", "until": now + 90.0}
    assert Server._guide_pin(srv, srv.guide) is None
    assert "keys" not in bank(srv.pins, now)
    assert pin.to_json() == before


def test_a_sure_or_sam_confirmed_keys_pin_is_answered_and_routed_as_before():
    for pin in (saved_keys(0.6), saved_keys(0.19, source="sam-3.1")):
        srv, now = server(5.0, [pin])
        srv.pins.items, srv.guide = frozenset({"keys"}), None
        ev = ask_keys(srv)
        assert ev["text"].startswith("Your keys were last seen near the sink"), ev["text"]
        assert ev["pin_id"] == pin.id and ev["confirmed"] is True
        assert "skipped_pin_id" not in ev
        assert srv.guide["pin"] == pin.id
        assert guide(srv, now)["bearing"] is not None


def test_the_guide_message_names_the_pin_turn_and_distance():
    """The fields the web guide arc and the belt follow (GAPS #5)."""
    t = time.time()
    srv, now = server(5.0, [desk(2, [2.0, 0.0, 2.0], placed=True, ts=t - 20)])
    cmd = [(0, 0)] * 8
    g = Server._guide(srv, cmd, now, True, [])
    assert g["pin_id"] == 2 and g["dist_m"] == round(2 * 2**0.5, 2)
    assert 40 < abs(g["bearing_deg"]) < 50
    assert g["turn"] == ("right" if g["bearing_deg"] > 0 else "left")
    assert any(m == 5 for m, _ in cmd)  # the belt's guide pattern is on

    srv.pins.pins[3] = desk(3, [0.0, 0.0, 3.0], placed=True, ts=t - 30)
    srv.guide = None
    assert Server.guide_to(srv, 3) and srv.guide["pin"] == 3
    g = guide(srv, now)
    assert g["pin_id"] == 2  # a newer placed desk takes over, as after an answer
    srv.guide = {"pin": 3, "label": "desk", "until": now + 90}
    srv.pins.pins.pop(2)
    g = guide(srv, now)
    assert g["pin_id"] == 3 and g["turn"] == "ahead" and abs(g["bearing_deg"]) < 1
    assert not Server.guide_to(srv, 99) and Server.guide_to(srv, None) and srv.guide is None


def test_walking_up_to_the_item_until_it_leaves_the_view_counts_as_found():
    """The chair drops out of the bottom of the view in the last step: found, once it has been
    gone half a second with the head still facing it. A turn away, a single missed frame or a
    last sighting from across the room is not found."""
    t = time.time()
    seen_box = SimpleNamespace(label="desk", box=(0.45, 0.4, 0.55, 0.9), dist_m=1.1)
    for gap, turned_deg, last_dist, found in [
        (0.8, 5.0, 1.1, True),
        (0.1, 5.0, 1.1, False),  # one missed detection
        (0.8, 60.0, 1.1, False),  # turned away
        (0.8, 5.0, 2.5, False),  # lost from across the room
        (3.0, 5.0, 1.1, False),  # long gone: the guide carries on as before
    ]:
        srv, now = server(10.0, [desk(2, [0.0, 0.0, 9.0], placed=True, ts=t - 20)])
        srv.loop = SimpleNamespace(create_task=lambda c: c.close())
        srv.hub = SimpleNamespace(json=lambda *a: asyncio.sleep(0))
        Server._guide(srv, [(0, 0)] * 8, now - gap, False, [replace_dist(seen_box, last_dist)])
        srv.heading.turn += math.radians(turned_deg)
        Server._guide(srv, [(0, 0)] * 8, now, False, [])
        assert (srv.guide is None) is found, (gap, turned_deg, last_dist)


def replace_dist(o, d):
    return SimpleNamespace(label=o.label, box=o.box, dist_m=d)


def test_a_sighting_points_the_guide_and_a_blink_holds_it_turned_with_the_head():
    """In view, the camera gives the direction; a frame or two later with no detection the last
    sighting still points, moved by the head's turn since; three seconds on it stops."""
    t = time.time()
    srv, now = server(10.0, [desk(2, [0.0, 0.0, 9.0], placed=False, ts=t - 20)])
    chair = SimpleNamespace(label="desk", box=(0.45, 0.4, 0.55, 0.9), dist_m=2.5)
    g = Server._guide(srv, [(0, 0)] * 8, now - 0.4, False, [chair])
    assert g["in_view"] and g["dist"] == 2.5 and abs(g["bearing"]) < 5
    srv.heading.turn += math.radians(20)  # the head turned 20 degrees left: it is now right
    g = Server._guide(srv, [(0, 0)] * 8, now, False, [])
    assert not g["in_view"] and 15 < g["bearing"] < 25
    g = Server._guide(srv, [(0, 0)] * 8, now + 4.0, False, [])
    assert g["bearing"] is None
