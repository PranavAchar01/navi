"""Rooms and the footage ledger: which room a camera is in, seconds counted, needed frames."""

import numpy as np

from metasight import rooms


def cam(x, z, yaw_deg=0.0):
    p = np.eye(4)
    a = np.radians(yaw_deg)
    p[:3, 2] = [np.sin(a), 0, np.cos(a)]
    p[:3, 0] = [np.cos(a), 0, -np.sin(a)]
    p[:3, 3] = [x, 1.5, z]
    return p


def test_room_at():
    bed = rooms.Room("bedroom", np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float))
    live = rooms.Room("living", np.array([[4, 0], [9, 0], [9, 5], [4, 5]], float))
    assert rooms.room_at([bed, live], cam(1, 1)[:3, 3]) == "bedroom"
    assert rooms.room_at([bed, live], cam(6, 4)[:3, 3]) == "living"
    assert rooms.room_at([bed, live], cam(-1, 1)[:3, 3]) is None


def test_ledger_counts_seconds_and_stops_needing_a_full_view():
    led = rooms.Ledger()
    needed = [led.add("bedroom", 0.1, cam(1, 1), True) for _ in range(10)]
    assert needed == [True] * rooms.KEEP_PER_BUCKET + [False] * (10 - rooms.KEEP_PER_BUCKET)
    assert led.add("bedroom", 0.1, cam(1, 1, yaw_deg=90), True)  # a new heading is needed
    assert not led.add("bedroom", 0.1, cam(2, 2), False)  # blurred frames never are
    assert not led.add("bedroom", 60.0, cam(1, 1), True)  # a gap is not footage
    rep = led.report()["bedroom"]
    assert np.isclose(rep["hours"] * 3600, 12 * 0.1 + rooms.MAX_GAP_S, atol=4)
    assert rep["frames"] == 13 and rep["view_buckets"] == 2


def test_ledger_roundtrip(tmp_path):
    led = rooms.Ledger(walks=["w"])
    led.add("bedroom", 0.2, cam(0, 0), True)
    led.save(tmp_path / "l.json")
    back = rooms.Ledger.load(tmp_path / "l.json")
    assert back.seconds == led.seconds and back.walks == ["w"]


def test_watch_retry_gets_a_fresh_version(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import room_watch

    monkeypatch.setattr(room_watch, "run", lambda args, log: False)
    a = SimpleNamespace(anchors="a", views=1, anchor_walks=["w"], room="r")
    state = {"versions": ["seed"], "current": None, "done": {}, "tries": {}}
    assert not room_watch.fold(tmp_path / "walk", tmp_path, a, state)
    assert not room_watch.fold(tmp_path / "walk", tmp_path, a, state)
    assert sorted(p.name for p in tmp_path.glob("v*")) == ["v002", "v003"]
