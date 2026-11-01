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


def test_old_versions_lose_only_their_big_files(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import room_watch

    trash = tmp_path / "trash"
    trash.mkdir()
    monkeypatch.setattr(room_watch, "TRASH", trash)
    rdir = tmp_path / "bedroom"
    for v in ("v001", "v002", "v003"):
        for f in ("state.pt", "mesh-s1.ply", "objects-s1.json"):
            (rdir / v).mkdir(parents=True, exist_ok=True)
            (rdir / v / f).write_text(v)
    room_watch.prune(rdir, ["v001", "v002", "v003"])
    assert sorted(p.name for p in (rdir / "v001").iterdir()) == ["objects-s1.json"]
    assert len(list((rdir / "v002").iterdir())) == 3 and len(list((rdir / "v003").iterdir())) == 3
    assert sorted(p.name for p in trash.iterdir()) == [
        "bedroom-v001-mesh-s1.ply",
        "bedroom-v001-state.pt",
    ]


def test_a_move_needs_two_walks():
    seen = [{"label": "tv", "pos": [2.0, 1.0, 0.0], "views": 30}]
    moved = [{**seen[0], "from": [0.0, 1.0, 0.0]}]
    flags = rooms.track_changes([], moved, seen, "w1")
    assert [f["status"] for f in flags] == ["check"]
    flags = rooms.track_changes(flags, [], seen, "w2")  # w2 sees it at the new spot
    assert flags[0]["status"] == "moved" and flags[0]["walks"] == ["w1", "w2"]
    assert rooms.track_changes(flags, [], seen, "w3") == []  # told once, then gone


def test_an_unconfirmed_move_is_dropped():
    flags = rooms.track_changes(
        [], [{"label": "tv", "pos": [2.0, 1.0, 0.0], "from": [0.0, 1.0, 0.0]}], [], "w1"
    )
    other = [{"label": "tv", "pos": [0.0, 1.0, 0.0]}]  # back where the index had it
    flags = rooms.track_changes(flags, [], other, "w2")
    assert flags and flags[0]["misses"] == 1
    assert rooms.track_changes(flags, [], other, "w3") == []


def test_a_flag_needs_enough_views():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from room_objects import FLAG_VIEWS, changes

    ref = [{"label": "tv", "pos": [0.0, 1.0, 0.0]}]
    new = {"label": "tv", "pos": [2.0, 1.0, 0.0]}
    assert changes(ref, [{**new, "views": FLAG_VIEWS - 1}]) == []
    (flag,) = changes(ref, [{**new, "views": FLAG_VIEWS}])
    assert flag["from"] == [0.0, 1.0, 0.0] and flag["nearest_m"] == 2.0


def test_a_move_is_measured_against_how_sure_both_placements_are():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from room_objects import FLAG_VIEWS, changes

    def obj(label, x, spread):
        return {"label": label, "pos": [x, 0.5, 0.0], "spread_m": spread, "views": FLAG_VIEWS}

    # A keyboard placed to 5 cm that shifts 25 cm moved; a flat 30 cm missed it.
    assert changes([obj("keyboard", 0.0, 0.05)], [obj("keyboard", 0.25, 0.05)])
    # A wide monitor whose placements spread 18 cm, 43 cm off: still where it was.
    assert not changes([obj("tv", 0.0, 0.19)], [obj("tv", 0.43, 0.18)])
    # The chair test: pushed half a metre.
    assert changes([obj("chair", 0.0, 0.14)], [obj("chair", 0.5, 0.08)])
    assert not changes([obj("chair", 0.0, 0.14)], [obj("chair", 0.12, 0.08)])
