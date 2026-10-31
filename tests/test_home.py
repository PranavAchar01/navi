"""The home map: saved across walks, pins merged into it, a new obstacle flagged from depth."""

import cv2
import numpy as np
import pytest

from metasight import home as homemod
from metasight.memory import Keyframe, MemoryEngine, _Placed
from metasight.pins import PinStore, layer_of

EYE = np.eye(3)


def engine() -> MemoryEngine:
    e = MemoryEngine.__new__(MemoryEngine)
    e.mode, e.pins, e.trail = "home", PinStore(), []
    e._anchors = None
    e.home = None
    e._reset_home()
    return e


def pin(i, label, pos, ts=1000.0, seen=3, kind="object", layer=None, placed=True, score=0.6):
    return {
        "id": i, "label": label, "kind": kind, "pos": list(pos), "ts": ts, "first_ts": ts,
        "seen": seen, "score": score, "near": [], "thumb": "", "source": "",
        "layer": layer or layer_of(label, kind), "placed": placed,
    }  # fmt: skip


def test_layers():
    assert layer_of("door", "object") == "structure"
    assert layer_of("bed", "object") == "furniture"
    assert layer_of("chair", "landmark") == "furniture"
    assert layer_of("keys", "object") == "carried"


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pose = np.eye(4)
    v = homemod.HomeView(
        12.0,
        b"jpgbytes",
        rng.normal(size=(4, 5, 3)).astype(np.float16),
        np.ones((4, 5), bool),
        pose,
    )
    occ = homemod.cells(rng.uniform(0, 3, size=(200, 3)))
    homemod.save(homemod.Home([pin(1, "keys", [1, 1, 1])], [v, v], occ, 2, 5.0), tmp_path)
    back = homemod.load(tmp_path)
    assert back.walks == 2 and len(back.views) == 2 and back.views[1].jpg == b"jpgbytes"
    assert back.views[0].pts.shape == (4, 5, 3) and back.pins[0]["label"] == "keys"
    assert np.array_equal(back.occ, occ)
    homemod.save_pins([pin(1, "keys", [9, 9, 9])], tmp_path)
    assert homemod.load(tmp_path).pins[0]["pos"] == [9, 9, 9]


def test_fixed_home_keeps_views_and_stores_pins_in_its_frame(tmp_path, monkeypatch):
    """A home built from the room mesh: a walk saves its pins in the mesh's frame and leaves
    the mesh's views alone."""
    monkeypatch.setenv("METASIGHT_HOME", str(tmp_path))
    v = homemod.HomeView(
        1.0, b"mesh", np.zeros((4, 5, 3), np.float16), np.ones((4, 5), bool), np.eye(4)
    )
    homemod.save(
        homemod.Home([], [v], np.empty((0, 3), np.int32), 0, 1.0, {"mesh": "m.ply"}), tmp_path
    )
    e = engine()
    e.home = homemod.load(tmp_path)
    assert e.home.fixed == {"mesh": "m.ply"}
    e._home_tf = (2.0, EYE, np.array([1.0, 0.0, 0.0]))  # mesh frame -> walk frame
    e._home_prior = [pin(1, "keys", [1, 1, 1])]
    e._home_placed = [pin(100001, "keys", [3, 2, 2])]
    e._persist_home([0] * 8, [], [], [], [], np.eye(4), np.zeros((1, 3)), True)
    back = homemod.load(tmp_path)
    assert back.views[0].jpg == b"mesh" and back.fixed == {"mesh": "m.ply"}
    assert back.pins[0]["label"] == "keys" and np.allclose(back.pins[0]["pos"], [1, 1, 1])


def test_load_missing(tmp_path):
    assert homemod.load(tmp_path / "none") is None


def test_thin_spreads():
    poses = []
    for x in [0.0, 0.1, 0.2, 5.0, 5.1, 10.0]:
        p = np.eye(4)
        p[0, 3] = x
        poses.append(p)
    assert homemod.thin(poses, 3) == [0, 3, 5]
    assert homemod.thin(poses, 10) == list(range(6))


def test_home_pin_refreshed_by_walk():
    e = engine()
    e._home_placed = [pin(100001, "bed", [0, 0, 5], ts=100.0, seen=4, kind="landmark")]
    out = e._merge_home([pin(1, "bed", [0.4, 0, 5], ts=900.0, seen=2, kind="landmark")])
    assert len(out) == 1 and out[0]["ts"] == 900.0 and out[0]["seen"] == 6
    assert out[0]["pos"] == [0.4, 0, 5]  # furniture takes the current place


def test_structure_keeps_home_position():
    e = engine()
    e._home_placed = [pin(100001, "door", [2, 0, 5], seen=4, layer="structure")]
    out = e._merge_home([pin(1, "door", [2.3, 0, 5], seen=2)])
    assert out[0]["pos"] == [2, 0, 5]


def test_carried_item_moved_replaces_old_pin():
    e = engine()
    e._home_placed = [pin(100001, "keys", [1, 0.8, 1], ts=100.0)]
    out = e._merge_home([pin(1, "keys", [4, 0.9, 3], ts=900.0, seen=2)])
    assert [p["pos"] for p in out] == [[4, 0.9, 3]]
    moved = e._drain_moved()
    assert moved[0]["label"] == "keys" and moved[0]["change"] == "moved"
    assert e._drain_moved() == []


def test_lone_chair_moved_but_second_chair_is_new():
    e = engine()
    e._home_placed = [pin(100001, "chair", [0, 0, 0], kind="landmark")]
    out = e._merge_home([pin(1, "chair", [3, 0, 0], seen=3, kind="landmark")])
    assert len(out) == 1  # the only chair moved
    e = engine()
    e._home_placed = [
        pin(100001, "chair", [0, 0, 0], kind="landmark"),
        pin(100002, "chair", [8, 0, 0], kind="landmark"),
    ]
    out = e._merge_home([pin(1, "chair", [4, 0, 0], seen=3, kind="landmark")])
    assert len(out) == 3


def test_unfirm_walk_pin_does_not_erase_home_pin():
    e = engine()
    e._home_placed = [pin(100001, "keys", [1, 0.8, 1])]
    out = e._merge_home([pin(1, "keys", [4, 0.9, 3], seen=1, score=0.2)])
    assert len(out) == 2


def test_pins_before_placement_are_unplaced():
    e = engine()
    e._home_prior = [pin(1, "keys", [1, 1, 1], placed=True)]
    out = e._pins_out()
    assert out[0]["placed"] is False and out[0]["id"] == 100001


# ---------------------------------------------------------------- new obstacle
def grid(box: bool):
    """A camera at the origin facing +z in a 4 m wide room: wall at z = 3.5, a box at z = 2."""
    xs, ys = np.meshgrid(np.linspace(-2, 2, 160), np.linspace(0.1, 1.8, 80))
    wall = np.stack([xs, ys, np.full_like(xs, 3.5)], axis=-1)
    if box:
        inside = (np.abs(xs) < 0.3) & (ys < 1.0)
        wall = wall.copy()
        wall[inside, 2] = 2.0
    return wall


def with_home(e: MemoryEngine):
    e._anchors = ((1.0, EYE, np.zeros(3)), np.zeros((1, 3)), np.eye(4)[None])
    e._home_tf = (1.0, EYE, np.zeros(3))
    side = np.concatenate(
        [
            np.stack([x, y, np.full_like(x, 3.5)], -1).reshape(-1, 3)
            for x, y in [np.meshgrid(np.linspace(-3, 3, 120), np.linspace(0.05, 1.9, 40))]
        ]
    )
    e._home_occ = homemod.cells(side)
    e._home_occ_keys = np.sort(e._cell_keys(e._home_occ))


def placed(box: bool, ts: float) -> _Placed:
    pts = grid(box)
    pose = np.eye(4)
    pose[:3, :3] = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    kf = Keyframe(1, ts, np.zeros((80, 160, 3), np.uint8))
    return _Placed(kf, pts, np.ones(pts.shape[:2], bool), pose)


def test_empty_room_flags_nothing():
    e = engine()
    with_home(e)
    assert e._change_check(placed(False, 10.0)) == []
    assert e._change_check(placed(False, 12.0)) == []


def test_box_flagged_on_second_window_with_bearing():
    e = engine()
    with_home(e)
    assert e._change_check(placed(True, 10.0)) == []  # one window is not enough
    out = e._change_check(placed(True, 12.0))
    assert len(out) == 1 and out[0]["change"] == "new_obstacle"
    assert out[0]["distance_m"] == pytest.approx(2.0, abs=0.35)
    assert abs(out[0]["bearing_deg"]) < 15
    assert e._change_check(placed(True, 14.0)) == []  # said once, then the cooldown


def test_unmapped_surroundings_flag_nothing():
    e = engine()
    with_home(e)
    e._home_occ = e._home_occ[:50]
    e._home_occ_keys = np.sort(e._cell_keys(e._home_occ))
    assert e._change_check(placed(True, 10.0)) == []
    assert e._change_check(placed(True, 12.0)) == []


def test_fixed_home_frame_from_agreed_cameras(tmp_path, monkeypatch):
    """Cameras agreed in the room give room -> walk frame, but only once they spread out."""
    monkeypatch.setenv("METASIGHT_HOME", str(tmp_path))
    e = engine()
    e.poses = {}
    a = np.radians(30)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    true = (1.5, rot, np.array([0.5, 0.0, -2.0]))  # room -> walk
    rng = np.random.default_rng(0)

    def place(q, x):  # a camera at x in the room, facing some way, and where the walk has it
        cam = np.eye(4)
        cam[:3, :3] = cv2.Rodrigues(rng.normal(0, 0.5, 3))[0]
        cam[:3, 3] = x
        e._room_pairs[q] = cam
        p = np.eye(4)
        p[:3, :3], p[:3, 3] = true[1] @ cam[:3, :3], true[0] * true[1] @ x + true[2]
        e.poses[q] = p

    for q, x in enumerate(rng.uniform([0, 1.2, 0], [0.3, 1.4, 0.3], (8, 3))):
        place(q, x)  # close together: not enough
    assert e._room_tf() is None
    for q, x in enumerate(rng.uniform([0, 1.2, 0], [3, 1.4, 3], (8, 3)), start=8):
        place(q, x)
    s_, r_, t_ = e._room_tf()
    assert np.isclose(s_, 1.5) and np.allclose(r_, rot, atol=1e-6) and np.allclose(t_, true[2])


def test_room_flags_are_spoken_once_when_near():
    from types import SimpleNamespace

    e = engine()
    flag = {"label": "tv", "from": [0.0, 1.0, 0.0], "to": [3.0, 1.0, 0.0], "status": "check"}
    e.home = homemod.Home([], [], np.empty((0, 3), np.int32), changes=[flag])
    e._anchors = ((1.0, np.eye(3), np.zeros(3)), np.zeros((1, 3)), [np.eye(4)])
    e._set_home_tf((1.0, np.eye(3), np.array([1.0, 0.0, 0.0])))  # tv lands at x = 4
    far, near = np.eye(4), np.eye(4)
    near[0, 3] = 3.0
    e._near_flags(SimpleNamespace(pose=far))
    assert e._drain_moved() == []
    e._near_flags(SimpleNamespace(pose=near))
    e._near_flags(SimpleNamespace(pose=near))
    (said,) = e._drain_moved()
    assert said["change"] == "maybe_moved" and said["to"] == [4.0, 1.0, 0.0]


def test_room_placement_needs_cameras_from_two_tries_that_agree():
    import numpy as np

    from metasight.align import apply_sim3
    from metasight.memory import room_consensus

    rng = np.random.default_rng(1)
    out = rng.uniform(-2, 2, (24, 3))
    rot = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    room = apply_sim3(out, 1.2, rot, np.array([1.0, 0, -2]))
    room += rng.normal(0, 0.05, room.shape)
    tries = np.repeat([1, 2, 3], 8)
    room[16:] += [2.0, 0, 1.0]  # the third try landed in the wrong place
    agree = room_consensus(out, room, tries)
    assert agree is not None and agree[:16].all() and not agree[16:].any()
    assert room_consensus(out[:8], room[:8], tries[:8]) is None  # one try confirms nothing


def test_a_frame_matched_to_a_saved_view_lands_on_that_views_camera():
    import cv2

    from metasight.relocalize import STRIDE, RoomMatcher

    h, w, f = 518, 294, 300.0
    rng = np.random.default_rng(3)
    img = np.full((h, w, 3), 200, np.uint8)
    for _ in range(300):  # a poster wall: shapes give SIFT corners, noise does not
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        cv2.circle(
            img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 15)), c, -1
        )
    # A wall 2 m in front of a camera at (1, 0.5, -3) looking down +z.
    v, u = np.mgrid[0:h:STRIDE, 0:w:STRIDE]
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 0.5, -3.0]
    pts = np.stack([(u - w / 2) / f * 2.0, (v - h / 2) / f * 2.0, np.full(u.shape, 2.0)], -1)
    view = homemod.HomeView(
        0.0, b"", (pts + pose[:3, 3]).astype(np.float16), np.ones(u.shape, bool), pose
    )
    m = RoomMatcher([img], [view])
    assert np.allclose(np.diag(m.k)[:2], f, rtol=0.02)
    hit = m.locate(img, [0])
    assert hit is not None and np.linalg.norm(hit[0][:3, 3] - pose[:3, 3]) < 0.05 and hit[1] >= 40
    assert np.allclose(hit[0][:3, :3], pose[:3, :3], atol=0.02)
    assert m.locate(np.full_like(img, 128), [0]) is None  # nothing to match, no guess
