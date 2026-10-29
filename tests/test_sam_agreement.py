"""A pin that rests on SAM 3.1 alone waits for a second sweep to agree: one box SAM drew round a
whole keyboard for "keys" (walk 20261028-135040, 16 s) must not walk the wearer to the desk."""

import numpy as np

from metasight.memory import Keyframe, MemoryEngine, _Placed
from metasight.pins import PinStore
from metasight.providers.finder import Detection

FRAME = np.zeros((64, 36, 3), np.uint8)


def engine(where: dict):
    """An engine whose finder returns `dets` and whose map puts each box at where[box]."""
    e = MemoryEngine.__new__(MemoryEngine)
    e.landmarks, e._kfs, e._sims, e._obs, e._sam_held = (
        None,
        {},
        [(1.0, np.eye(3), np.zeros(3))],
        [],
        [],
    )
    e.pins, e.s = PinStore(), type("S", (), {"watchlist": ["keys"]})()
    e._locate = lambda placed, box: where[box]
    e._near_landmarks = lambda pos, label: []
    return e


def sweep(e, ts: float, dets: list[Detection]) -> None:
    e.finder = type("F", (), {"find": lambda self, bgr, labels: dets, "name": "yoloe + sam-3.1"})()
    kf = Keyframe(int(ts), ts, FRAME, None, 1.0, None, [], None)
    e._pin_detections(_Placed(kf, None, None, np.eye(4)), run_finder=True)


A, B = (0.0, 0.0, 10.0, 10.0), (1.0, 1.0, 11.0, 11.0)


def keys(box, backed=False):
    return Detection("keys", 0.56, box, "sam-3.1", backed)


def test_one_sam_sweep_places_nothing_and_a_second_that_agrees_places_one_pin():
    e = engine({A: [1.0, 0.0, 2.0], B: [1.2, 0.0, 2.1]})
    sweep(e, 100.0, [keys(A)])
    assert not e.pins.pins
    sweep(e, 105.0, [keys(B)])  # 0.22 m from the first
    assert [p.label for p in e.pins.pins.values()] == ["keys"]


def test_two_sweeps_a_metre_apart_or_too_far_apart_in_time_place_nothing():
    e = engine({A: [1.0, 0.0, 2.0], B: [2.0, 0.0, 2.0]})
    sweep(e, 100.0, [keys(A)])
    sweep(e, 105.0, [keys(B)])  # 1 m away
    assert not e.pins.pins

    e = engine({A: [1.0, 0.0, 2.0], B: [1.1, 0.0, 2.0]})
    sweep(e, 100.0, [keys(A)])
    sweep(e, 140.0, [keys(B)])  # 40 s later: the first has lapsed
    assert not e.pins.pins


def test_two_boxes_in_one_frame_do_not_agree_with_each_other():
    e = engine({A: [1.0, 0.0, 2.0], B: [1.1, 0.0, 2.0]})
    sweep(e, 100.0, [keys(A), keys(B)])
    assert not e.pins.pins


def test_a_sam_box_backed_by_a_sure_local_hit_and_local_hits_pin_at_once():
    e = engine({A: [1.0, 0.0, 2.0], B: [3.0, 0.0, 2.0]})
    sweep(e, 100.0, [keys(A, backed=True), Detection("wallet", 0.5, B, "yoloe-26l-seg")])
    assert sorted(p.label for p in e.pins.pins.values()) == ["keys", "wallet"]
