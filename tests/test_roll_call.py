import numpy as np
import pytest

from metasight import roll_call as rc
from metasight import spatial_sound as ss

HEAD = np.array([0.0, 1.6, 0.0])
FWD = np.array([0.0, 0.0, -1.0])
UP = np.array([0.0, 1.0, 0.0])


def to(p):
    d = ss.relative(HEAD, FWD, UP, p)
    return d.ground_m, d.az_deg


def pin(label, x, z, seen=50, score=0.8, placed=True):
    return {"label": label, "pos": [x, 0.7, z], "seen": seen, "score": score, "placed": placed}


ROOM = [
    pin("bed", 1.7, -1.0),  # ahead right: about two o'clock
    pin("desk", -1.0, -0.6),  # ahead left: ten o'clock
    pin("desk", -1.3, -0.5, seen=30),  # the same desk again
    pin("door", 0.0, 2.0),  # behind
    pin("chair", 0.0, -1.0),  # straight ahead
    pin("person", 0.5, -0.5),  # people move: never named
    pin("laptop", 1.6, -1.0, seen=200),  # on the bed: never named apart from it
    pin("lamp", -1.5, 1.5, seen=90),  # small things only fill spare places
    pin("fan", 1.0, 1.0, seen=5),  # too few sightings
    pin("cabinet", -2.0, 0.0, score=0.2),  # too unsure
    pin("tv", 2.0, 0.0, placed=False),  # not in this walk's frame
]


def test_pick_names_the_layout_clockwise_from_ahead():
    items = rc.pick(ROOM, to, k=4)
    assert [i.label for i in items] == ["chair", "bed", "door", "desk"]
    desk = items[-1]
    assert desk.pos[0] == pytest.approx(-1.0)  # the most-seen desk, once
    assert rc.pick(ROOM, to, k=6)[-2].label != "person"
    five = [i.label for i in rc.pick(ROOM, to, k=6)]
    assert "lamp" in five and "laptop" not in five and "fan" not in five


def test_no_pose_names_nothing():
    assert rc.pick(ROOM, lambda p: None) == []


@pytest.mark.parametrize(
    "bearing,said",
    [(0, "twelve"), (14, "twelve"), (16, "one"), (60, "two"), (90, "three"), (180, "six"),
     (-90, "nine"), (-60, "ten"), (-16, "eleven"), (-170, "six")],
)  # fmt: skip
def test_clock(bearing, said):
    assert rc.clock(bearing) == f"{said} o'clock"


def test_metres():
    assert rc.metres(0.3) == "half a metre"
    assert rc.metres(1.1) == "one metre"
    assert rc.metres(2.3) == "two and a half metres"
    assert rc.metres(3.0) == "three metres"


def test_line_and_message_point_at_each_name():
    items = rc.pick(ROOM, to, k=3)
    text, spans = rc.line(items, "bedroom")
    # three places: door, bed, desk outrank the chair; still read clockwise from ahead
    assert text == (
        "Bedroom. Bed at two o'clock, two metres. Door at six o'clock, two metres. "
        "Desk at ten o'clock, one metre."
    )
    for it, (s, n) in zip(items, spans, strict=True):
        assert text[s : s + n].lower() == it.label
    msg = rc.message(items, "bedroom")
    assert msg["type"] == "rollcall" and msg["items"][0]["bearing"] > 0
    assert rc.line([], None)[0].startswith("I don't know")


def test_render_plucks_from_each_side():
    left = [rc.Item("desk", (-2.0, 1.6, 0.0), 2.0, -90.0)]
    st = rc.render(left, HEAD, FWD, UP)
    e = (st.astype(float) ** 2).sum(0)
    assert e[0] > 2 * e[1]
    voice = [np.full(4410, 0.1), np.full(8820, 0.1)]  # room name, then the one item
    st = rc.render(left, HEAD, FWD, UP, voice=voice)
    assert len(st) > (4410 + 8820) and np.abs(st).max() < 1


def test_taps_follow_the_names():
    msg = rc.message(rc.pick(ROOM, to, k=3), "bedroom")
    t = rc.taps(msg)
    assert [b for _, b in t] == [it["bearing"] for it in msg["items"]]
    assert t[0][0] == pytest.approx(9 / rc.SPEECH_CHARS_PER_S) and t[0][0] < t[1][0] < t[2][0]
