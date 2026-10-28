import numpy as np

from metasight.pins import Pin, PinStore
from metasight.query import (
    direction_phrase,
    distance_phrase,
    is_question,
    parse_query,
    template_answer,
    time_ago,
    valid_sentence,
)


def pose_facing_minus_z():
    pose = np.eye(4)
    pose[:3, 2] = [0, 0, -1]
    pose[:3, 1] = [0, -1, 0]
    pose[:3, 0] = np.cross(pose[:3, 1], pose[:3, 2])
    pose[:3, 3] = [0, 1.4, 0]
    return pose


def pin(pos, near=(), ts=1000.0, label="keys"):
    return Pin(1, label, "object", list(pos), ts, ts, 3, 0.9, list(near))


def test_parse_the_keys_question_in_the_ways_people_ask_it():
    for q in [
        "where are my keys",
        "Where are my keys?",
        "hey, where did I leave my car keys",
        "find my keys",
    ]:
        assert parse_query(q) == "keys"
    assert parse_query("where's my phone") == "phone"
    assert parse_query("have you seen the tv remote") == "remote control"
    assert parse_query("what time is it") is None


def test_parse_uses_labels_from_the_map():
    assert parse_query("where is my passport", ["passport"]) == "passport"


def test_is_question():
    assert is_question("where are my keys")
    assert not is_question("turn on the lights")


def test_directions():
    assert direction_phrase(0) == "straight ahead"
    assert direction_phrase(50) == "ahead of you to the right"
    assert direction_phrase(-90) == "on your left"
    assert direction_phrase(135) == "behind you to the right"
    assert direction_phrase(-179) == "behind you"
    assert direction_phrase(200) == direction_phrase(-160)


def test_distances_and_times():
    assert distance_phrase(0.6) == "less than a metre"
    assert distance_phrase(1.2) == "about a metre"
    assert distance_phrase(4.1) == "about 4 metres"
    assert distance_phrase(4.4) == "about 4.5 metres"
    assert time_ago(10) == "just now"
    assert time_ago(125) == "2 minutes ago"
    assert time_ago(3 * 3600) == "3 hours ago"


def test_answer_places_the_keys_relative_to_you():
    text = template_answer(
        "keys", pin([0, 0.9, -4.0], near=["dining table"]), pose_facing_minus_z(), 1120.0
    )
    expected = "Your keys were near the dining table, about 4 metres straight ahead,"
    assert text == expected + " seen 2 minutes ago."


def test_answer_behind_and_to_the_left():
    text = template_answer("keys", pin([-3.0, 0.9, 3.0]), pose_facing_minus_z(), 1000.0)
    assert "behind you to the left" in text and "about 4 metres" in text


def test_answer_when_never_seen():
    assert template_answer("keys", None, pose_facing_minus_z(), 0).startswith(
        "I haven't seen your keys"
    )


def test_singular_objects_get_was():
    text = template_answer("phone", pin([0, 0.9, -2], label="phone"), pose_facing_minus_z(), 1000.0)
    assert text.startswith("Your phone was")


def test_valid_sentence_guard():
    assert valid_sentence("Your keys are on the sofa.")
    assert not valid_sentence("")
    assert not valid_sentence("one\ntwo")
    assert not valid_sentence("x" * 300)


def test_pin_store_merges_repeat_sightings_and_keeps_moved_objects_separate():
    s = PinStore()
    a = s.observe("keys", "object", [1.0, 0.9, 1.0], 100.0, 0.6)
    b = s.observe("keys", "object", [1.1, 0.9, 1.0], 110.0, 0.7)
    assert a is b and b.seen == 2 and b.ts == 110.0
    c = s.observe("keys", "object", [4.0, 0.9, 1.0], 200.0, 0.4)
    assert c is not a
    assert s.latest("keys") is a or s.latest("keys") is c
    assert len(s.pins) == 2


def test_latest_prefers_confirmed_sightings_over_a_single_weak_one():
    s = PinStore()
    s.observe("keys", "object", [1, 0, 1], 100.0, 0.6)
    s.observe("keys", "object", [1, 0, 1], 105.0, 0.6)
    s.observe("keys", "object", [5, 0, 5], 200.0, 0.2)  # one weak, newer sighting
    assert s.latest("keys").ts == 105.0


def test_a_look_alike_is_offered_as_unconfirmed():
    cup = pin([0.0, 0.9, -2.0], near=["desk"], label="cup")
    text = template_answer("water bottle", cup, pose_facing_minus_z(), 1000.0, stand_in=True)
    assert text.startswith("I'm not sure I saw your water bottle, but a cup was near the desk")
    assert "straight ahead" in text
