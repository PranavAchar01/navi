import numpy as np

from metasight.pins import Pin, PinStore
from metasight.query import (
    asked_for,
    direction_phrase,
    distance_phrase,
    is_question,
    live_answer,
    parse_query,
    template_answer,
    time_ago,
    unseen_answer,
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


def test_plurals_find_their_label_and_nothing_else():
    known = ["shoe", "bed", "headphones", "book", "keys", "glasses case", "chair"]
    assert parse_query("Where are my shoes?", known) == "shoe"
    assert parse_query("where's my shoe", known) == "shoe"
    assert parse_query("where are my headphones", known) == "headphones"
    assert parse_query("where is my headphone", known) == "headphones"
    assert parse_query("where are my books", known) == "book"
    assert parse_query("where are my keys", known) == "keys"
    assert parse_query("where's my key", known) == "keys"
    assert parse_query("where are my glasses", known) == "glasses case"
    assert parse_query("where are my beds", known) == "bed"
    assert parse_query("where are my socks", known) is None  # never a different thing


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
    text = template_answer("keys", None, pose_facing_minus_z(), 0)
    assert text.startswith("I haven't seen your keys") and "Find My" in text


def test_an_unknown_thing_is_still_named_back():
    assert asked_for("Where is my umbrella?") == "umbrella"
    assert asked_for("where are my reading glasses") == "reading glasses"
    assert asked_for("what's that noise") is None
    assert "If it has an AirTag" in unseen_answer("umbrella")
    assert "If they have an AirTag" in unseen_answer("keys")


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


def test_a_fair_sighting_long_after_the_firm_pin_means_it_moved():
    """The old house pin (seen twice, hours ago) must not outvote this walk's one clear look."""
    s = PinStore()
    s.observe("laptop", "object", [1, 0, 1], 100.0, 0.6)
    s.observe("laptop", "object", [1, 0, 1], 105.0, 0.6)
    s.observe("laptop", "object", [5, 0, 5], 105.0 + 6 * 3600, 0.4)
    assert s.latest("laptop").pos == [5, 0, 5]
    s.observe("laptop", "object", [9, 0, 9], 106.0 + 6 * 3600, 0.2)  # weak: not enough
    assert s.latest("laptop").pos == [5, 0, 5]


def test_a_drifted_lone_sighting_on_the_same_visit_folds_into_the_pin():
    """Map drift drops a second laptop pin 1.6 m off on the same pass: one laptop, one pin. The
    same lone look hours later stays apart, so the moved rule can still use it."""
    s = PinStore()
    s.observe("laptop", "object", [0, 0, 0], 100.0, 0.6)
    s.observe("laptop", "object", [0, 0, 0], 102.0, 0.6)
    s.observe("laptop", "object", [1.6, 0, 0], 130.0, 0.3)
    s.observe("cup", "object", [0, 0, 0], 100.0, 0.6)
    s.observe("cup", "object", [0, 0, 0], 101.0, 0.6)
    s.observe("cup", "object", [1.6, 0, 0], 102.0, 0.6)  # firm on its own: a second cup
    s.observe("cup", "object", [1.6, 0, 0], 103.0, 0.6)
    s.merge(1.0)
    assert len([p for p in s.pins.values() if p.label == "laptop"]) == 1
    assert len([p for p in s.pins.values() if p.label == "cup"]) == 2

    s.observe("laptop", "object", [1.6, 0, 0], 130.0 + 3600, 0.3)
    s.merge(1.0)
    assert len([p for p in s.pins.values() if p.label == "laptop"]) == 2
    assert s.latest("laptop").pos == [1.6, 0, 0]


def test_a_look_alike_is_offered_as_unconfirmed():
    cup = pin([0.0, 0.9, -2.0], near=["desk"], label="cup")
    text = template_answer("water bottle", cup, pose_facing_minus_z(), 1000.0, stand_in=True)
    assert text.startswith("I haven't seen your water bottle. I did see a cup near the desk")
    assert "straight ahead" in text and "maybe" in text and text.endswith("play a sound.")


def test_a_never_seen_wallet_is_not_answered_with_the_handbag():
    from types import SimpleNamespace as NS

    from metasight.pins import PinStore
    from metasight.server import Server

    s = PinStore()
    s.pins[1] = pin([0.0, 0.9, -3.0], near=["sink"], label="handbag")
    s.pins[1].seen = 5
    srv = NS(
        pins=s,
        language=NS(name="template"),
        live_seen=(0.0, []),
        live_last={},
        pose=None,
        map_pose_ts=0.0,
        guide=None,
    )
    text = Server._answer_sync(srv, "where is my wallet")["text"]
    assert text.startswith("I haven't seen your wallet.") and "handbag" in text
    assert "Find My" in text


def test_plural_shoes_answer_with_the_shoe_not_the_bed():
    """Row 5 proof: "Where are my shoes?" came back with the bed pin."""
    from types import SimpleNamespace as NS

    from metasight.server import Server

    s = PinStore()
    s.pins[1] = Pin(1, "shoe", "object", [0.0, 0.0, -2.0], 1000.0, 1000.0, 3, 0.8, ["sink"])
    s.pins[2] = Pin(2, "bed", "object", [3.0, 0.0, -2.0], 1000.0, 1000.0, 1, 0.9, ["wall"])
    srv = NS(
        pins=s,
        language=NS(name="template"),
        live_seen=(0.0, []),
        live_last={},
        pose=None,
        map_pose_ts=0.0,
        guide=None,
    )
    res = Server._answer_sync(srv, "Where are my shoes?")
    assert res["label"] == "shoe" and "sink" in res["text"] and "bed" not in res["text"]


def test_a_model_guess_is_only_offered_as_a_maybe():
    from types import SimpleNamespace as NS

    from metasight.server import Server

    s = PinStore()
    s.pins[2] = Pin(2, "bed", "object", [3.0, 0.0, -2.0], 1000.0, 1000.0, 1, 0.9, ["wall"])
    srv = NS(
        pins=s,
        language=NS(name="muse-spark", complete=lambda *a, **k: "bed"),
        live_seen=(0.0, []),
        live_last={},
        pose=None,
        map_pose_ts=0.0,
        guide=None,
    )
    text = Server._answer_sync(srv, "where are my sneakers")["text"]
    assert text.startswith("I haven't seen your sneakers.") and "maybe" in text


def test_live_answer_names_the_live_place():
    assert live_answer("laptop", 1.5, -40) == (
        "Your laptop is about 1.5 metres ahead of you to the left, in view now."
    )
    assert live_answer("keys", 0.6, 5) == "Your keys are right in front of you, in view now."


def test_a_live_sighting_beats_an_old_memory():
    """Cycle 10: "near the TV, seen six hours ago" while the laptop was in frame."""
    import time
    from types import SimpleNamespace

    from metasight.server import Server

    store = PinStore()
    old = Pin(
        id=1,
        label="laptop",
        kind="object",
        pos=[2.0, 1.0, -3.0],
        ts=time.time() - 6 * 3600,
        first_ts=time.time() - 7 * 3600,
        seen=3,
        score=0.9,
        near=["TV"],
    )
    store.pins[old.id] = old
    srv = SimpleNamespace(
        pins=store,
        language=SimpleNamespace(name="template"),
        pose=None,
        map_pose_ts=0.0,
        guide=None,
        live_seen=(time.time(), [("laptop", 3.0, 1.4)]),
        live_last={},
    )
    res = Server._answer_sync(srv, "where is my laptop")
    assert res["live"] and "in view now" in res["text"] and "hours" not in res["text"]
    assert srv.guide["label"] == "laptop"

    srv.live_seen = (time.time() - 5, [("laptop", 3.0, 1.4)])  # out of frame for a while
    res = Server._answer_sync(srv, "where is my laptop")
    assert "hours ago" in res["text"]

    # Seen by the camera 40 s ago, this walk: the sighting is the answer, never the six-hour-old
    # place (drafts of 10-29: "laptop near the TV, 7 hours ago" with the laptop in the walk).
    srv.live_last = {"laptop": time.time() - 40}
    res = Server._answer_sync(srv, "where is my laptop")
    assert res["text"] == (
        "I saw your laptop a moment ago, but I don't have its place on the map yet."
    )
    assert res["pin"] is None and "hours" not in res["text"]
    srv.live_last = {"laptop": time.time() - 300}
    res = Server._answer_sync(srv, "where is my laptop")
    assert res["text"].startswith("I saw your laptop 5 minutes ago") and "hours" not in res["text"]


def test_a_sighting_a_moment_ago_still_counts_as_in_view():
    """The detector blinks: a laptop in frame 2 s before the question is in view, not memory."""
    import time
    from types import SimpleNamespace

    from metasight.server import Server

    store = PinStore()
    store.pins[1] = Pin(
        1,
        "laptop",
        "object",
        [2.0, 1.0, -3.0],
        time.time() - 7 * 3600,
        time.time() - 7 * 3600,
        3,
        0.9,
        ["TV"],
    )
    now = time.time()
    srv = SimpleNamespace(
        pins=store,
        language=SimpleNamespace(name="template"),
        pose=None,
        map_pose_ts=0.0,
        guide=None,
        live_seen=(now - 2, [("laptop", 3.0, 1.4)]),
        live_last={"laptop": now - 2},
        live_where={"laptop": (now - 2, 3.0, 1.4, None)},
    )
    res = Server._answer_sync(srv, "where is my laptop")
    assert res["live"] and "in view now" in res["text"] and "hours" not in res["text"]


def test_a_placed_sighting_answers_with_its_direction_from_here():
    """Seen 40 s ago with a fresh pose: the answer points from where the wearer is now."""
    import time
    from types import SimpleNamespace

    import numpy as np

    from metasight.heading import Heading
    from metasight.server import Server

    store = PinStore()
    store.pins[1] = Pin(
        1,
        "backpack",
        "object",
        [5.0, 0.0, 5.0],
        time.time() - 7 * 3600,
        time.time() - 7 * 3600,
        3,
        0.9,
        ["sink"],
    )
    hd = Heading(60.0)
    hd.track(np.zeros(3), np.array([0.0, 0.0, 1.0]))  # at the origin, facing +z
    seen_at = hd.ahead(2.0, 0.0)  # it was 2 m straight ahead
    assert np.allclose(hd.to(seen_at), (2.0, 0.0))
    hd.track(np.zeros(3), np.array([1.0, 0.0, 0.0]))  # turned a quarter: now off to one side
    side = "right" if hd.to(seen_at)[1] > 0 else "left"
    now = time.time()
    srv = SimpleNamespace(
        pins=store,
        language=SimpleNamespace(name="template"),
        pose=None,
        map_pose_ts=0.0,
        guide=None,
        heading=hd,
        live_seen=(now - 40, []),
        live_last={"backpack": now - 40},
        live_where={"backpack": (now - 40, 0.0, 2.0, seen_at)},
        guide_pose_fresh=lambda t: True,
    )
    res = Server._answer_sync(srv, "where is my backpack")
    assert res["text"].startswith("I saw your backpack a moment ago, about 2 metres")
    assert side in res["text"] and "sink" not in res["text"] and "hours" not in res["text"]


def test_a_weak_glimpse_this_walk_is_offered_as_a_maybe():
    """Too unsure to move the six-hour-old pin, but fresher: say it, hedged, with its place."""
    import time
    from types import SimpleNamespace

    from metasight.server import Server

    now = time.time()
    store = PinStore()
    store.observe("backpack", "object", [2.0, 0.0, -3.0], now - 6 * 3600, 0.9)
    store.observe("backpack", "object", [2.0, 0.0, -3.0], now - 6 * 3600 + 5, 0.9)
    srv = SimpleNamespace(
        pins=store,
        language=SimpleNamespace(name="template"),
        pose=None,
        map_pose_ts=0.0,
        guide=None,
        live_seen=(0.0, []),
        live_last={},
    )
    assert "6 hours ago" in Server._answer_sync(srv, "where is my backpack")["text"]

    store.observe("backpack", "object", [6.0, 0.0, 1.0], now - 60, 0.2)
    glimpse = max(store.pins.values(), key=lambda p: p.ts)
    glimpse.near = ["desk"]
    res = Server._answer_sync(srv, "where is my backpack")
    assert res["text"] == "I think I glimpsed your backpack by the desk 1 minute ago."
    assert srv.guide["pin"] == glimpse.id
    assert store.latest("backpack").score == 0.9  # the sure pin itself has not moved


def test_a_sighting_without_a_pose_is_placed_by_what_was_beside_it():
    """Replay 131247 has no poses: "where is my laptop" said "no place on the map yet" although
    the camera saw it by the TV, where the saved pin has it."""
    import time
    from types import SimpleNamespace

    from metasight.server import Server, beside_in_frame

    frame = [("laptop", 3.0, 1.4), ("person", 2.0, 2.0), ("cup", 4.0, 1.3), ("TV", -8.0, 2.5)]
    assert beside_in_frame("laptop", 3.0, frame) == "TV"  # the cup is carried, never a place
    assert beside_in_frame("laptop", 3.0, [("laptop", 3.0, 1.4), ("door", 4.0, 3.0)]) == "door"
    assert beside_in_frame("laptop", 3.0, [("laptop", 3.0, 1.4), ("cup", 4.0, 1.3)]) is None

    old = time.time() - 7 * 3600
    store = PinStore()
    store.pins[1] = Pin(1, "laptop", "object", [2.0, 1.0, -3.0], old, old, 3, 0.9, ["TV"])
    now = time.time()
    srv = SimpleNamespace(
        pins=store,
        language=SimpleNamespace(name="template"),
        pose=None,
        map_pose_ts=0.0,
        guide=None,
        live_seen=(now - 20, []),
        live_last={"laptop": now - 20},
        live_where={"laptop": (now - 20, 3.0, 1.4, None, "TV")},
        guide_pose_fresh=lambda t: False,
    )
    res = Server._answer_sync(srv, "where is my laptop")
    assert res["text"] == "I saw your laptop a moment ago, by the TV."
    assert res["pin"]["id"] == 1 and srv.guide["pin"] == 1  # the same place: fly to the pin

    srv.live_where = {"laptop": (now - 20, 3.0, 1.4, None, "bed")}  # moved: not the old place
    res = Server._answer_sync(srv, "where is my laptop")
    assert res["text"] == "I saw your laptop a moment ago, by the bed."
    assert res["pin"] is None and "hours" not in res["text"]


def test_a_weak_local_keys_pin_is_not_given_as_the_keys():
    """The saved house's keys pin (score 0.19 from the local finder, 0.6 m from the keyboard)
    is held back on purpose: personal items need SAM 3.1 or a sure local score."""
    import time
    from types import SimpleNamespace

    from metasight.server import Server

    t = time.time() - 3600
    store = PinStore(items=["keys"])
    store.pins[12] = Pin(12, "keys", "object", [0.5, 1.1, -0.3], t, t, 13, 0.19, ["sink"])
    srv = SimpleNamespace(
        pins=store,
        language=SimpleNamespace(name="template"),
        pose=None,
        map_pose_ts=0.0,
        guide=None,
        live_seen=(0.0, []),
        live_last={},
    )
    res = Server._answer_sync(srv, "where are my keys")
    assert res["pin"] is None and res.get("skipped_pin_id") == 12
