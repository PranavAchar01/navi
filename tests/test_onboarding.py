from metasight.belt import MODES
from metasight.onboarding import (
    PATTERNS,
    WALK_DONE_M2,
    WALK_LINE,
    Onboarding,
    item_label,
    pattern_cmd,
    pattern_length,
)


def walk(ob, n, step=0.3):
    out = []
    for i in range(n):
        out += ob.trail([[i * step, 0.0, 0.0, float(i)]])
    return out


def test_naming_phrases_become_watch_words():
    assert item_label("these are my keys") == "keys"
    assert item_label("This is my car keys.") == "keys"  # alias, one canonical word
    assert item_label("that's my water bottle") == "water bottle"
    assert item_label("remember my passport") == "passport"
    assert item_label("where is my wallet") is None
    assert item_label("my keys are great but whatever") is None


def test_start_says_the_welcome_and_walks():
    ob = Onboarding()
    ev = ob.start()
    assert ev["step"] == "walk" and ev["say"] == WALK_LINE and ev["progress"] == 0.0
    assert ob.active


def test_standing_still_makes_no_progress_and_no_lines():
    ob = Onboarding()
    ob.start()
    assert ob.trail([[0.1, 0, 0.1, 1.0], [0.12, 0, 0.1, 2.0]]) != []  # first cell
    assert ob.trail([[0.1, 0, 0.1, 3.0]]) == []
    assert ob.area_m2 == 0.25


def test_walking_speaks_milestones_once_then_moves_to_naming():
    ob = Onboarding()
    ob.start()
    evs = []
    for row in range(40):  # sweep rows of floor, one cell wide
        evs += ob.trail([[col * 0.5, 0.0, row * 0.5, 0.0] for col in range(12)])
        if ob.step == "name":
            break
    said = [e["say"] for e in evs if "say" in e]
    assert said[0] == "Good. Keep going." and said.count("Good. Keep going.") == 1
    assert ob.step == "name" and ob.area_m2 >= WALK_DONE_M2
    assert said[-2:] == ["That's a good first look at your home.", said[-1]]
    assert "keys" in said[-1]
    assert evs[-1]["progress"] == 1.0


def test_naming_items_adds_them_and_done_moves_on():
    ob = Onboarding()
    ob.start()
    ob.step = "name"
    handled, evs, label = ob.hear("these are my keys")
    assert handled and label == "keys" and evs[0]["say"] == "Got it, keys."
    assert ob.hear("that's all")[0] and ob.step == "belt"


def test_done_without_items_stays_in_naming():
    ob = Onboarding()
    ob.start()
    ob.step = "name"
    assert ob.hear("that's all") == (False, [], None)
    assert ob.step == "name"


def test_spoken_done_ends_the_walk_early():
    ob = Onboarding()
    ob.start()
    handled, evs, _ = ob.hear("okay I'm done walking")
    assert handled and ob.step == "name"


def test_unrelated_speech_is_not_handled():
    ob = Onboarding()
    ob.start()
    assert ob.hear("where is my wallet") == (False, [], None)


def test_belt_intro_goes_through_every_mode_and_ends_quietly():
    ob = Onboarding()
    ev = ob.belt_events()
    assert [m for _, m in ev] == ["", "guide", "find", "talk", "explore"]
    assert all("say" in e for e, _ in ev)
    done = ob.finish()
    assert done["step"] == "done" and not ob.active


def test_patterns_use_the_belt_and_are_off_outside_their_segments():
    for mode, segs in PATTERNS.items():
        assert pattern_length(segs) < 3
        on = pattern_cmd(segs, segs[0][0] + 0.01)
        assert any(m != MODES["off"] for m, _ in on), mode
        assert pattern_cmd(segs, 10.0) == [(MODES["off"], 0)] * 8
    ob = Onboarding()
    assert ob.demo_cmd(5.0) is None
    ob.demo_mode, ob.demo_t0 = "guide", 5.0
    assert any(m != MODES["off"] for m, _ in ob.demo_cmd(5.05))


def test_skip_finishes():
    ob = Onboarding()
    ob.start()
    assert ob.skip()["step"] == "done" and not ob.active
