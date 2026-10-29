import numpy as np

from metasight.belt import FLAG_DROP, MODES
from metasight.conversation import Talk
from metasight.modes import (
    EXPLORE,
    FIND,
    GUIDE,
    LINES,
    TALK,
    Activity,
    command,
    implied,
)


def test_spoken_commands():
    assert command("find mode") == FIND
    assert command("Switch to talk mode please") == TALK
    assert command("go to explore") == EXPLORE
    assert command("guide mode on") == GUIDE
    assert command("where is my wallet") is None
    assert command("they will find the mode of the data") is None


def test_questions_imply_a_mode():
    assert implied("who is talking right now", False) == TALK
    assert implied("what is around me", False) == EXPLORE
    assert implied("where are my keys", True) == FIND
    assert implied("what time is it", False) is None


def test_switch_says_a_line_once_and_ignores_no_change():
    a = Activity()
    assert a.mode == GUIDE and a.set(GUIDE, "voice") is None
    ev = a.set(TALK, "voice")
    assert ev["say"] == LINES[TALK] and ev["prev"] == GUIDE and ev["speaker_cues"] is True
    assert a.set(TALK, "voice") is None
    assert "say" not in a.set(FIND, "question", speak=False)
    assert a.set("nonsense", "app") is None and a.mode == FIND


def test_belt_flag_bits_leave_drop_and_found_alone():
    a = Activity()
    seen = []
    for m in (GUIDE, FIND, TALK, EXPLORE):
        a.set(m, "app")
        bits = a.flag_bits()
        assert bits & 3 == 0
        seen.append(bits >> 2)
    assert seen == [0, 1, 2, 3]


def test_talk_is_calmer_but_never_mutes_urgent_cues():
    a = Activity()
    cmd = [(MODES["slow"], 200), (MODES["solid"], 255), (MODES["off"], 0)]
    assert a.calm(cmd) == cmd  # guide: unchanged
    a.set(TALK, "app")
    out = a.calm(cmd)
    assert out[0] == (MODES["slow"], 120) and out[1] == cmd[1] and out[2] == cmd[2]
    assert a.calm(cmd, FLAG_DROP) == cmd


def test_speaker_cues_only_flow_when_the_mode_allows():
    from tests.test_conversation import heading

    t = Talk(90.0)
    cue = {"bearing": 40.0, "turn": "right", "motors": [[1, 0.9]], "id": 1}
    t.conv.step = lambda *a, **k: {"cue": dict(cue), "tracks": []}
    bgr = np.zeros((240, 320, 3), np.uint8)
    off = [(MODES["off"], 0)] * 8
    t.cues_enabled = False
    assert t.frame(1.0, [], bgr, heading())["cue"] is None
    assert t.overlay(off, 1.1) == off
    t.cues_enabled = True
    assert t.frame(2.0, [], bgr, heading())["cue"]["turn"] == "right"
    assert t.overlay(off, 2.1)[1][0] == MODES["slow"]
