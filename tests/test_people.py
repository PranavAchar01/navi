import math

import numpy as np

from metasight.people import (
    SPEAK_MIN,
    Conversation,
    Det,
    PeopleTracker,
    bearing_cue,
    head_patch,
    lip_activity,
    pick_speaker,
)


def box(cx, w=0.15, h=0.5, top=0.2):
    return (cx - w / 2, top, cx + w / 2, top + h)


def run(paths, dist, n=60, fps=15.0, seed=0, miss=0.0):
    """Tracks for people walking along `paths(t)`; returns {true index: set of track ids}."""
    rng = np.random.default_rng(seed)
    tk, seen = PeopleTracker(), {}
    for k in range(n):
        t = k / fps
        dets = []
        for i, p in enumerate(paths):
            if rng.random() < miss:
                continue
            cx = p(t) + rng.normal(0, 0.003)
            dets.append((i, Det(box(cx), 0.9, dist[i])))
        for tr in tk.update([d for _, d in dets], t):
            if not dets:
                continue
            j = max(dets, key=lambda x: _iou_c(tr.box, x[1].box))[0]
            seen.setdefault(j, set()).add(tr.id)
    return seen


def _iou_c(a, b):
    from metasight.people import iou

    return iou(a, b)


def test_two_people_cross_at_different_depths_keep_their_ids():
    paths = [lambda t: 0.2 + 0.4 * t, lambda t: 0.8 - 0.4 * t]
    seen = run(paths, [1.5, 4.0])
    assert all(len(v) == 1 for v in seen.values()) and len(seen) == 2


def test_track_survives_missed_detections():
    seen = run([lambda t: 0.3 + 0.1 * t], [2.0], n=90, miss=0.3, seed=2)
    assert len(seen[0]) == 1


def test_weak_box_keeps_track_alive():
    tk = PeopleTracker()
    for k in range(4):
        tk.update([Det(box(0.5), 0.9, 2.0)], k / 15)
    out = tk.update([Det(box(0.5), 0.3, 2.0)], 4 / 15)
    assert [t.id for t in out] == [1]


def test_forgotten_after_leaving():
    tk = PeopleTracker()
    for k in range(4):
        tk.update([Det(box(0.5), 0.9, 2.0)], k / 15)
    tk.update([], 5.0)
    assert tk.tracks == []


def test_no_identity_state_is_kept():
    tk = PeopleTracker()
    for k in range(4):
        tk.update([Det(box(0.5), 0.9, 2.0)], k / 15)
    assert not any(hasattr(t, a) for t in tk.tracks for a in ("embedding", "name", "face"))


def test_bearing_cue_uses_belt_pan():
    c = bearing_cue(30.0)
    assert c["turn"] == "right" and c["motors"][0][0] == 0 and c["motors"][1][0] == 1
    assert abs(c["motors"][0][1] - 0.5) < 0.02 and abs(c["motors"][1][1] - 0.87) < 0.02
    assert bearing_cue(5.0) is None
    assert bearing_cue(-40.0)["turn"] == "left"


def test_nobody_speaks_without_voice():
    s, ev = pick_speaker(False, [(1, 0.0, 9.0, 2.0)], 0.0)
    assert s == {1: 0.0} and ev == "none"


def test_doa_picks_the_person_in_that_direction():
    cands = [(1, -30.0, None, 2.0), (2, 25.0, None, 2.0)]
    s, ev = pick_speaker(True, cands, 22.0)
    assert s[2] > SPEAK_MIN > s[1] and ev == "doa"


def test_lip_and_doa_agree_beats_doa_alone():
    cands = [(1, 20.0, 6.0, 2.0), (2, 25.0, 0.2, 2.0)]
    s, _ = pick_speaker(True, cands, 22.0)
    assert s[1] > s[2]


def test_alone_with_voice_is_called_but_flagged():
    s, ev = pick_speaker(True, [(1, 0.0, None, 2.0)], None)
    assert s[1] >= SPEAK_MIN and ev == "alone"


def test_two_people_voice_no_evidence_calls_nobody():
    s, ev = pick_speaker(True, [(1, 0.0, None, 2.0), (2, 20.0, None, 2.0)], None)
    assert max(s.values()) < SPEAK_MIN


def test_lip_activity_needs_mouth_half_to_move():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(12, 24)).astype(np.float32)
    b = rng.normal(size=(12, 24)).astype(np.float32)
    still = (a, b)
    mouth = (a + rng.normal(size=a.shape).astype(np.float32) * 3, b)
    shake = (
        a + 3 * rng.normal(size=a.shape).astype(np.float32),
        b + 3 * rng.normal(size=b.shape).astype(np.float32),
    )
    assert lip_activity(still, still) == 0
    assert lip_activity(still, mouth) > 2 > lip_activity(still, shake) - 1


def test_head_patch_too_small_is_unreadable():
    g = np.zeros((480, 640), np.uint8)
    assert head_patch(g, (0.5, 0.2, 0.52, 0.4))[0] is None
    assert head_patch(g, (0.3, 0.1, 0.6, 0.9))[0] is not None


def test_conversation_cues_toward_offscreen_voice_and_marks_map_position():
    cv = Conversation(hfov_deg=90.0)
    g = np.zeros((240, 320), np.uint8)
    ev = None
    for k in range(5):
        ev = cv.step(k / 15, [Det(box(0.5), 0.9, 2.0)], g, True, 60.0, (1.0, 2.0, 0.0))
    # nobody in view near 60 degrees: the voice is off screen and the cue turns right
    assert ev["offscreen"] == 60.0
    cues = [
        e
        for e in (cv.step(2.0 + k / 15, [Det(box(0.5), 0.9, 2.0)], g, True, 60.0) for k in range(3))
        if e["cue"]
    ]
    assert cues and cues[0]["cue"]["turn"] == "right"
    t = ev["tracks"][0]
    assert math.isclose(t["map"][0], 1.0, abs_tol=0.05) and math.isclose(
        t["map"][1], 4.0, abs_tol=0.05
    )


def test_mono_microphone_falls_back_to_the_mouth_that_moves():
    """Glasses audio is mono: no DOA, so whoever's mouth moves when the VAD fires is the speaker
    and the wearer is cued toward them."""
    rng = np.random.default_rng(3)
    base = rng.integers(60, 200, size=(480, 640)).astype(np.uint8)
    conv = Conversation(hfov_deg=90.0)
    left, right = (0.1, 0.1, 0.3, 0.9), (0.7, 0.1, 0.9, 0.9)
    cue = None
    for k in range(40):
        g = base.copy()
        g[86:125, 474:550] = rng.integers(0, 255, size=(39, 76))  # right person's mouth moves
        ev = conv.step(
            k / 15, [Det(left, 0.9, 2.0), Det(right, 0.9, 2.0)], g, voice=True, doa_deg=None
        )
        cue = cue or ev["cue"]
    assert ev["doa"] is None and ev["evidence"] == "lip"
    right_id = max(ev["tracks"], key=lambda t: t["bearing"])["id"]
    assert ev["speaker"] == right_id
    assert cue and cue["turn"] == "right"


def test_mono_with_two_still_faces_calls_nobody_and_does_not_cue():
    rng = np.random.default_rng(4)
    g = rng.integers(60, 200, size=(480, 640)).astype(np.uint8)
    conv = Conversation(hfov_deg=90.0)
    for k in range(40):
        ev = conv.step(
            k / 15,
            [Det((0.1, 0.1, 0.3, 0.9), 0.9, 2.0), Det((0.7, 0.1, 0.9, 0.9), 0.9, 2.0)],
            g,
            voice=True,
            doa_deg=None,
        )
        assert ev["cue"] is None
    assert ev["speaker"] is None


def test_voice_from_the_side_outranks_a_mouth_moving_ahead():
    """Someone ahead moves their mouth while the heard voice is 40 degrees right: turn right."""
    rng = np.random.default_rng(5)
    base = rng.integers(60, 200, size=(480, 640)).astype(np.uint8)
    conv = Conversation(hfov_deg=90.0)
    ahead = (0.4, 0.1, 0.6, 0.9)
    cues = []
    for k in range(40):
        g = base.copy()
        g[86:125, 282:358] = rng.integers(0, 255, size=(39, 76))
        ev = conv.step(k / 15, [Det(ahead, 0.9, 2.0)], g, voice=True, doa_deg=40.0)
        cues += [ev["cue"]] if ev["cue"] and k >= 4 else []
    assert cues and cues[0]["turn"] == "right"


def test_voice_well_off_to_the_side_cues_even_when_the_moving_mouth_is_nearly_ahead():
    """A face about 9 degrees right moves its mouth; the voice is at 39. Facing it gives no cue,
    so the heard direction has to be the one that turns the wearer."""
    rng = np.random.default_rng(8)
    base = rng.integers(60, 200, size=(480, 640)).astype(np.uint8)
    conv = Conversation(hfov_deg=90.0)
    near = (0.5, 0.1, 0.7, 0.9)
    cues = []
    for k in range(40):
        g = base.copy()
        g[86:125, 350:426] = rng.integers(0, 255, size=(39, 76))
        ev = conv.step(k / 15, [Det(near, 0.9, 2.0)], g, voice=True, doa_deg=39.0)
        cues += [ev["cue"]] if ev["cue"] and k >= 4 else []
    assert cues and cues[0]["turn"] == "right"


def test_steady_voice_near_a_face_that_is_not_read_still_cues():
    """Someone stands where the voice is but the mouth gives nothing: the cue still turns."""
    rng = np.random.default_rng(6)
    g = rng.integers(60, 200, size=(480, 640)).astype(np.uint8)
    conv = Conversation(hfov_deg=90.0)
    right = (0.58, 0.1, 0.78, 0.9)  # about 20 degrees right, the voice is at 40
    cues = []
    for k in range(40):
        ev = conv.step(k / 15, [Det(right, 0.9, 2.0)], g, voice=True, doa_deg=40.0)
        cues += [ev["cue"]] if ev["cue"] and k >= 4 else []
    assert cues and all(c["turn"] == "right" for c in cues)


def test_cue_toward_a_new_direction_does_not_wait_for_the_repeat_gap():
    rng = np.random.default_rng(7)
    g = rng.integers(60, 200, size=(480, 640)).astype(np.uint8)
    conv = Conversation(hfov_deg=90.0)
    left, right = [], []
    for k in range(6):
        e = conv.step(k / 15, [], g, voice=True, doa_deg=-50.0)
        left += [e["cue"]] if e["cue"] else []
    for k in range(6, 12):
        e = conv.step(k / 15, [], g, voice=True, doa_deg=40.0)
        right += [e["cue"]] if e["cue"] else []
    assert left and left[0]["turn"] == "left"
    assert right and right[0]["turn"] == "right"  # well inside 1.5 s of the left cue


def test_the_people_list_is_bearings_only():
    cv = Conversation(hfov_deg=90.0)
    g = np.zeros((240, 320), np.uint8)
    ev = None
    for k in range(4):
        ev = cv.step(k / 15, [Det(box(0.2), 0.9, 2.0), Det(box(0.8), 0.9, 3.0)], g, False, None)
    assert len(ev["people"]) == 2
    assert all(set(p) == {"bearing_deg"} for p in ev["people"])
    assert sorted(p["bearing_deg"] for p in ev["people"]) == sorted(
        t["bearing"] for t in ev["tracks"]
    )
