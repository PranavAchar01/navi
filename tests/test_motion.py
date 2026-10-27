import numpy as np

from metasight.motion import FallDetector, G, Sample, ShakeDetector


def feed(det, samples):
    events = []
    for s in samples:
        events += det.update(s)
    return events


def standing(t0, t1, hz=100, down=(0, 0, 1)):
    return [Sample(t, *(np.array(down) * G)) for t in np.arange(t0, t1, 1 / hz)]


def fall_trace(rotate=True, hz=100):
    pre = standing(0, 3, hz)
    impact = [Sample(3.0 + i / hz, 0.0, 0.0, 3.6 * G) for i in range(3)]
    down = (1, 0, 0) if rotate else (0, 0, 1)
    rng = np.random.default_rng(0)
    after = [
        Sample(t, *(np.array(down) * G + rng.normal(0, 0.02 * G, 3)))
        for t in np.arange(3.05, 8, 1 / hz)
    ]
    return pre + impact + after


def test_fall_asks_after_impact_tilt_and_stillness():
    ev = feed(FallDetector(), fall_trace())
    assert [e.kind for e in ev] == ["fall_prompt"]
    assert ev[0].message == "Are you okay? Do you need me to call an ambulance?"


def test_impact_without_a_lasting_orientation_change_is_not_a_fall():
    assert feed(FallDetector(), fall_trace(rotate=False)) == []


def test_walking_is_not_a_fall():
    t = np.arange(0, 20, 0.01)
    walk = [Sample(x, 0.0, 0.5 * G * np.sin(2 * np.pi * 1.9 * x), G) for x in t]
    assert feed(FallDetector(), walk) == []


def test_spoken_okay_cancels_the_check_in():
    det = FallDetector()
    feed(det, fall_trace())
    assert det.state == "asking"
    ev = det.respond_ok(9.0)
    assert [e.kind for e in ev] == ["fall_cancelled"]
    assert det.state == "idle"


def test_no_answer_only_says_it_would_call_and_never_dials():
    det = FallDetector(answer_s=30)
    ev = feed(det, fall_trace() + standing(8, 45))
    assert [e.kind for e in ev] == ["fall_prompt", "fall_escalate"]
    assert "would call" in ev[1].message


def test_test_button_runs_the_same_flow():
    det = FallDetector()
    assert [e.kind for e in det.trigger_test(0.0)] == ["fall_prompt"]
    assert [e.kind for e in det.respond_ok(2.0)] == ["fall_cancelled"]


def sine(freq, amp_g, seconds, hz=60):
    return [
        Sample(t, amp_g * G * np.sin(2 * np.pi * freq * t), 0.0, G)
        for t in np.arange(0, seconds, 1 / hz)
    ]


def test_sustained_shaking_gives_an_observational_notice():
    ev = feed(ShakeDetector(), sine(5.0, 1.0, 12))
    assert [e.kind for e in ev] == ["shake_notice"]
    assert ev[0].message == "I noticed sustained shaking. Are you okay?"
    assert "seiz" not in ev[0].message.lower()


def test_a_short_burst_does_not_trigger():
    assert feed(ShakeDetector(), sine(5.0, 1.0, 3.5) + sine(0.2, 0.01, 1)) == []


def test_walking_cadence_does_not_trigger():
    assert feed(ShakeDetector(), sine(1.9, 0.5, 20)) == []


def test_quiet_phone_does_not_trigger():
    assert feed(ShakeDetector(), standing(0, 20, hz=60)) == []


def test_shake_notice_has_a_cooldown():
    ev = feed(ShakeDetector(), sine(5.0, 1.0, 40))
    assert len(ev) == 1
