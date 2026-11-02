"""The simulated listener and the take renderer (SIMULATED LISTENER MODEL, NOT A HUMAN STUDY)."""

from __future__ import annotations

import numpy as np
import pytest

from metasight import spatial_listener as sl
from metasight import spatial_sound as ss


def at(az: float) -> ss.Direction:
    return ss.Direction(az, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("method", ["hrtf", "pan"])
def test_the_listener_hears_left_and_right(method: str) -> None:
    pluck = ss.beacon_sound(False)
    assert sl.hear(ss.place(pluck, at(60), method)).itd_s > 1e-4
    assert sl.hear(ss.place(pluck, at(-60), method)).itd_s < -1e-4
    assert abs(sl.hear(ss.place(pluck, at(0), method)).itd_s) < 2e-5


@pytest.mark.parametrize("method", ["hrtf", "pan"])
@pytest.mark.parametrize("az", [-150, -60, 30, 120])
def test_every_cue_lands_near_its_direction_after_a_head_turn(method: str, az: float) -> None:
    rng = np.random.default_rng(0)
    for mono in (ss.beacon_sound(False), *(ss.tick_sound(k) for k in ss.TICK_KINDS)):
        r = sl.localise(mono, az, method, rng, reps=5)
        assert r["az_err"] < 20 and r["wrong_side"] == 0


def test_a_pure_stop_tone_is_hard_to_place_and_the_octave_fixes_it() -> None:
    hz, dur, gain = ss.TICK_KINDS["stop"]
    t = np.arange(int(dur * ss.SAMPLE_RATE)) / ss.SAMPLE_RATE
    pure = gain * np.sin(2 * np.pi * hz * t) * ss._envelope(len(t))
    rng = np.random.default_rng(0)
    pure_err = np.mean([sl.localise(pure, az, "hrtf", rng, 5)["az_err"] for az in (60, 90, 120)])
    beep_err = np.mean(
        [sl.localise(ss.tick_sound("stop"), az, "hrtf", rng, 5)["az_err"] for az in (60, 90, 120)]
    )
    assert pure_err > 30 > 10 > beep_err


def test_beeps_carry_the_octave_below_and_keep_their_peak() -> None:
    for kind, (hz, _dur, gain) in ss.TICK_KINDS.items():
        x = ss.tick_sound(kind)
        spec = np.abs(np.fft.rfft(x, 1 << 16))
        f = np.fft.rfftfreq(1 << 16, 1 / ss.SAMPLE_RATE)
        sub = spec[np.argmin(abs(f - hz / 2))]
        assert 0.3 < sub / spec[np.argmin(abs(f - hz))] < 0.7, kind
        assert np.abs(x).max() <= gain + 1e-6


def safety(t: float, **kw) -> dict:
    m = {
        "wall": 1000.0 + t,
        "type": "safety",
        "sectors": [None] * 8,
        "cues": [{"mode": "off"}] * 8,
        "path": None,
        "guide": None,
        "found": False,
    }
    m.update(kw)
    return m


def test_a_take_plays_the_guide_beacon_from_its_bearing_then_arrives() -> None:
    g = {"label": "chair", "bearing": 90.0, "dist": 2.0}
    msgs = [safety(i * 0.1, guide=g) for i in range(50)]
    msgs += [safety(5 + i * 0.1, guide=g, found=True) for i in range(10)]
    ev = ss.schedule_take(msgs)
    beacons = [e for e in ev if e.kind == "beacon"]
    assert beacons and all(e.where.az_deg == 90 for e in beacons)
    gaps = np.diff([e.t for e in beacons])
    assert np.allclose(gaps, ss.beacon_period(2.0), atol=ss.STEP_S + 1e-9)
    assert [e.kind for e in ev].count("arrive") == 1


def test_a_take_beeps_the_most_urgent_sector_and_falls_silent_when_stale() -> None:
    cues = [{"mode": "solid"}] + [{"mode": "off"}] * 6 + [{"mode": "fast"}]
    sectors = [None] * 7 + [0.7]
    msgs = [safety(i * 0.1, cues=cues, sectors=sectors) for i in range(20)]
    msgs.append(safety(10.0))  # after an 8 s gap: the stale stretch plays nothing
    ev = ss.schedule_take(msgs)
    stops = [e for e in ev if e.kind == "stop"]
    assert stops and all(e.where.az_deg == 0 for e in stops)
    assert not [e for e in ev if e.kind == "fast"]
    assert all(e.t < 3.6 for e in ev)
    assert ss.take_warning(safety(0, cues=[{"mode": "unknown"}] * 8, sectors=sectors)) is None
    w = ss.take_warning(safety(0, cues=[{"mode": "off"}] * 7 + [{"mode": "fast"}], sectors=sectors))
    assert w == (7, 0.7, "fast")


def test_the_walking_path_counts_for_the_front_sector() -> None:
    m = safety(0, path=0.4)
    assert ss.take_warning(m) == (0, 0.4, "stop")
    x = ss.render_take([safety(i * 0.1, path=0.4) for i in range(30)])
    assert x.shape[1] == 2 and np.abs(x).max() > 0


def test_a_held_warning_settles_and_a_flicker_does_not_restart_it() -> None:
    off = [{"mode": "off"}] * 8
    near, mid = [0.3] + [None] * 7, [0.7] + [None] * 7
    flicker = [near if (i // 8) % 2 == 0 else mid for i in range(250)]  # 10 s at a desk
    msgs = [safety(i * 0.04, cues=off, sectors=d) for i, d in enumerate(flicker)]
    t = [e.t for e in ss.schedule_take(msgs)]
    assert len([x for x in t if x < 3]) >= 8
    assert len([x for x in t if x >= 3.2]) <= 7
    # The head turns: the warning hops to the right and keeps the settled pace from there.
    side = [None, None, 0.3] + [None] * 5
    msgs.append(safety(10.04, cues=off, sectors=side))
    last = ss.schedule_take(msgs)[-1]
    assert last.t < 10.0
    msgs += [safety(10.04 + i * 0.04, cues=off, sectors=side) for i in range(1, 25)]
    hop = ss.schedule_take(msgs)[-1]
    assert hop.where.az_deg == 90 and 10.7 < hop.t < 10.9
    # All clear, then a new warning: it sounds at once.
    msgs.append(safety(11.1, cues=off))
    msgs.append(safety(11.2, cues=off, sectors=side))
    assert ss.schedule_take(msgs)[-1].t == pytest.approx(11.2, abs=0.05)


def test_the_strike_tells_front_from_back_and_the_side_has_no_mirror(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    azs = (-150, -30, 30, 150)
    with_strike = [sl.front_back(ss.tick_sound("fast"), az, "hrtf", rng, 10) for az in azs]
    assert max(with_strike) <= 0.1
    monkeypatch.setattr(ss, "AIR", 0.0)
    bare = [sl.front_back(ss.tick_sound("fast"), az, "hrtf", rng, 10) for az in azs]
    assert np.mean(bare) >= 0.3  # a bare tone is close to a coin toss
    assert sl.front_back(ss.tick_sound("fast"), 90, "hrtf", rng) is None
    assert sl.mirror(30) == 150 and sl.mirror(-170) == -10
