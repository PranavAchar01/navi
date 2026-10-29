import numpy as np
import pytest

from metasight import belt_audio as ba
from metasight.belt import MODES, NAMES
from metasight.config import ROOT

OFF = (0, 0)


def only(i, mode="solid", inten=255):
    cmd = [OFF] * 8
    cmd[i] = (MODES[mode], inten)
    return cmd


def test_pan_follows_the_angle():
    assert ba.pan(0) == pytest.approx(0) and ba.pan(4) == pytest.approx(0)
    assert ba.pan(2) == pytest.approx(1) and ba.pan(6) == pytest.approx(-1)
    assert ba.pan(1) > 0 > ba.pan(7)
    left, right = ba.stereo_gains(6)
    assert left > 0.99 and right < 0.01
    for i in range(8):
        lg, rg = ba.stereo_gains(i)
        assert lg * lg + rg * rg == pytest.approx(1)  # equal power


def test_front_is_higher_than_back_and_sides_in_between():
    f = [ba.frequency(i) for i in range(8)]
    assert f[0] > f[1] > f[2] > f[3] > f[4]
    assert f[2] == pytest.approx(f[6]) == pytest.approx(ba.BASE_HZ)
    assert f[0] / f[4] == pytest.approx(2.0)
    assert ba.frequency(2, MODES["guide"]) == pytest.approx(ba.BASE_HZ * 1.5)


def test_amplitude_follows_duty():
    assert ba.amplitude(0) == 0 and ba.amplitude(255) == pytest.approx(ba.VOICE_GAIN)
    assert ba.amplitude(128) < ba.amplitude(255)


def test_duty_mirrors_the_belt_rhythms():
    # belt_core.h: slow 200 ms on of 667, fast 125 on of 250, solid always, guide two taps a second
    assert ba.duty_at(MODES["slow"], 200, 0.1) == 200 and ba.duty_at(MODES["slow"], 200, 0.3) == 0
    assert ba.duty_at(MODES["fast"], 200, 0.1) == 200 and ba.duty_at(MODES["fast"], 200, 0.2) == 0
    assert ba.duty_at(MODES["solid"], 200, 0.9) == 200
    assert [ba.duty_at(MODES["guide"], 9, t) for t in (0.03, 0.1, 0.18, 0.5)] == [9, 0, 9, 0]
    assert ba.duty_at(MODES["unknown"], 255, 0.0) == 0


def level(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def peak_hz(x):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / ba.SAMPLE_RATE)[int(np.argmax(spec))]


def test_render_places_sound_on_the_correct_side():
    for motor, louder in [(2, 1), (6, 0), (7, 0), (1, 1)]:
        s = ba.render([(0.0, only(motor))], 0.5)
        assert level(s[:, louder]) > 2 * level(s[:, 1 - louder])
    mid = ba.render([(0.0, only(0))], 0.5)
    assert level(mid[:, 0]) == pytest.approx(level(mid[:, 1]), rel=0.01)


def test_render_pitch_separates_front_from_back():
    front = ba.render([(0.0, only(0))], 0.5)[:, 0]
    back = ba.render([(0.0, only(4))], 0.5)[:, 0]
    assert abs(peak_hz(front) - ba.frequency(0)) < 15
    assert abs(peak_hz(back) - ba.frequency(4)) < 15


def test_silent_motors_make_no_sound_and_intensity_sets_loudness():
    assert level(ba.render([(0.0, [OFF] * 8)], 0.2)) == 0
    soft = level(ba.render([(0.0, only(0, inten=60))], 0.3))
    loud = level(ba.render([(0.0, only(0, inten=250))], 0.3))
    assert 0 < soft < loud


def test_braille_plays_as_audible_pulses_one_motor_at_a_time():
    frames, total = ba.braille_frames("b")  # dots 1 and 2
    hot = [[i for i, (m, _) in enumerate(f) if m] for _, f in frames]
    assert {tuple(h) for h in hot if h} == {(0,), (7,)}
    s = ba.render(frames, total)
    assert level(s) > 0
    # pulse 1 (front motor) is centred; pulse 2 (front-left, motor 7) lands on the left
    first = s[int(0.02 * ba.SAMPLE_RATE) : int(0.08 * ba.SAMPLE_RATE)]
    second = s[int(0.18 * ba.SAMPLE_RATE) : int(0.24 * ba.SAMPLE_RATE)]
    assert level(first[:, 0]) == pytest.approx(level(first[:, 1]), rel=0.05)
    assert level(second[:, 0]) > level(second[:, 1])


def test_wav_roundtrip(tmp_path):
    import wave

    out = tmp_path / "t.wav"
    ba.write_wav(out, ba.render([(0.0, only(2))], 0.25))
    with wave.open(str(out)) as w:
        assert w.getnchannels() == 2 and w.getframerate() == ba.SAMPLE_RATE
        assert w.getnframes() == int(0.25 * ba.SAMPLE_RATE)
    assert (ROOT / "scripts" / "render_belt_audio.py").exists()
    assert NAMES[MODES["guide"]] == "guide"
