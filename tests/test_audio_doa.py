import math

import numpy as np

from metasight.audio_doa import SOUND_MPS, SR, Doa, Vad, gcc_phat


def two_mics(deg, spacing=0.14, n=SR // 4, snr_db=20, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.normal(size=n + 200).astype(np.float32)
    lag = spacing / SOUND_MPS * math.sin(math.radians(deg)) * SR  # samples left hears it later
    k = np.arange(n)
    left = np.interp(k - lag / 2 + 100, np.arange(len(src)), src)
    right = np.interp(k + lag / 2 + 100, np.arange(len(src)), src)
    x = np.stack([left, right], 1)
    x += rng.normal(size=x.shape) * 10 ** (-snr_db / 20) * x.std()
    return x.astype(np.float32)


def test_gcc_phat_sign_and_size():
    x = two_mics(45)
    lag, peak = gcc_phat(x[:, 0], x[:, 1], 0.14 / SOUND_MPS)
    assert lag > 0 and peak > 0.3
    assert abs(math.degrees(math.asin(lag * SOUND_MPS / 0.14)) - 45) < 4


def test_doa_bearings_within_five_degrees():
    for deg in (-60, -30, 0, 30, 60):
        d = Doa(0.14)
        for s in range(6):
            d.push(two_mics(deg, seed=s), True)
        assert abs(d.bearing() - deg) < 5, deg


def test_doa_ignores_unvoiced_and_mono():
    d = Doa(0.14)
    assert d.push(two_mics(30), False) is None
    assert d.push(two_mics(30)[:, :1], True) is None
    assert d.bearing() is None


def test_doa_refuses_noise_without_a_clear_peak():
    d = Doa(0.14)
    rng = np.random.default_rng(1)
    for _ in range(8):
        d.push(rng.normal(size=(4000, 2)).astype(np.float32), True)
    assert d.bearing() is None or len(d.votes) < 8


def test_energy_vad_gates_a_burst():
    v = Vad(model=False)
    rng = np.random.default_rng(0)
    quiet = (rng.normal(size=SR) * 0.003).astype(np.float32)
    loud = (rng.normal(size=SR // 2) * 0.1).astype(np.float32)
    v.push(quiet)
    assert not v.active()
    v.push(loud)
    assert v.active()


def test_decorrelated_channels_give_no_direction_not_a_wrong_one():
    """Two channels that share no signal (a codec that smeared the lag, a mono mic copied twice
    with independent noise) must yield None, so the cue logic falls back to the camera."""
    rng = np.random.default_rng(5)
    d = Doa(0.14)
    for _ in range(40):
        d.push(rng.normal(size=(512, 2)) * 0.2, True)
    assert d.bearing() is None


def test_noise_only_blocks_rarely_give_a_bearing():
    """Each false bearing is a belt cue pointing at nobody: 512-sample blocks of independent noise,
    all flagged as voiced (the VAD's worst mistake), seeds the gate was not tuned on."""
    hits = 0
    for seed in range(200, 205):
        rng = np.random.default_rng(seed)
        for _ in range(60):
            d = Doa(0.14)
            for _ in range(int(rng.integers(8, 41))):
                d.push(rng.normal(size=(512, 2)) * 0.2, True)
                if d.bearing() is not None:
                    hits += 1
                    break
    assert hits <= 3  # of 300 runs
