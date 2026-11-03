"""Gyro calibration recovers the clock offset, axes and bias from the map's own turns."""

import math

import numpy as np
import pytest

from metasight.gyro import GyroTrack, _exp, angle_deg, calibrate, synth

T = np.arange(0, 120, 1 / 30)
CAM_FROM_GYRO = np.diag([1.0, -1, -1]) @ _exp([0.05, 0.03, -0.04])


def _head_turns() -> np.ndarray:
    """Two minutes of a head looking around: wide yaw, some pitch, little roll, not periodic."""
    rng = np.random.default_rng(7)

    def smooth(sd: float, k: int) -> np.ndarray:
        x = np.convolve(np.cumsum(rng.normal(0, 1, len(T))), np.ones(k) / k, "same")
        return sd * (x - x.mean()) / x.std()

    yaw, pitch, roll = smooth(1.5, 15), smooth(0.3, 30), smooth(0.1, 30)
    return np.array(
        [
            _exp([0, y, 0]) @ _exp([p, 0, 0]) @ _exp([0, 0, r])
            for y, p, r in zip(yaw, pitch, roll, strict=True)
        ]
    )


def _pairs(rots: np.ndarray, noise_deg: float) -> list:
    """Keyframes at 1.5 Hz in windows of 8, as the map pairs them, with 1 degree of map noise."""
    rng = np.random.default_rng(1)
    idx = [int(np.argmin(abs(T - t))) for t in np.arange(1, 118, 1 / 1.5)]
    out = []
    for w in range(0, len(idx) - 8, 6):
        ix = idx[w : w + 8]
        for a, b in list(zip(ix[:-1], ix[1:], strict=False)) + [(ix[0], k) for k in ix[2:]]:
            noise = _exp(rng.normal(0, math.radians(noise_deg), 3))
            out.append((T[a], T[b], rots[a].T @ rots[b] @ noise))
    return out


@pytest.mark.parametrize("clock_known", [False, True])
def test_calibrate_recovers_offset_axes_and_turns(clock_known):
    rots = _head_turns()
    offset = 0.137 if clock_known else 5000.137
    t, w = synth(T, rots, bias_dps=0.8, cam_from_gyro=CAM_FROM_GYRO, offset=offset, seed=3)
    track = GyroTrack()
    track.clock_known = clock_known
    track.add(t, w)

    assert calibrate(track, _pairs(rots, 1.0))
    assert abs(track.offset - offset) < 0.02
    assert angle_deg(track.cam_from_gyro.T @ CAM_FROM_GYRO) < 1.5
    # Five seconds of turning, between keyframes the calibration never saw as a pair.
    assert angle_deg(track.rel(T[300], T[450]).T @ (rots[300].T @ rots[450])) < 1.5


def test_too_few_pairs_leaves_it_uncalibrated():
    rots = _head_turns()
    track = GyroTrack()
    track.add(*synth(T, rots, seed=3))
    assert not calibrate(track, _pairs(rots, 1.0)[:10])
    assert not track.calibrated
