"""A simulated listener for the spatial sound cues: where would a person hear this?

SIMULATED LISTENER MODEL, NOT A HUMAN STUDY. The model hears direction from the interaural
time difference alone, the cue that carries left and right for sounds like ours:

- When the part below 1.2 kHz is within 20 dB of the whole, it reads the fine-structure
  delay: the peak of the left/right cross-correlation within the head's range (+-0.85 ms:
  KEMAR's low-frequency delay reaches 0.8). A pure tone has a peak every period, so a tone
  whose period is shorter than twice that range can be heard on the wrong side, as people
  do with tones.
- Otherwise it reads the delay of the envelope (Hilbert envelope, 400 Hz smoothing), which
  is coarse.
- Internal noise: a Gaussian jitter on the delay, 20 us for fine structure and 80 us for
  envelope (about the human just-noticeable differences).

The delay maps to a lateral angle through the listener's own head: the delays the KEMAR
head gives, at each angle, for noise with the sound's spectrum. Lateral angle leaves front
and back folded (the cone of confusion); a listener who turns the head 10 degrees right and
hears the next repeat unfolds it from which way the sound moved. Spectral (pinna) cues and
level differences are not modelled, so elevation and static front/back are not scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from metasight import spatial_sound as ss

FINE_HZ = 1200.0
MAX_ITD_S = 0.00085
# Low-frequency delay dominates whenever it is audible (Wightman and Kistler 1992): within
# 20 dB of the whole sound counts.
FINE_SHARE = 0.01
FINE_JITTER_S = 20e-6
ENV_JITTER_S = 80e-6
TURN_DEG = 10.0


@dataclass(frozen=True)
class Heard:
    itd_s: float  # positive: the right ear leads (source on the right)
    fine: bool  # read from fine structure (else the envelope)


def _xcorr_lag(a: np.ndarray, b: np.ndarray) -> float:
    """Seconds by which b lags a, from the cross-correlation peak within +-MAX_ITD_S, with a
    parabolic fit between samples."""
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    m = int(math.ceil(MAX_ITD_S * ss.SAMPLE_RATE))
    lags = np.arange(-m, m + 1)
    v = c[lags % size]
    i = int(np.argmax(v))
    shift = 0.0
    if 0 < i < len(v) - 1:
        den = v[i - 1] - 2 * v[i] + v[i + 1]
        if den != 0:
            shift = 0.5 * (v[i - 1] - v[i + 1]) / den
    return (lags[i] + shift) / ss.SAMPLE_RATE


def hear(x: np.ndarray, fine: bool | None = None) -> Heard:
    """The interaural delay of a stereo sound (n, 2), noise-free. fine forces the path."""
    left, right = x[:, 0].astype(float), x[:, 1].astype(float)
    lo = butter(4, FINE_HZ, "low", fs=ss.SAMPLE_RATE, output="sos")
    ll, rl = sosfiltfilt(lo, left), sosfiltfilt(lo, right)
    if fine is None:
        total = float(np.sum(left**2 + right**2)) or 1.0
        fine = float(np.sum(ll**2 + rl**2)) / total >= FINE_SHARE
    if fine:
        return Heard(_xcorr_lag(rl, ll), True)  # left lags when the right ear leads
    hi = butter(4, FINE_HZ, "high", fs=ss.SAMPLE_RATE, output="sos")
    smooth = butter(2, 400.0, "low", fs=ss.SAMPLE_RATE, output="sos")
    el = sosfiltfilt(smooth, np.abs(hilbert(sosfiltfilt(hi, left))))
    er = sosfiltfilt(smooth, np.abs(hilbert(sosfiltfilt(hi, right))))
    return Heard(_xcorr_lag(er, el), False)


_maps: dict[tuple[bytes, bool], tuple[np.ndarray, np.ndarray]] = {}


def head_map(mono: np.ndarray, fine: bool) -> tuple[np.ndarray, np.ndarray]:
    """The listener's learned delay for each lateral angle 0..90 at this sound's frequencies:
    what the KEMAR head (the stand-in for a real head) does to noise with the sound's
    spectrum. A real head's delay grows toward low frequencies, so the map is per sound; the
    noise has no period, so the map itself is never ambiguous."""
    key = (mono.tobytes(), fine)
    if key not in _maps:
        n = int(0.2 * ss.SAMPLE_RATE)
        mag = np.abs(np.fft.rfft(mono, n))
        phase = np.random.default_rng(7).uniform(0, 2 * np.pi, len(mag))
        noise = np.fft.irfft(mag * np.exp(1j * phase), n)
        angles = np.arange(0.0, 91.0, 5.0)
        itds = [hear(ss.place(noise, ss.Direction(a, 0.0, 1.0, 1.0)), fine).itd_s for a in angles]
        _maps[key] = (np.maximum.accumulate(itds), angles)
    return _maps[key]


def lateral_deg(itd_s: float, head: tuple[np.ndarray, np.ndarray]) -> float:
    """The lateral angle (deg, + right) heard for a delay on a head map, held at 90 beyond
    the largest delay it knows."""
    itds, angles = head
    return math.copysign(float(np.interp(abs(itd_s), itds, angles)), itd_s)


def true_lateral(az_deg: float) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, math.sin(math.radians(az_deg))))))


def wrap(deg: float) -> float:
    return ((deg + 180) % 360) - 180


def localise(
    mono: np.ndarray, az_deg: float, method: str, rng: np.random.Generator, reps: int = 20
) -> dict[str, float]:
    """Play mono at az_deg (elevation 0), then again after a 10 degree head turn right, and
    score the simulated listener over reps internal-noise draws: lateral error (deg), full
    azimuth error after the turn (deg), and the share heard on the wrong side."""
    d0 = ss.Direction(az_deg, 0.0, 1.0, 1.0)
    d1 = ss.Direction(wrap(az_deg - TURN_DEG), 0.0, 1.0, 1.0)
    h0 = hear(ss.place(mono, d0, method))
    h1 = hear(ss.place(mono, d1, method), h0.fine)
    head = head_map(mono, h0.fine)
    lat_err, az_err, wrong_side = [], [], 0
    truth = true_lateral(az_deg)
    for _ in range(reps):
        j = FINE_JITTER_S if h0.fine else ENV_JITTER_S
        lat0 = lateral_deg(h0.itd_s + rng.normal(0, j), head)
        lat1 = lateral_deg(h1.itd_s + rng.normal(0, j), head)
        lat_err.append(abs(lat0 - truth))
        front = lat1 <= lat0  # turning right moves a source ahead to the left
        est = lat0 if front else wrap(math.copysign(180.0, lat0 or 1.0) - lat0)
        az_err.append(abs(wrap(est - az_deg)))
        if abs(truth) > 5 and np.sign(lat0) != np.sign(truth):
            wrong_side += 1
    return {
        "lateral_err": float(np.mean(lat_err)),
        "az_err": float(np.mean(az_err)),
        "wrong_side": wrong_side / reps,
    }
