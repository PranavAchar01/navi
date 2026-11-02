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
hears the next repeat unfolds it from which way the sound moved.

Static front/back (front_back): from one presentation, with no turn, the listener compares
the third-octave spectrum at each ear (1 to 16 kHz, level-normalised, its own beep's spectrum
known) against its learned transfer for the direction and for the mirrored one, with 1 dB of
internal noise per band. Its ears are not the renderer's (KEMAR's large pinna against the
normal one), as a wearer's are not the phone's generic HRTF. Elevation is not scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, hilbert, sosfiltfilt

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


# ---- static front/back: the spectral (pinna) cue, one sound, no head turn ----

# Third-octave bands where the pinna's front/back notches and peaks sit.
FB_BANDS_HZ = tuple(1000 * 2 ** (k / 3) for k in range(0, 13))  # 1 to 16 kHz
ALL_BANDS_HZ = tuple(1000 * 2 ** (k / 3) for k in range(-10, 13))  # 100 Hz to 16 kHz
FB_FLOOR_DB = 40.0  # a band this far below the sound's strongest band carries no cue
FB_JITTER_DB = 1.0  # internal noise per band and ear, about the level JND
FB_MIN_SEP_DEG = 30.0  # a direction this close to its mirror has no front/back to get wrong


def _band_db(x: np.ndarray, n: int, bands: tuple[float, ...] = FB_BANDS_HZ) -> np.ndarray:
    """Energy (dB) in each third-octave band of a mono signal, zero-padded to n."""
    p = np.abs(np.fft.rfft(x, n)) ** 2
    f = np.fft.rfftfreq(n, 1 / ss.SAMPLE_RATE)
    out = []
    for c in bands:
        sel = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        out.append(10 * np.log10(np.sum(p[sel]) + 1e-20))
    return np.array(out)


def _ears_db(x: np.ndarray, n: int) -> np.ndarray:
    return np.stack([_band_db(x[:, 0], n), _band_db(x[:, 1], n)])


# The listener's own ears are not the renderer's: KEMAR's large pinna (DB-065, the full set's
# right ear) against the normal pinna (DB-061) the compact set renders with. Two pinnae on one
# head is the nearest stand-in in this data for a stranger's ears, which is why real wearers
# confuse front and back on generic HRTFs.
LISTENER_EARS = Path(__file__).parent / "hrtf" / "mit_kemar_large_pinna_el0.npz"


@cache
def _listener_ears() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(LISTENER_EARS)
    return z["azim"].astype(float), z["hrir_right"].astype(np.float64) / 32768


def listener_hrir(az_deg: float) -> np.ndarray:
    """(taps, 2) for the listener's ears at ear height: the right ear's response at az, the
    left ear's at the mirrored az (a symmetric head)."""
    azim, ir = _listener_ears()

    def ear(a: float) -> np.ndarray:
        a %= 360.0
        j = int(a // 5) % len(azim)
        w = (a - azim[j]) / 5.0
        return (1 - w) * ir[j] + w * ir[(j + 1) % len(azim)]

    return np.stack([ear(-az_deg), ear(az_deg)], axis=1)


_templates: dict[float, np.ndarray] = {}


def template(az_deg: float) -> np.ndarray:
    """The listener's learned transfer at a direction: what its own ears do to flat noise, per
    band and ear (dB). The same for both renderers: the wearer's ears are what it learned."""
    if az_deg not in _templates:
        n = 1 << 14
        noise = np.random.default_rng(11).standard_normal(n // 2)
        h = listener_hrir(az_deg)
        y = np.stack([fftconvolve(noise, h[:, 0]), fftconvolve(noise, h[:, 1])], axis=1)
        _templates[az_deg] = _ears_db(y, n) - _band_db(noise, n)
    return _templates[az_deg]


def _shape_dist(heard: np.ndarray, t: np.ndarray, use: np.ndarray) -> float:
    """Mean band difference once the overall level is taken out (distance changes level)."""
    d = (heard - t)[:, use]
    return float(np.mean(np.abs(d - d.mean())))


def mirror(az_deg: float) -> float:
    """The direction with the same lateral angle on the other side of the ear axis."""
    return wrap(180.0 - az_deg)


def front_back(
    mono: np.ndarray, az_deg: float, method: str, rng: np.random.Generator, reps: int = 20
) -> float | None:
    """Share of single presentations heard in the mirrored hemisphere (front for back or back
    for front), from spectral shape alone. The listener knows the cue's own spectrum (it hears
    the same beep all day) and compares what reaches each ear, level-normalised, against its
    learned transfer for the true direction and for the mirror. Bands the sound does not fill
    carry nothing; with no usable band it is a coin toss. None where the mirror is too close
    to call."""
    m = mirror(az_deg)
    if abs(wrap(m - az_deg)) < FB_MIN_SEP_DEG:
        return None
    n = 1 << max(14, (len(mono) * 2 - 1).bit_length())
    src = _band_db(mono, n)
    # Usable: within FB_FLOOR_DB of the strongest third-octave band anywhere in the sound.
    use = src >= _band_db(mono, n, ALL_BANDS_HZ).max() - FB_FLOOR_DB
    if not use.any():
        return 0.5
    got = _ears_db(ss.place(mono, ss.Direction(az_deg, 0.0, 1.0, 1.0), method), n) - src
    t_true, t_mirror = template(az_deg), template(m)
    wrong = 0
    for _ in range(reps):
        heard = got + rng.normal(0, FB_JITTER_DB, got.shape)
        wrong += _shape_dist(heard, t_mirror, use) < _shape_dist(heard, t_true, use)
    return wrong / reps
