"""Who is talking and from where, from the wearer's microphones.

Two jobs, both anonymous (no voice is ever matched to a person, nothing is kept but the last
fraction of a second):

* `Vad`: is anyone speaking right now. Silero VAD (MIT, 16 kHz, 32 ms hops) when the package is
  installed, a plain energy gate when it is not.
* `gcc_phat` / `Doa`: which way the voice comes from, from the delay between two microphones.
  GCC-PHAT finds the lag with the sharpest cross-correlation peak after whitening, and the
  bearing is asin(c * lag / spacing). Two microphones on one axis cannot tell front from back:
  the answer is taken as in front (the camera is there, and a person you can see is the useful
  case); a source behind the wearer reads as its mirror image in front. One microphone gives no
  direction at all, and `Doa` says so by returning None.

Bearing is degrees clockwise from straight ahead (positive right), like the rest of the project.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

SOUND_MPS = 343.0
SR = 16000
AGREE_DEG = 20.0  # votes within this of the median count as agreeing
AGREE_SHARE = 0.7  # and they must be at least this share of the window
BAND = (200.0, 4000.0)  # where speech has energy; GCC-PHAT looks only here
HOP = 512  # Silero's frame at 16 kHz (32 ms)


def to_mono16k(pcm: np.ndarray, sr: int) -> np.ndarray:
    """float32 mono at 16 kHz from float32 mono at `sr` (linear resample, enough for a gate)."""
    if sr == SR:
        return pcm
    n = int(round(len(pcm) * SR / sr))
    return np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.float32)


class Vad:
    """Streaming voice activity. `push` takes mono float32 at 16 kHz and returns the speech
    probability of the newest complete 32 ms hop (or None until one is complete)."""

    def __init__(self, threshold: float = 0.5, hangover_s: float = 0.4, model: bool = True):
        self.threshold, self.hangover_s = threshold, hangover_s
        self.model = None
        if model:
            try:
                from silero_vad import load_silero_vad

                self.model = load_silero_vad()
            except Exception:
                self.model = None
        self.buf = np.zeros(0, np.float32)
        self.prob = 0.0
        self.last_voice = -math.inf
        self.t = 0.0  # audio clock, seconds
        self.noise = 1.0  # energy fallback: running floor of the quiet level

    @property
    def kind(self) -> str:
        return "silero" if self.model is not None else "energy"

    def _hop_prob(self, x: np.ndarray) -> float:
        if self.model is not None:
            import torch

            with torch.no_grad():
                return float(self.model(torch.from_numpy(x), SR).item())
        rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
        # The floor drops to any quiet hop at once and creeps up about 10% a second, so speech
        # that lasts a few seconds does not become the floor.
        self.noise = rms if rms < self.noise else min(self.noise * 1.003, rms)
        return float(np.clip(rms / (4 * self.noise) - 0.5, 0.0, 1.0))

    def push(self, mono16k: np.ndarray) -> float | None:
        self.buf = np.concatenate([self.buf, mono16k.astype(np.float32)])
        out = None
        while len(self.buf) >= HOP:
            hop, self.buf = self.buf[:HOP], self.buf[HOP:]
            self.prob = self._hop_prob(hop)
            self.t += HOP / SR
            if self.prob >= self.threshold:
                self.last_voice = self.t
            out = self.prob
        return out

    def active(self) -> bool:
        return self.t - self.last_voice <= self.hangover_s

    def reset(self) -> None:
        self.buf = np.zeros(0, np.float32)
        self.last_voice, self.t = -math.inf, 0.0
        if self.model is not None:
            self.model.reset_states()


def gcc_phat(a: np.ndarray, b: np.ndarray, max_lag_s: float, sr: int = SR, interp: int = 8):
    """Delay of `a` behind `b` in seconds (positive: `a` hears it later), and the peak's
    coherence in 0..1 (the share of the speech band whose phases agree at that delay)."""
    n = 1 << (len(a) + len(b) - 1).bit_length()
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    f = np.fft.rfftfreq(n, 1 / sr)
    inband = (f >= BAND[0]) & (f <= BAND[1])
    R[~inband] = 0  # whitening an empty band lets noise set the peak
    cc = np.fft.irfft(R, n * interp)
    m = int(interp * sr * max_lag_s) + 1
    cc = np.concatenate([cc[-m:], cc[: m + 1]])
    k = int(np.argmax(cc))
    coh = float(cc[k] * n * interp / (2 * max(1, int(inband.sum()))))
    return (k - m) / (interp * sr), coh


class Doa:
    """Bearing of the voice from a two-microphone pair, smoothed over the voiced stretch.

    Channel 0 is the left microphone, channel 1 the right, `spacing_m` apart on the head's
    left-right axis. `push` takes a (n, 2) float32 block and a flag saying the VAD heard speech;
    only voiced blocks vote. `bearing()` is the median of the recent votes, or None when there are
    too few or most of them disagree (noise, a smeared lag, or two people talking at once).
    """

    def __init__(
        self, spacing_m: float, window: int = 12, min_votes: int = 3, peak_min: float = 0.12
    ):
        self.spacing_m = spacing_m
        self.min_votes, self.peak_min = min_votes, peak_min
        self.votes: deque[float] = deque(maxlen=window)

    def push(self, block: np.ndarray, voiced: bool) -> float | None:
        if not voiced or block.ndim != 2 or block.shape[1] < 2 or len(block) < 256:
            return None
        lag, peak = gcc_phat(block[:, 0], block[:, 1], self.spacing_m / SOUND_MPS)
        if peak < self.peak_min:
            return None
        s = lag * SOUND_MPS / self.spacing_m
        # Channel 0 hearing it later means the source is on the right.
        deg = math.degrees(math.asin(max(-1.0, min(1.0, s))))
        self.votes.append(deg)
        return deg

    def bearing(self) -> float | None:
        if len(self.votes) < self.min_votes:
            return None
        v = np.array(self.votes)
        med = float(np.median(v))
        # A real voice makes most votes agree; noise or a smeared lag scatters them across the arc.
        if float(np.mean(np.abs(v - med) <= AGREE_DEG)) < AGREE_SHARE:
            return None
        return med

    def reset(self) -> None:
        self.votes.clear()
