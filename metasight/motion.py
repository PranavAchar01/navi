"""Fall check-in and shaking notice from device motion. Pure logic: time comes from the samples.

Fall: the public threshold approach (Bourke et al. 2007, Gait & Posture): an impact spike, then a
sustained orientation change, then stillness. A spoken or tapped "I'm okay" cancels it; no answer
for `answer_s` only reports that it *would* call the emergency contact. Nothing here dials.

Shaking: sustained rhythmic, high-energy motion over a multi-second window (the public
wrist-accelerometer literature, for example Beniczky et al. 2013, Epilepsia). Thresholds are
placeholders to be tuned on staged recordings. The wording is observational only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

G = 9.80665

FALL_PROMPT = "Are you okay? Do you need me to call an ambulance?"
FALL_ESCALATE = "No answer. This would call your emergency contact."
SHAKE_PROMPT = "I noticed sustained shaking. Are you okay?"


@dataclass(frozen=True)
class Sample:
    t: float  # seconds
    ax: float  # acceleration including gravity, m/s^2 (sign convention does not matter)
    ay: float
    az: float

    @property
    def vec(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az]) / G


@dataclass(frozen=True)
class Event:
    kind: str  # fall_prompt | fall_cancelled | fall_escalate | shake_notice
    t: float
    message: str = ""


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


class FallDetector:
    def __init__(
        self,
        impact_g: float = 2.5,
        tilt_deg: float = 45.0,
        still_std_g: float = 0.15,
        settle_s: float = 0.6,
        window_s: float = 1.6,
        answer_s: float = 30.0,
        cooldown_s: float = 20.0,
    ):
        self.impact_g, self.tilt_deg, self.still_std_g = impact_g, tilt_deg, still_std_g
        self.settle_s, self.window_s = settle_s, window_s
        self.answer_s, self.cooldown_s = answer_s, cooldown_s
        self.state = "idle"  # idle | watching | asking
        self._hist: deque[Sample] = deque()
        self._baseline: np.ndarray | None = None
        self._impact_t = 0.0
        self._deadline = 0.0
        self._quiet_until = 0.0

    def _baseline_gravity(self, now: float) -> np.ndarray | None:
        old = [s.vec for s in self._hist if now - 2.5 <= s.t <= now - 0.3]
        return np.mean(old, axis=0) if len(old) >= 5 else None

    def update(self, s: Sample) -> list[Event]:
        events: list[Event] = []
        self._hist.append(s)
        while self._hist and s.t - self._hist[0].t > 6.0:
            self._hist.popleft()
        mag = float(np.linalg.norm(s.vec))

        if self.state == "idle":
            if mag > self.impact_g and s.t >= self._quiet_until:
                base = self._baseline_gravity(s.t)
                if base is not None:
                    self._baseline, self._impact_t, self.state = base, s.t, "watching"
        elif self.state == "watching":
            lo, hi = self._impact_t + self.settle_s, self._impact_t + self.settle_s + self.window_s
            if s.t >= hi:
                win = [x.vec for x in self._hist if lo <= x.t <= hi]
                self.state = "idle"
                if len(win) >= 5:
                    mean = np.mean(win, axis=0)
                    mags = np.linalg.norm(win, axis=1)
                    tilted = _angle_deg(mean, self._baseline) > self.tilt_deg
                    still = float(np.std(mags)) < self.still_std_g
                    if tilted and still:
                        self.state = "asking"
                        self._deadline = s.t + self.answer_s
                        events.append(Event("fall_prompt", s.t, FALL_PROMPT))
        elif self.state == "asking" and s.t >= self._deadline:
            self.state, self._quiet_until = "idle", s.t + self.cooldown_s
            events.append(Event("fall_escalate", s.t, FALL_ESCALATE))
        return events

    def respond_ok(self, t: float) -> list[Event]:
        if self.state != "asking":
            return []
        self.state, self._quiet_until = "idle", t + self.cooldown_s
        return [Event("fall_cancelled", t, "Okay. Glad you're alright.")]

    def trigger_test(self, t: float) -> list[Event]:
        """Demo button: start the check-in without a fall. Same prompt, same cancel and timeout."""
        self.state, self._deadline = "asking", t + self.answer_s
        return [Event("fall_prompt", t, FALL_PROMPT)]


class ShakeDetector:
    def __init__(
        self,
        window_s: float = 4.0,
        rate_hz: float = 50.0,
        band: tuple[float, float] = (3.0, 8.0),
        rms_g: float = 0.35,
        band_fraction: float = 0.5,
        sustain_s: float = 5.0,
        cooldown_s: float = 60.0,
        step_s: float = 0.5,
    ):
        self.window_s, self.rate_hz, self.band = window_s, rate_hz, band
        self.rms_g, self.band_fraction = rms_g, band_fraction
        self.sustain_s, self.cooldown_s, self.step_s = sustain_s, cooldown_s, step_s
        self._hist: deque[Sample] = deque()
        self._next_eval = 0.0
        self._run_start: float | None = None
        self._last_hit = 0.0
        self._quiet_until = 0.0

    def window_stats(self, now: float) -> tuple[float, float] | None:
        """(band RMS in g, fraction of 0.5-15 Hz energy inside the band) for the last window."""
        win = [x for x in self._hist if now - self.window_s <= x.t <= now]
        if len(win) < 20 or win[-1].t - win[0].t < self.window_s * 0.75:
            return None
        t = np.array([x.t for x in win])
        grid = np.arange(t[0], t[-1], 1.0 / self.rate_hz)
        sig = np.stack([np.interp(grid, t, [x.vec[i] for x in win]) for i in range(3)])
        sig = sig - sig.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(sig, axis=1)) ** 2
        freqs = np.fft.rfftfreq(sig.shape[1], 1.0 / self.rate_hz)
        power = spec.sum(axis=0)
        total = power[(freqs >= 0.5) & (freqs <= 15)].sum()
        in_band = (freqs >= self.band[0]) & (freqs <= self.band[1])
        if total <= 0:
            return None
        n = sig.shape[1]
        band_rms = float(np.sqrt(2 * power[in_band].sum() / n**2))  # Parseval, one-sided
        return band_rms, float(power[in_band].sum() / total)

    def update(self, s: Sample) -> list[Event]:
        self._hist.append(s)
        while self._hist and s.t - self._hist[0].t > self.window_s + 1.0:
            self._hist.popleft()
        if s.t < self._next_eval:
            return []
        self._next_eval = s.t + self.step_s
        stats = self.window_stats(s.t)
        hit = stats is not None and stats[0] >= self.rms_g and stats[1] >= self.band_fraction
        if not hit:
            if self._run_start is not None and s.t - self._last_hit > 1.0:
                self._run_start = None
            return []
        self._last_hit = s.t
        if self._run_start is None:
            self._run_start = s.t
        if s.t - self._run_start >= self.sustain_s and s.t >= self._quiet_until:
            self._run_start, self._quiet_until = None, s.t + self.cooldown_s
            return [Event("shake_notice", s.t, SHAKE_PROMPT)]
        return []
