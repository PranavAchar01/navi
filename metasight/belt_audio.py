"""The sound stand-in for the belt's buzzers: what each motor command should sound like.

For early testing the 8 motors are heard instead of felt. One tone per motor, placed in stereo by
the motor's angle (front = 0, clockwise), pitched so front and back differ, gated by the motor's
duty so the belt's own rhythm (slow, fast, solid, guide taps) comes through as beeps. The iOS app
(ios/.../BeltAudio.swift) plays the same rules live; this module is the reference for them, the
unit tests, and the WAV renderer used to inspect the sound without speakers. Prototype.
"""

from __future__ import annotations

import asyncio
import math
import wave
from pathlib import Path

import numpy as np

from .belt import MODES
from .haptics import BeltBraille, play

N = 8
SAMPLE_RATE = 22_050
BASE_HZ = 660.0
FRONT_BACK_OCTAVES = 0.5  # front is half an octave above the sides, back half an octave below
GUIDE_RATIO = 1.5  # the "guide" taps sit a fifth up, unlike any obstacle tone
VOICE_GAIN = 0.35
RAMP_S = 0.004  # gain smoothing so the gate does not click


def angle_deg(i: int) -> float:
    return 45.0 * i


def pan(i: int) -> float:
    """-1 (left) .. +1 (right). Front and back both sit in the middle; pitch tells them apart."""
    return math.sin(math.radians(angle_deg(i)))


def stereo_gains(i: int) -> tuple[float, float]:
    """Equal-power (left, right) gains for motor i."""
    p = (pan(i) + 1) * math.pi / 4
    return math.cos(p), math.sin(p)


def frequency(i: int, mode: int = MODES["solid"]) -> float:
    f = BASE_HZ * 2 ** (FRONT_BACK_OCTAVES * math.cos(math.radians(angle_deg(i))))
    return f * GUIDE_RATIO if mode == MODES["guide"] else f


def amplitude(duty: int) -> float:
    return VOICE_GAIN * max(0, min(255, duty)) / 255


def duty_at(mode: int, inten: int, t_s: float) -> int:
    """Mirror of belt_core.h for one motor (no flags), so cues can be rendered without the belt."""
    ms = int(t_s * 1000)
    on = {
        MODES["slow"]: ms % 667 < 200,
        MODES["fast"]: ms % 250 < 125,
        MODES["solid"]: True,
        MODES["guide"]: ms % 1000 < 70 or 150 <= ms % 1000 < 220,
    }.get(mode, False)
    return inten if on else 0


Frames = list[tuple[float, list[tuple[int, int]]]]  # (start time s, 8 x (mode, duty))


def render(frames: Frames, total_s: float) -> np.ndarray:
    """Stereo float32 samples (n, 2). Each frame holds until the next one starts."""
    n = int(total_s * SAMPLE_RATE)
    out = np.zeros((n, 2), dtype=np.float64)
    t = np.arange(n) / SAMPLE_RATE
    starts = np.array([f[0] for f in frames])
    idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(frames) - 1)
    for m in range(N):
        amp = np.array([amplitude(f[1][m][1]) if f[1][m][0] else 0.0 for f in frames])[idx]
        amp[t < starts[0]] = 0.0
        k = max(1, int(RAMP_S * SAMPLE_RATE))
        amp = np.convolve(amp, np.ones(k) / k, mode="same")
        # a motor's pitch follows its mode; guide frames shift it, others use the base tone
        hz = np.array([frequency(m, f[1][m][0]) for f in frames])[idx]
        phase = 2 * math.pi * np.cumsum(hz) / SAMPLE_RATE
        voice = amp * np.sin(phase)
        left, right = stereo_gains(m)
        out[:, 0] += voice * left
        out[:, 1] += voice * right
    return np.tanh(out).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def cue_frames(modes: list[tuple[int, int]], seconds: float, step_s: float = 0.01) -> Frames:
    """Frames for a steady set of belt commands, with each mode's own rhythm applied."""
    frames: Frames = []
    for k in range(int(seconds / step_s)):
        t = k * step_s
        frames.append((t, [(m, duty_at(m, i, t)) for m, i in modes]))
    return frames


def braille_frames(cues: str) -> tuple[Frames, float]:
    """Run the belt Braille driver on a fake clock and capture what it sends, with times."""
    now = 0.0
    frames: Frames = []

    def send(cmd, flags):
        frames.append((now, list(cmd)))

    async def sleep(s):
        nonlocal now
        now += s

    drv = BeltBraille(send, MODES["solid"])
    drv.sleep = sleep
    asyncio.run(play(drv, "", cues))
    return frames, now + 0.2
