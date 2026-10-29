"""A two-microphone track with two spoken sentences from different sides, for the end-to-end
check: python make_two_voices.py video.mp4 out.mov [secs] -> out.mov = video + this audio
(lossless: a codec smears the 0.3 ms inter-channel lag).
Voice 1 at -50 degrees from 1 s, voice 2 at +40 degrees from 10 s (printed as onsets)."""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

SR, SPACING, C = 16000, 0.14, 343.0
tmp = Path(tempfile.mkdtemp())


def say(voice, text):
    subprocess.run(["say", "-v", voice, "-o", str(tmp / "a.aiff"), text], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(tmp / "a.aiff"),
            "-ar",
            str(SR),
            "-ac",
            "1",
            str(tmp / "a.wav"),
        ],
        check=True,
    )
    return wavfile.read(tmp / "a.wav")[1].astype(np.float32) / 32768


def place(x, deg, rng):
    lag = SPACING / C * math.sin(math.radians(deg)) * SR
    k = np.arange(len(x))
    ch = [np.interp(k - s * lag / 2, k, x) for s in (1, -1)]
    tail = lambda: np.r_[0, rng.normal(size=2999) * np.exp(-6.9 * np.arange(2999) / (0.4 * SR))]  # noqa: E731
    return np.stack([c + 0.03 * fftconvolve(c, tail())[: len(c)] for c in ch], 1)


video, out = sys.argv[1], sys.argv[2]
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 27
rng = np.random.default_rng(0)
mix = np.zeros((int(secs * SR), 2), np.float32)
for onset, deg, voice, text in (
    (
        1.0,
        -50,
        "Samantha",
        "Hey, did you find the keys I left near the kitchen table this morning?",
    ),
    (10.0, 40, "Daniel", "I think we should take the left door, it leads straight to the garden."),
):
    seg = place(say(voice, text), deg, rng)
    i = int(onset * SR)
    mix[i : i + len(seg)] += seg[: len(mix) - i]
mix += rng.normal(size=mix.shape).astype(np.float32) * 0.003
wavfile.write(tmp / "mix.wav", SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        video,
        "-i",
        str(tmp / "mix.wav"),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "pcm_s16le",
        "-t",
        str(secs),
        out,
    ],
    check=True,
)
print("onsets 1.0 (-50 deg), 10.0 (+40 deg)")
