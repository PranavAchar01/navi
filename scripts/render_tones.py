"""Render the glasses-speaker cues to WAVs and measure them (GAP 5), so they can be heard on
earbuds without a phone build.

  side beeps   the obstacle beeps of ios/.../BeepCue.swift (tick 2000 Hz 12 ms, soft 880 Hz 70 ms,
               clear 1320 Hz 90 ms, repeat rates from BeepCue.interval), placed left, right,
               ahead and behind
  guide tones  the "find my X" taps (metasight/belt_audio.py: a fifth above the obstacle tone),
               far to near, on the left, ahead and the right

Stereo here is equal-power panning by sin(bearing). The app's real beeps go through
AVAudio3D head-related filtering, which this does not reproduce, so left/right levels match but
the in-head versus out-of-head feel does not. Written to showcase/rec/audio/tones/ with
report.json (peak, RMS, clipping, left/right balance, dominant frequency per file).

    PYTHONPATH=. .venv/bin/python scripts/render_tones.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from metasight import belt_audio as ba
from metasight.belt import MODES, guide

OUT = Path(__file__).resolve().parent.parent / "showcase" / "rec" / "audio" / "tones"
SR = ba.SAMPLE_RATE
KINDS = {"tick": (2000.0, 0.012, 0.15), "soft": (880.0, 0.07, 0.4), "clear": (1320.0, 0.09, 0.8)}
# (kind, seconds between beeps) as BeepCue.interval at a few distances
BANDS = {"yellow 1.5m": ("tick", 1.6), "orange 0.8m": ("soft", 0.5), "red 0.3m": ("clear", 0.2)}


def beep(kind: str, volume: float = 1.0) -> np.ndarray:
    hz, dur, gain = KINDS[kind]
    n = int(dur * SR)
    t = np.arange(n)
    ramp = max(1, int(n * 0.25))
    edge = np.minimum(t, n - 1 - t)
    env = np.where(edge >= ramp, 1.0, 0.5 * (1 - np.cos(np.pi * edge / ramp)))
    return (np.sin(2 * np.pi * hz * t / SR) * env * gain * volume).astype(np.float64)


def place(mono: np.ndarray, bearing_deg: float) -> np.ndarray:
    p = (math.sin(math.radians(bearing_deg)) + 1) * math.pi / 4
    return np.stack([mono * math.cos(p), mono * math.sin(p)], axis=1)


def side_beeps(bearing: float, kind: str, gap: float, seconds: float = 4.0) -> np.ndarray:
    out = np.zeros((int(seconds * SR), 2))
    b = place(beep(kind), bearing)
    t = 0.2
    while t + len(b) / SR < seconds:
        i = int(t * SR)
        out[i : i + len(b)] += b
        t += gap
    return out


def guide_tone(bearing: float, dist: float, seconds: float = 3.0) -> np.ndarray:
    cmd = [(MODES["off"], 0)] * 8
    cmd = list(cmd)
    guide(cmd, bearing, dist)
    return ba.render(ba.cue_frames(cmd, seconds), seconds)


def measure(x: np.ndarray) -> dict:
    mono = x.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    peak = float(np.abs(x).max())
    rms = lambda a: float(np.sqrt(np.mean(a**2)))  # noqa: E731
    active = np.abs(mono) > 1e-4
    left, right = rms(x[:, 0]), rms(x[:, 1])
    sounding = rms(mono[active]) if active.any() else 1e-9
    return {
        "peak_dbfs": round(20 * math.log10(max(peak, 1e-9)), 1),
        "rms_dbfs_while_sounding": round(20 * math.log10(max(sounding, 1e-9)), 1),
        "clipped_samples": int((np.abs(x) >= 0.999).sum()),
        "lr_db": round(20 * math.log10(max(right, 1e-9) / max(left, 1e-9)), 1),
        "dominant_hz": round(float(np.argmax(spec) * SR / len(mono)), 0),
        "sounding_fraction": round(float(active.mean()), 2),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for side, bearing in (("left", 270.0), ("right", 90.0), ("ahead", 0.0), ("behind", 180.0)):
        for label, (kind, gap) in BANDS.items():
            name = f"beep-{side}-{label.replace(' ', '-')}.wav"
            x = side_beeps(bearing, kind, gap)
            ba.write_wav(OUT / name, x)
            report[name] = measure(x)
    for side, bearing in (("left", 270.0), ("ahead", 0.0), ("right", 90.0)):
        for dist in (4.0, 2.0, 0.8):
            name = f"guide-{side}-{dist:g}m.wav"
            x = guide_tone(bearing if bearing < 180 else bearing - 360, dist)
            ba.write_wav(OUT / name, x)
            report[name] = measure(x)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(
            f"{k:34s} peak {v['peak_dbfs']:6} rms {v['rms_dbfs_while_sounding']:6} "
            f"clip {v['clipped_samples']} L/R {v['lr_db']:6} Hz {v['dominant_hz']}"
        )


if __name__ == "__main__":
    main()
