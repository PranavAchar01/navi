"""Tune the voice-direction gate: false bearings on noise-only runs (two independent channels,
512-sample blocks, every block flagged as voiced, the worst case) against how long a real voice
takes to get its first bearing. Tuning seeds and held-out seeds are separate.

python bench/people/tune_doa.py [two_voice_track.mov]  (the track from make_two_voices.py)
"""

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metasight.audio_doa import SR, Doa, Vad  # noqa: E402

BLOCK = 512
ONSETS = (1.0, 10.0)
TRUE = (-50.0, 40.0)


def false_rate(seeds, runs=300, **kw):
    hit = 0
    for sd in seeds:
        rng = np.random.default_rng(sd)
        for _ in range(runs // len(seeds)):
            d = Doa(0.14, **kw)
            for _ in range(int(rng.integers(8, 41))):
                d.push(rng.normal(size=(BLOCK, 2)) * 0.2, True)
                hit += d.bearing() is not None
                if d.bearing() is not None:
                    break
    return hit / runs


def voice_cost(track, **kw):
    """Seconds from each voice onset to its first bearing, and that bearing's error."""
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            track,
            "-vn",
            "-ar",
            str(SR),
            "-ac",
            "2",
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    x = np.frombuffer(pcm, "<i2").astype(np.float32).reshape(-1, 2) / 32768
    vad, out = Vad(), []
    for on, tr in zip(ONSETS, TRUE, strict=True):
        d, got = Doa(0.14, **kw), None
        i = int(on * SR) // BLOCK * BLOCK
        vad.reset()
        while i + BLOCK <= len(x) and i < (on + 6) * SR:
            blk = x[i : i + BLOCK]
            vad.push(blk.mean(axis=1))
            d.push(blk, vad.active())
            b = d.bearing()
            if b is not None:
                got = (round((i + BLOCK) / SR - on, 2), round(abs(b - tr), 1))
                break
            i += BLOCK
        out.append(got)
    return out


if __name__ == "__main__":
    track = sys.argv[1] if len(sys.argv) > 1 else None
    tune, held = (1, 2, 3), (101, 102, 103)
    grid = itertools.product((3, 6, 8, 12), (0.12, 0.2, 0.3), (10.0, 15.0, 20.0), (0.7, 0.85))
    for mv, pk, ad, ash in grid:
        kw = dict(min_votes=mv, peak_min=pk, agree_deg=ad, agree_share=ash)
        fa = false_rate(tune, **kw)
        row = f"min_votes={mv} peak={pk} agree={ad}/{ash}: noise tune {fa:.1%}"
        if fa <= 0.02:
            row += f" held {false_rate(held, **kw):.1%}"
            if track:
                row += f" voice {voice_cost(track, **kw)}"
        print(row)
