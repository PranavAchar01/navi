"""How often does the voice direction read more than 35 degrees from the truth (the point where
`Conversation` trusts the heard direction over a lip match) in a reverberant room, and how often
does independent reverb-and-noise alone produce a bearing? 1600-sample blocks, like the phone.

python bench/people/reverb_false_cues.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import eval_audio as E  # noqa: E402

from metasight.audio_doa import Doa, Vad  # noqa: E402

BLOCK = 1600


def run(x2, vad):
    d = Doa(E.SPACING)
    vad.reset()
    est = None
    for s in range(0, len(x2) - BLOCK, BLOCK):
        b = x2[s : s + BLOCK]
        vad.push(b.mean(1))
        d.push(b, vad.active())
        est = d.bearing() or est
    return est


def main():
    rng = np.random.default_rng(7)
    clean = [E.speech(i) for i in range(len(E.TEXT))]
    vad = Vad()
    total = answered = wrong = 0
    for rt60 in (0.4, 0.8):
        for snr in (10, 0):
            for noise in ("white", "diffuse"):
                for deg in (-60, -20, 0, 20, 60):
                    for x in clean:
                        est = run(E.spatialise(x, deg, rng, rt60, snr, noise), vad)
                        total += 1
                        if est is not None:
                            answered += 1
                            wrong += abs(est - deg) > 35
    print(f"reverberant runs {total}, answered {answered}, more than 35 deg off {wrong}")
    # reverb-and-noise only: the room's tail with no direct voice, every block flagged voiced
    hit = 0
    runs = 600
    for _ in range(runs):
        d = Doa(E.SPACING)
        for _ in range(int(rng.integers(8, 41))):
            tail = np.stack(
                [np.convolve(rng.normal(size=BLOCK), E.rir(rng, 0.8))[:BLOCK] for _ in (0, 1)], 1
            )
            d.push(tail.astype(np.float32), True)
        hit += d.bearing() is not None
    print(f"tail-only runs {runs}, false bearings {hit}")


if __name__ == "__main__":
    main()
