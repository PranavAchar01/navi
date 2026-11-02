"""What the phone plays over a recorded take, as a stereo WAV to lay under the clip.

    PYTHONPATH=. .venv/bin/python scripts/render_take_sound.py bench/results/demo-1102/room.jsonl
    ... --out bench/results/room-sound.wav --pan --no-ticks

t = 0 is the take's first message. The guide beacon, the arrival notes and the hazard beeps
come from the recorded safety messages, decided on the phone's 40 ms grid
(metasight/spatial_sound.schedule_take).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from metasight import spatial_sound as ss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("take", type=Path, help="a judge replay or live .jsonl")
    ap.add_argument("--out", type=Path, help="default: next to the take, <name>-sound.wav")
    ap.add_argument("--pan", action="store_true", help="stereo pan instead of the KEMAR HRTF")
    ap.add_argument("--no-ticks", action="store_true", help="the beacon only")
    ap.add_argument("--no-beacon", action="store_true", help="the hazard beeps only")
    a = ap.parse_args()

    msgs = [json.loads(line) for line in a.take.read_text().splitlines() if line.strip()]
    beacon, ticks = not a.no_beacon, not a.no_ticks
    events = ss.schedule_take(msgs, beacon, ticks)
    x = ss.render_take(msgs, "pan" if a.pan else "hrtf", beacon, ticks)
    out = a.out or a.take.with_name(a.take.stem + "-sound.wav")
    ss.write_wav(out, x)
    kinds = Counter(e.kind for e in events)
    first = next((e.t for e in events if e.kind.startswith("beacon")), None)
    print(f"{out}: {len(x) / ss.SAMPLE_RATE:.1f} s, peak {np.abs(x).max():.2f}, {dict(kinds)}")
    if first is not None:
        print(f"first beacon at {first:.2f} s")


if __name__ == "__main__":
    main()
