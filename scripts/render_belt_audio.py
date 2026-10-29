"""Render the belt's sound stand-in to WAV files in bench/results, to inspect without speakers.

Run: PYTHONPATH=. .venv/bin/python scripts/render_belt_audio.py
"""

from metasight import belt_audio as ba
from metasight.belt import MODES
from metasight.config import ROOT

OUT = ROOT / "bench" / "results"
OFF = (0, 0)


def only(i: int, mode: str, inten: int = 230) -> list[tuple[int, int]]:
    cmd = [OFF] * 8
    cmd[i] = (MODES[mode], inten)
    return cmd


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Each of the 8 motors in turn, front then clockwise, 0.5 s apiece: pan and pitch sweep.
    sweep = [(0.5 * i, only(i, "solid")) for i in range(8)]
    ba.write_wav(OUT / "belt-audio-sweep.wav", ba.render(sweep, 4.0))
    # Rhythms on the front motor, then the guide taps on the right one.
    rhythm = []
    for k, (mode, i) in enumerate([("slow", 0), ("fast", 0), ("solid", 0), ("guide", 2)]):
        rhythm += [(t + 2 * k, c) for t, c in ba.cue_frames(only(i, mode), 2.0)]
    ba.write_wav(OUT / "belt-audio-rhythms.wav", ba.render(rhythm, 8.0))
    frames, total = ba.braille_frames("keys tbl 4m ahd")
    ba.write_wav(OUT / "belt-audio-braille.wav", ba.render(frames, total))
    for f in sorted(OUT.glob("belt-audio-*.wav")):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
