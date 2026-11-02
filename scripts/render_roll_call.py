"""Render the room roll call from a walk's pins to WAV for the video: the spoken line (OpenAI
TTS, voice cedar) with a pluck from each named thing's true direction as its name is said.

Run: PYTHONPATH=. .venv/bin/python scripts/render_roll_call.py data/walks/<walk>/pins.json \
        [--twin data/rooms/bedroom/v003/objects-s1.json] [--head X Y Z] [--yaw DEG] \
        [--room bedroom] [--no-voice]
The head looks along -z turned by --yaw degrees to the right, y up (the map frame). The OpenAI
key comes from the keychain (OPENAI_API_KEY) or the environment.
"""

import argparse
import io
import json
import math
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from metasight import roll_call as rc
from metasight import spatial_sound as ss
from metasight.config import ROOT

OUT = ROOT / "bench" / "results"


def openai_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    return subprocess.run(
        ["security", "find-generic-password", "-s", "OPENAI_API_KEY", "-w"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


def speak(text: str, key: str) -> np.ndarray:
    """Mono float clip at ss.SAMPLE_RATE from OpenAI TTS (cedar)."""
    import httpx

    r = httpx.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "gpt-4o-mini-tts",
            "voice": "cedar",
            "input": text,
            "response_format": "wav",
        },
        timeout=60,
    )
    r.raise_for_status()
    with wave.open(io.BytesIO(r.content)) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768
    g = math.gcd(rate, ss.SAMPLE_RATE)
    return resample_poly(x, ss.SAMPLE_RATE // g, rate // g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pins", type=Path)
    ap.add_argument("--head", type=float, nargs=3, default=(0.0, 1.5, 0.0))
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--room", default="bedroom")
    ap.add_argument("--twin", type=Path, help="the room twin's objects-s1.json, same frame")
    ap.add_argument("--no-voice", action="store_true")
    a = ap.parse_args()
    head = np.array(a.head)
    y = math.radians(a.yaw)
    fwd, up = np.array([math.sin(y), 0.0, -math.cos(y)]), np.array([0.0, 1.0, 0.0])

    def to(p):
        d = ss.relative(head, fwd, up, p)
        return d.ground_m, d.az_deg

    twin = json.loads(a.twin.read_text()) if a.twin else []
    items = rc.pick(json.loads(a.pins.read_text()), to, twin=twin)
    text, _ = rc.line(items, a.room)
    print(text)
    voice = None
    if not a.no_voice:
        key = openai_key()
        sentences = [f"{a.room.capitalize()}."] + [rc.sentence(g)[0] for g in rc.phrases(items)]
        voice = [speak(s, key) for s in sentences]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "roll-call.wav"
    ss.write_wav(path, rc.render(items, head, fwd, up, voice))
    (OUT / "roll-call.json").write_text(json.dumps(rc.message(items, a.room), indent=1))
    print(path)


if __name__ == "__main__":
    main()
