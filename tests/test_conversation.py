import base64
from types import SimpleNamespace

import numpy as np

from metasight.belt import MODES
from metasight.conversation import Talk


def chunk(x, ch=2, sr=16000, spacing=0.14):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    return {"type": "audio", "sr": sr, "ch": ch, "spacing": spacing, "pcm": base64.b64encode(pcm)}


def heading():
    return SimpleNamespace(pos=np.array([0.0, 0.0, 0.0]), yaw=0.0, hfov_deg=90.0)


def person(cx, dist=2.0):
    return SimpleNamespace(
        label="person", score=0.9, box=(cx - 0.08, 0.2, cx + 0.08, 0.8), dist_m=dist
    )


def noisy_burst(n, deg, rng, spacing=0.14):
    lag = spacing / 343.0 * np.sin(np.radians(deg)) * 16000
    src = rng.normal(size=n + 100) * 0.2
    k = np.arange(n)
    return np.stack(
        [
            np.interp(k - lag / 2 + 50, np.arange(len(src)), src),
            np.interp(k + lag / 2 + 50, np.arange(len(src)), src),
        ],
        1,
    )


def test_no_audio_means_no_speaker_and_no_cue():
    t = Talk(90.0)
    bgr = np.zeros((240, 320, 3), np.uint8)
    for k in range(6):
        ev = t.frame(k / 15, [person(0.5)], bgr, heading())
    assert ev["speaker"] is None and ev["cue"] is None and ev["mics"] == 0
    assert ev["tracks"] and "embedding" not in ev["tracks"][0]


def test_overlay_only_fills_free_motors_and_expires():
    t = Talk(90.0)
    t.cue = {"motors": [[1, 0.9], [2, 0.4]]}
    t.cue_until = 1.0
    cmd = [(MODES["off"], 0)] * 8
    cmd[1] = (MODES["fast"], 255)
    out = t.overlay(cmd, 0.5)
    assert out[1] == (MODES["fast"], 255) and out[2][0] == MODES["slow"]
    assert t.overlay(cmd, 2.0) == cmd


def test_audio_chunk_decodes_and_marks_mics():
    t = Talk(90.0)
    rng = np.random.default_rng(0)
    t.audio(chunk(rng.normal(size=(4000, 2)) * 0.01), 100.0)
    assert t.mics == 2 and t.vad is not None and t.doa is not None
    t.audio(chunk(rng.normal(size=(4000, 1)) * 0.01, ch=1), 100.1)
    assert t.mics == 1
