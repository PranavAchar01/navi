"""Does voice isolation help Muse Voice Transcribe hear a question over room noise?

Five spoken questions (macOS `say`, 16 kHz) are mixed with two kinds of household noise at a
fixed loudness: a fan-like hum (pink noise) and a TV in the room (a second voice reading a news
paragraph). Each mix is sent to Transcribe twice, raw and after the isolator
(`metasight/isolate.py`, Meta's open-weight dns48), and scored by word error rate against the
question, and by SI-SDR against the clean speech. When the Meta key is refused (billing), the
words are heard by parakeet on this Mac instead, and the result file says so. The isolator is
fed 64 ms chunks one at a time, exactly as the server feeds it, so the per-chunk time and the
model's look-ahead are what the live path pays.

Run: .venv/bin/python bench/voice_isolation.py [snr_db ...]  ->  bench/results/voice-isolation.json
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))
from meta_models_probe import transcribe_file, wav  # noqa: E402

from metasight.isolate import Isolator  # noqa: E402

SR = 16000
CHUNK = 1024  # what web/src/mic.ts sends (64 ms)
SNRS = [float(x) for x in sys.argv[1:]] or [0.0, 5.0]
QUESTIONS = [
    "where are my keys",
    "where did I leave my wallet",
    "is there anything in front of the door",
    "where is my phone",
    "what changed in the kitchen",
]
TV = (
    "Good evening. Officials said on Tuesday that the new bridge will open next spring, two "
    "years later than planned, after costs rose sharply over the summer. In other news, the "
    "weather stays mild through the weekend with light rain expected on Sunday afternoon."
)
OUT = ROOT / "bench" / "results" / "voice-isolation.json"


def say(text: str, voice: str | None = None) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        aiff = Path(d) / "s.aiff"
        subprocess.run(
            ["say", *(["-v", voice] if voice else []), "-o", str(aiff), text], check=True
        )
        raw = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(aiff), "-f", "s16le", "-ac", "1",
             "-ar", str(SR), "-"],
            check=True, capture_output=True,
        ).stdout  # fmt: skip
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / np.sqrt(np.maximum(f, 20))
    x = np.fft.irfft(spec, n)
    return (x / np.std(x)).astype(np.float32)


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    pad = np.zeros(int(0.4 * SR), np.float32)  # room sound before and after, as a live mic
    s = np.concatenate([pad, speech, pad])
    n = np.resize(noise, s.size)
    gain = np.sqrt(np.mean(speech**2) / (np.mean(n**2) * 10 ** (snr_db / 10)))
    y = s + gain * n
    return (y / max(1.0, float(np.max(np.abs(y))))).astype(np.float32)


def isolate(x: np.ndarray, iso: Isolator) -> tuple[np.ndarray, list[float]]:
    out, ms = [], []
    for i in range(0, len(x), CHUNK):
        t0 = time.perf_counter()
        out.append(iso.feed(x[i : i + CHUNK]))
        ms.append(1000 * (time.perf_counter() - t0))
    out.append(iso.flush())
    return np.concatenate(out), ms


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def words(t: str) -> list[str]:
    return re.sub(r"[^a-z' ]", " ", t.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


LOCAL_ASR = (
    "import json, sys\n"
    "from parakeet_mlx import from_pretrained\n"
    "m = from_pretrained('mlx-community/parakeet-tdt-0.6b-v3')\n"
    "print(json.dumps([m.transcribe(f).text for f in sys.argv[1:]]))\n"
)


def transcribe_all(clips: list[np.ndarray]) -> tuple[list[str], str]:
    """Muse Voice Transcribe when the key works; otherwise parakeet on this Mac, and say so."""
    first = transcribe_file(pcm16(clips[0]))
    if first["status"] == 200:
        out = []
        for x in clips:
            out.append(str(json.loads(transcribe_file(pcm16(x))["body"]).get("transcript", "")))
        return out, "muse-voice-transcribe-1.0 (file endpoint)"
    reason = json.loads(first["body"]).get("error", {}).get("code", first["status"])
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, x in enumerate(clips):
            f = Path(d) / f"{i}.wav"
            f.write_bytes(wav(pcm16(x), SR))
            files.append(str(f))
        r = subprocess.run(
            ["uv", "run", "-q", "--no-project", "--with", "parakeet-mlx", "python", "-c",
             LOCAL_ASR, *files],
            check=True, capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )  # fmt: skip
    asr = f"parakeet-tdt-0.6b-v3 on this Mac (Transcribe refused: {reason})"
    return json.loads(r.stdout.strip().splitlines()[-1]), asr


def si_sdr(ref: np.ndarray, est: np.ndarray, max_lag: int = 1600) -> float:
    """Scale-invariant signal-to-distortion ratio in dB, at the best lag up to 0.1 s."""
    best = -np.inf
    for lag in range(0, max_lag + 1, 16):
        e = est[lag : lag + ref.size]
        r = ref[: e.size]
        a = np.dot(e, r) / max(1e-9, np.dot(r, r))
        t, n = a * r, e - a * r
        best = max(best, 10 * np.log10(max(1e-12, np.dot(t, t)) / max(1e-12, np.dot(n, n))))
    return round(float(best), 2)


def main() -> None:
    rng = np.random.default_rng(7)
    noises = {"fan": pink(SR * 30, rng), "tv": say(TV, "Daniel")}
    rows, clips, chunk_ms = [], [], []
    pad = np.zeros(int(0.4 * SR), np.float32)
    for q in QUESTIONS:
        speech = say(q)
        ref = np.concatenate([pad, speech, pad])
        rows.append({"question": q, "noise": "none", "snr_db": None, "isolated": False})
        clips.append(ref)
        for name, noise in noises.items():
            for snr in SNRS:
                noisy = mix(speech, noise, snr)
                clean, ms = isolate(noisy, Isolator())  # a fresh stream per clip
                chunk_ms += ms
                for isolated, x in ((False, noisy), (True, clean)):
                    rows.append({"question": q, "noise": name, "snr_db": snr,
                                 "isolated": isolated, "si_sdr_db": si_sdr(ref, x)})  # fmt: skip
                    clips.append(x)
    texts, asr = transcribe_all(clips)
    for r, t in zip(rows, texts, strict=True):
        r["heard"], r["wer"] = t, round(wer(r["question"], t), 3)

    def mean(key: str, **k) -> float:
        sel = [r[key] for r in rows if all(r[a] == b for a, b in k.items())]
        return round(statistics.mean(sel), 3)

    summary = {"clean": {"wer": mean("wer", noise="none")}}
    for name in noises:
        for snr in SNRS:
            summary[f"{name}@{snr:g}dB"] = {
                f"{key}_{tag}": mean(key, noise=name, snr_db=snr, isolated=iso)
                for key in ("wer", "si_sdr_db")
                for tag, iso in (("raw", False), ("isolated", True))
            }
    chunk_ms.sort()
    out = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": "facebookresearch/denoiser dns48 (Meta, open weights), streaming, CPU",
        "asr": asr,
        "chunk_ms_audio": 1000 * CHUNK / SR,
        "isolate_ms_per_chunk": {
            "median": round(statistics.median(chunk_ms), 2),
            "p95": round(chunk_ms[int(0.95 * len(chunk_ms))], 2),
            "max": round(chunk_ms[-1], 2),
            "n": len(chunk_ms),
        },
        "lookahead_ms": round(1000 * Isolator().latency_s, 1),
        "mean": summary,
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("asr", "isolate_ms_per_chunk", "lookahead_ms", "mean")},
                     indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
