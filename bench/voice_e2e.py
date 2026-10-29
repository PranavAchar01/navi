"""Spoken question to answer, end to end against the running server (scripts/service.sh).

A synthetic voice (macOS `say`) is streamed as `audio` messages at real time, followed by room
silence, as a live microphone would. Two pipelines are timed from the moment the last word has been
sent:
  streamed:   Muse Voice Transcribe on the server, endpointing, answer started from the partial.
  sequential: the old shape. Wait for the speaker to stop, transcribe the whole clip
              (POST /v1/asr/transcribe), then send the text as an `ask`.
Writes bench/results/voice-e2e.json.  Run: .venv/bin/python bench/voice_e2e.py [ws-url] [n]
"""

from __future__ import annotations

import asyncio
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from meta_models_probe import KEY, speech_pcm, wav  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8445/ws"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SR = 16000
CHUNK = 1024  # samples, what web/src/mic.ts sends (64 ms)


async def until(ws, pred, timeout=15.0):
    end = time.perf_counter() + timeout
    while True:
        m = await asyncio.wait_for(ws.recv(), max(0.01, end - time.perf_counter()))
        if isinstance(m, bytes):
            continue
        m = json.loads(m)
        if pred(m):
            return m, time.perf_counter()


def chunks(pcm: bytes):
    step = CHUNK * 2
    for i in range(0, len(pcm), step):
        yield pcm[i : i + step]


async def stream(ws, pcm: bytes, tail_s: float) -> float:
    """Send speech then silence at real time; returns perf time the last word was sent."""
    t = time.perf_counter()
    for c in chunks(pcm):
        await ws.send(
            json.dumps({"type": "audio", "sr": SR, "ch": 1, "pcm": base64.b64encode(c).decode()})
        )
        t += CHUNK / SR
        await asyncio.sleep(max(0, t - time.perf_counter()))
    spoken = time.perf_counter()

    async def tail():
        nonlocal t
        silence = b"\0" * (CHUNK * 2)
        for _ in range(int(tail_s * SR / CHUNK)):
            await ws.send(
                json.dumps(
                    {"type": "audio", "sr": SR, "ch": 1, "pcm": base64.b64encode(silence).decode()}
                )
            )
            t += CHUNK / SR
            await asyncio.sleep(max(0, t - time.perf_counter()))

    asyncio.create_task(tail())
    return spoken


async def streamed(question: str, pcm: bytes) -> dict:
    async with websockets.connect(URL, max_size=2**26) as ws:
        await asyncio.sleep(2.0)  # the server opens its Transcribe session on connect
        first_partial = None

        async def watch():
            nonlocal first_partial
            m, t = await until(ws, lambda m: m.get("type") == "transcript" and not m["final"])
            first_partial = t
            f, tf = await until(ws, lambda m: m.get("type") == "transcript" and m["final"])
            a, ta = await until(ws, lambda m: m.get("type") == "answer")
            return f, tf, a, ta

        w = asyncio.create_task(watch())
        t_start = time.perf_counter()
        spoken = await stream(ws, pcm, 3.0)
        f, tf, a, ta = await w
        return {
            "heard": f["text"],
            "answer": a["text"],
            "label": a.get("label"),
            "language": a.get("language"),
            "early": a.get("voice", {}).get("early"),
            "first_partial_from_start_ms": round(1000 * (first_partial - t_start)),
            "final_transcript_ms": round(1000 * (tf - spoken)),
            "answer_ms": round(1000 * (ta - spoken)),
            "server_speech_end_to_answer_ms": a.get("voice", {}).get("speech_end_to_answer_ms"),
        }


async def sequential(question: str, pcm: bytes) -> dict:
    async with websockets.connect(URL, max_size=2**26) as ws:
        await asyncio.sleep(0.5)
        spoken = time.perf_counter()
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.post(
                "https://api.meta.ai/v1/asr/transcribe",
                headers={"Authorization": f"Bearer {KEY}"},
                files={
                    "request": (
                        None,
                        json.dumps(
                            {
                                "mode": "PUSH_TO_TALK",
                                "model": "muse-voice-transcribe-1.0",
                                "audioEncoding": "WAV",
                            }
                        ),
                        "application/json",
                    ),
                    "audio": ("q.wav", wav(pcm, SR), "audio/wav"),
                },
            )
        text = r.json()["transcript"].rstrip(":;, ")
        t_text = time.perf_counter()
        await ws.send(json.dumps({"type": "ask", "text": text}))
        a, ta = await until(ws, lambda m: m.get("type") == "answer")
        return {
            "heard": text,
            "answer": a["text"],
            "language": a.get("language"),
            "transcript_ms": round(1000 * (t_text - spoken)),
            "answer_ms": round(1000 * (ta - spoken)),
        }


async def main() -> None:
    state = httpx.get(URL.replace("ws://", "http://").replace("/ws", "/api/state")).json()
    labels = sorted({p["label"] for p in state["pins"]})
    asks = [f"where is my {lab}" for lab in labels[:N]] or ["where are my keys"]
    out = {"at": time.strftime("%Y-%m-%d %H:%M %Z"), "streamed": [], "sequential": []}
    for q in asks:
        pcm = speech_pcm(q, SR)
        out["streamed"].append(await streamed(q, pcm))
        print("streamed", out["streamed"][-1], flush=True)
        out["sequential"].append(await sequential(q, pcm))
        print("sequential", out["sequential"][-1], flush=True)
    for k in ("streamed", "sequential"):
        v = [r["answer_ms"] for r in out[k]]
        out[f"{k}_median_answer_ms"] = round(statistics.median(v))
    print({k: v for k, v in out.items() if k.endswith("_ms")})
    (ROOT / "bench/results/voice-e2e.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
