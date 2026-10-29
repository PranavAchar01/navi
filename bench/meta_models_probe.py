"""One tiny live call per Meta Model API model, timed. Writes bench/results/meta-models-probe.json.

Run: .venv/bin/python bench/meta_models_probe.py [--image]   (--image spends $0.01 on Muse Image)
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import websockets
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metasight.config import META_BASE_URL, secret  # noqa: E402

KEY = secret("META_API_KEY", "MODEL_API_KEY")
OUT = ROOT / "bench" / "results" / "meta-models-probe.json"
QUESTION = "where are my keys"
FACTS = (
    "object: keys\nseen: 2 minutes ago\nnear: sofa\ndistance: about 3 metres\n"
    "direction: behind you to the left\nheight: below eye level"
)
SYSTEM = "Answer in ONE short spoken sentence using only the facts given."


def chat_stream(model: str, extra: dict) -> dict:
    c = OpenAI(base_url=META_BASE_URL, api_key=KEY, timeout=60)
    t0 = time.perf_counter()
    first, text = None, ""
    s = c.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": FACTS}],
        max_tokens=400,
        stream=True,
        extra_body=extra,
    )
    for ch in s:
        d = ch.choices[0].delta.content if ch.choices else None
        if d:
            first = first or time.perf_counter()
            text += d
    t1 = time.perf_counter()
    return {
        "first_token_ms": round(1000 * (first - t0)) if first else None,
        "total_ms": round(1000 * (t1 - t0)),
        "text": text.strip(),
    }


def speech_pcm(text: str, rate: int = 24000) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        aiff = Path(d) / "q.aiff"
        subprocess.run(["say", "-o", str(aiff), text], check=True)
        return subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(aiff),
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                str(rate),
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout


def wav(pcm: bytes, rate: int = 24000) -> bytes:
    import io
    import wave

    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return b.getvalue()


def transcribe_file(pcm: bytes) -> dict:
    req = {"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    t0 = time.perf_counter()
    r = httpx.post(
        f"{META_BASE_URL}/asr/transcribe",
        headers={"Authorization": f"Bearer {KEY}"},
        files={
            "request": (None, json.dumps(req), "application/json"),
            "audio": ("q.wav", wav(pcm), "audio/wav"),
        },
        timeout=60,
    )
    ms = round(1000 * (time.perf_counter() - t0))
    return {"status": r.status_code, "total_ms": ms, "body": r.text[:400]}


async def transcribe_realtime(pcm: bytes, rate: int = 24000) -> dict:
    """Paced like a live mic (80 ms frames). Times are from the last audio frame sent."""
    out: dict = {"events": []}
    t_connect = time.perf_counter()
    async with websockets.connect("wss://api.meta.ai/v1/asr/realtime", open_timeout=30) as ws:
        await ws.send(
            json.dumps(
                {
                    "authorization": {"accessToken": f"Bearer {KEY}"},
                    "audioEncoding": "PCM_24KHZ",
                    "model": "muse-voice-transcribe-1.0",
                    "mode": "PUSH_TO_TALK",
                    "partialMode": "CUMULATIVE",
                    "keywords": ["keys", "wallet", "glasses case"],
                }
            )
        )
        hs = json.loads(await ws.recv())
        out["handshake_ms"] = round(1000 * (time.perf_counter() - t_connect))
        out["handshake"] = hs
        step = rate * 2 * 80 // 1000
        t_audio_end = 0.0
        first_partial = None

        async def send():
            nonlocal t_audio_end
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i : i + step])
                await asyncio.sleep(0.08)
            t_audio_end = time.perf_counter()
            await ws.send(json.dumps({"type": "endStream"}))

        sender = asyncio.create_task(send())
        t_start = time.perf_counter()
        async for msg in ws:
            if isinstance(msg, bytes):
                continue
            ev = json.loads(msg)
            now = time.perf_counter()
            out["events"].append(ev.get("type"))
            if ev.get("type") == "transcript":
                if first_partial is None:
                    first_partial = now - t_start
                if ev.get("final"):
                    out["final"] = ev["transcript"]
                    out["final_after_audio_end_ms"] = round(1000 * (now - t_audio_end))
            if ev.get("type") == "error":
                out["error"] = ev
        await sender
        out["first_partial_after_audio_start_ms"] = (
            round(1000 * first_partial) if first_partial else None
        )
        out["audio_ms"] = round(1000 * len(pcm) / (rate * 2))
    return out


def image_gen() -> dict:
    c = OpenAI(base_url=META_BASE_URL, api_key=KEY, timeout=120)
    t0 = time.perf_counter()
    r = c.images.generate(model="muse-image-1.0", prompt="a simple flat icon of a key", n=1)
    ms = round(1000 * (time.perf_counter() - t0))
    d = r.data[0]
    return {"total_ms": ms, "has_b64": bool(d.b64_json), "url": bool(d.url)}


def sam() -> dict:
    import cv2
    import numpy as np

    img = np.full((360, 640, 3), 230, np.uint8)
    cv2.rectangle(img, (240, 120), (400, 260), (40, 40, 160), -1)
    ok, jpg = cv2.imencode(".jpg", img)
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    c = OpenAI(base_url=META_BASE_URL, api_key=KEY, timeout=60)
    t0 = time.perf_counter()
    r = c.responses.create(
        model="sam-3.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": url},
                    {"type": "input_text", "text": "red square"},
                ],
            }
        ],
    )
    return {"total_ms": round(1000 * (time.perf_counter() - t0)), "out": str(r.output)[:200]}


def run(name: str, fn, res: dict) -> None:
    try:
        res[name] = fn()
    except Exception as e:  # a probe records the failure and moves on
        res[name] = {"error": f"{type(e).__name__}: {e}"[:400]}
    print(name, json.dumps(res[name])[:300], flush=True)


def main() -> None:
    res: dict = {"at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    for m, extra in [
        ("muse-spark-1.3", {"reasoning_effort": "minimal"}),
        ("muse-spark-1.3", {"reasoning_effort": "minimal"}),  # warm connection
        ("muse-spark-1.2", {"reasoning_effort": "minimal"}),
        ("muse-spark-1.1", {}),
        ("muse-spark-1.3-contributor", {"reasoning_effort": "minimal"}),
        ("muse-wonder-1.0", {"reasoning_effort": "low"}),
    ]:
        key = m + ("#warm" if m in res else "")
        run(key, lambda m=m, extra=extra: chat_stream(m, extra), res)
    pcm = speech_pcm(QUESTION)
    run("transcribe-file", lambda: transcribe_file(pcm), res)
    run("transcribe-realtime", lambda: asyncio.run(transcribe_realtime(pcm)), res)
    run("sam-3.1", sam, res)
    if "--image" in sys.argv:
        run("muse-image-1.0", image_gen, res)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
