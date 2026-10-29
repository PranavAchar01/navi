"""Typed question to answer on the live server, for the two answer paths.

  single: "where is my X". The pin's facts go to Muse Spark in one call.
  tools:  people, voices, changes. Muse Spark calls tools (providers/tools.py), 1 to 3 calls.
The answer is one sentence sent whole, so ask-to-first-word equals ask-to-answer here.
Writes bench/results/answer-paths.json.  Run: .venv/bin/python bench/answer_paths.py [ws-url]
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8445/ws"
ASKS = {
    "single": ["where is my tv", "where is my backpack", "where is my basket", "where is my bed"],
    "tools": [
        "what changed in the last ten minutes",
        "is anyone talking",
        "how many people are in front of me",
        "did someone move my backpack",
    ],
}


async def ask(ws, text: str) -> dict:
    t = time.perf_counter()
    await ws.send(json.dumps({"type": "ask", "text": text}))
    while True:
        m = await asyncio.wait_for(ws.recv(), 20)
        if isinstance(m, str) and (a := json.loads(m)).get("type") == "answer":
            return {
                "ask": text,
                "answer": a["text"],
                "language": a.get("language"),
                "tools": bool(a.get("tools")),
                "client_ms": round(1000 * (time.perf_counter() - t)),
                "server_ms": a.get("ms"),
            }


async def main() -> None:
    out: dict = {"at": time.strftime("%Y-%m-%d %H:%M %Z")}
    async with websockets.connect(URL, max_size=2**26) as ws:
        await asyncio.sleep(1.0)
        for path, asks in ASKS.items():
            out[path] = []
            for _ in range(2):
                for q in asks:
                    out[path].append(await ask(ws, q))
                    print(path, out[path][-1], flush=True)
            out[f"{path}_median_ms"] = round(statistics.median(r["client_ms"] for r in out[path]))
    print({k: v for k, v in out.items() if k.endswith("_ms")})
    (ROOT / "bench/results/answer-paths.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
