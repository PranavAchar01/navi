"""Spoken questions: Muse Voice Transcribe in, an answer out, with the answer started early.

The transcript grows word by word while the wearer is still talking. As soon as a partial already
reads as "where is my <thing>", the answer for that thing starts (map lookup plus the Muse Spark
phrasing, about 0.8 s). When the model marks the end of the turn, that answer is usually done, so
the wait after the last word is the endpointing (~0.15 s) rather than endpointing plus the model.
If the finished sentence names a different thing, the early answer is dropped and a fresh one runs.
"""

from __future__ import annotations

import asyncio
import base64
import time

import numpy as np

from .audio_doa import SR, to_mono16k
from .providers.transcribe import Heard, Transcriber
from .query import is_question, needs_tools, parse_query

SPEC_FRESH_S = 5.0  # an early answer older than this is stale (the wearer may have moved)


def pcm_mono16k(m: dict) -> np.ndarray:
    """The `audio` message's int16 interleaved PCM as 16 kHz mono float32."""
    pcm = np.frombuffer(base64.b64decode(m["pcm"]), dtype="<i2").astype(np.float32) / 32768
    ch, sr = max(1, int(m.get("ch", 1))), int(m.get("sr", SR))
    x = pcm[: len(pcm) // ch * ch].reshape(-1, ch).mean(axis=1)
    return to_mono16k(x, sr)


class VoiceAsk:
    def __init__(self, srv, key: str, keywords: list[str], connect=None):
        self.srv = srv
        self.asr = Transcriber(key, self.on_heard, keywords, connect=connect)
        self.spec: tuple[str, asyncio.Future, float] | None = None  # (label, answer, started)
        self.last: dict = {}  # timings of the newest spoken question, for /api/state and logs

    def audio(self, m: dict) -> None:
        self.asr.push(pcm_mono16k(m))

    async def on_heard(self, h: Heard) -> None:
        await self.srv.hub.json(
            {"type": "transcript", "text": h.text, "final": h.final, "source": "transcribe"}
        )
        if not is_question(h.text):
            return
        label = parse_query(h.text, self.srv.known_labels())
        tools = needs_tools(h.text)  # the tool path cannot start from a fragment
        if not h.final:
            if label and not tools and (self.spec is None or self.spec[0] != label):
                fut = self.srv.loop.run_in_executor(
                    self.srv.llm_pool, self.srv._answer_sync, h.text
                )
                self.spec = (label, fut, time.time())
            return
        spec, self.spec = self.spec, None
        pending = None
        if spec and not tools and spec[0] == label and time.time() - spec[2] < SPEC_FRESH_S:
            pending = spec[1]
        res = await self.srv.answer(
            h.text, pending=pending, quiet_unknown=True, spoken_end=h.spoken_end
        )
        if res is not None:
            self.last = {"text": h.text, **res.get("voice", {})}
