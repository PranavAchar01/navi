"""Muse Voice Transcribe: the wearer's spoken questions, streamed to Meta's realtime endpoint.

One WebSocket to wss://api.meta.ai/v1/asr/realtime in ENDPOINTING mode: the model finds where each
utterance ends, so there is no button and no local end-of-speech timer. The session is opened
before anyone speaks (the handshake costs ~0.6 s) and kept alive with silence, because the
endpoint closes a stream that runs slower than real time after about 15 s. Watch-list labels go in
as keyword biasing so "glasses case" and "keys" come back spelled the way the map knows them.
A fresh session finalizes its first turn about 0.3 s slower than later ones (0.44 to 1.27 s against
0.21 to 0.38 s after a warm-up, bench 2026-10-29), so each session first hears a 1.3 s synthetic
"Okay, ready." (macOS `say`, warmup.pcm) and drops that turn.

The endpointer sometimes holds a finished turn for about 3 s before closing it (2 of 16 full
questions and 1 of 16 short commands, any length, 2026-10-29), and the API has no flush message.
So when the microphone has been quiet for STILL_S and the partial has stopped changing, the
partial is passed on as the final; the endpointer's late final for the same words is dropped.

Audio arrives as 16 kHz mono float32 from the same `audio` messages the conversation feature
reads; this module never touches the camera or the people tracker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("metasight.asr")

URL = "wss://api.meta.ai/v1/asr/realtime"
MODEL = "muse-voice-transcribe-1.0"
SR = 16000
FRAME_S = 0.08  # 80 ms frames, the docs' pacing
AHEAD_S = 0.2  # send at most this far ahead of the wall clock
GAP_S = 0.3  # no microphone audio for this long: pad with silence so the stream keeps time
ROTATE_S = 50 * 60  # sessions end at 60 minutes; start a fresh one before that
WARMUP = (Path(__file__).parent / "warmup.pcm").read_bytes()  # "Okay, ready.", 16 kHz PCM16
STILL_S = 0.6  # quiet microphone and an unchanged partial this long: the turn is over
VOICE_RMS = 0.01  # about -40 dBFS; quieter than this counts as silence
HANGING_S = 0.9  # a final ending on a word that needs another ("where are my") waits this long
HANGING = {"my", "the", "to", "a", "an", "um", "uh", "your", "of", "is", "are", "where's"}
WARM_MAX_S = 4.0  # the warm-up turn is done in ~1.5 s; past this, whatever comes is the wearer
WARM_GAP = b"\0" * int(SR * 0.5) * 2  # silence after the clip, so "ready" never runs into "where"
EARLY_S = 3.0  # microphone audio kept while the session opens (the handshake takes ~0.65 s)
_WARM_TEXT = re.compile(r"^\s*ok(ay)?([\s,.]*ready)?[\s.!?,]*", re.I)  # or a partial of it


@dataclass
class Heard:
    text: str
    final: bool
    turn: int | None
    spoken_end: float | None  # wall time the last word's audio reached this Mac (finals only)


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class AudioClock:
    """Maps the session's audio position (what `audioProcessedMs` counts) to the wall time that
    audio arrived, so a turn's end can be dated to when the wearer stopped talking."""

    def __init__(self) -> None:
        # (session audio position at the end of a sent chunk, wall time that audio arrived)
        self.marks: list[tuple[float, float]] = []

    def mark(self, audio_end_s: float, arrived: float) -> None:
        self.marks.append((audio_end_s, arrived))
        if len(self.marks) > 4000:
            del self.marks[:2000]

    def wall(self, audio_s: float) -> float | None:
        """Arrival time of the chunk holding position `audio_s`; None if it was never sent."""
        best = None
        for a, t in reversed(self.marks):
            if a + 1e-6 < audio_s:
                break
            best = t
        return best


class Transcriber:
    """Keeps one realtime session open and pushes `Heard` events to `on_heard`."""

    def __init__(
        self,
        key: str,
        on_heard: Callable[[Heard], Awaitable[None]],
        keywords: list[str] | None = None,
        connect=None,
        warmup: bytes = WARMUP,
    ):
        self.key, self.on_heard = key, on_heard
        self.warmup = warmup
        self.warm_until = 0.0  # while before this, the session's first turn is the warm-up
        self.keywords = keywords or []
        self._connect = connect  # tests inject a fake socket factory
        self.buf = bytearray()  # microphone audio not yet sent, PCM16 at 16 kHz
        self.buf_arrived = 0.0  # wall time of the newest pushed audio
        self.clock = AudioClock()
        self.connected = False
        self.wanted = False  # someone is connected who might speak
        self.turn_start: dict[int, float] = {}
        self.voiced_at = 0.0  # wall time of the newest microphone audio above VOICE_RMS
        self.partial: tuple[str, float] | None = None  # newest partial of the open turn, when
        self.forced: str | None = None  # a partial passed on as final; its late final is dropped
        self.held: str | None = None  # a final cut off mid-question, waiting for its last words
        self.held_at = 0.0
        self._release: asyncio.TimerHandle | None = None
        self.latest, self.partial_at = "", 0.0  # newest partial of any turn, and when
        self._task: asyncio.Task | None = None
        self.stats = {
            "sessions": 0, "turns": 0, "forced": 0, "handshake_ms": None, "warmup_ms": None,
        }  # fmt: skip

    # ------------------------------------------------------------ control
    def want(self, on: bool) -> None:
        """Open (or let close) the warm session; the server calls this as clients come and go."""
        self.wanted = on
        if on and (self._task is None or self._task.done()):
            self._task = asyncio.get_running_loop().create_task(self._run())

    def push(self, mono16k: np.ndarray) -> None:
        self.buf += float_to_pcm16(mono16k)
        self.buf_arrived = time.time()
        if len(mono16k) and float(np.sqrt(np.mean(mono16k**2))) > VOICE_RMS:
            self.voiced_at = self.buf_arrived
        if not self.connected:
            del self.buf[: max(0, len(self.buf) - int(SR * EARLY_S) * 2)]
            self.want(True)

    # ------------------------------------------------------------ session
    async def _open(self):
        if self._connect is not None:
            return await self._connect()
        import websockets

        return await websockets.connect(URL, open_timeout=10, max_size=2**20)

    def _handshake(self) -> str:
        hs = {
            "authorization": {"accessToken": f"Bearer {self.key}"},
            "audioEncoding": "PCM_16KHZ",
            "model": MODEL,
            "mode": "ENDPOINTING",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
            "languageBias": ["English"],
        }
        if self.keywords:
            hs["keywords"] = self.keywords
        return json.dumps(hs)

    async def _run(self) -> None:
        backoff = 0.5
        while self.wanted:
            try:
                await self._session()
                backoff = 0.5
            except Exception as e:  # network drop or server close: reopen, the wearer may speak
                log.warning("transcribe session ended: %s", e)
                self.connected = False
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 8.0)

    async def _session(self) -> None:
        t0 = time.perf_counter()
        ws = await self._open()
        try:
            await ws.send(self._handshake())
            ack = json.loads(await ws.recv())
            if "sessionId" not in ack:
                raise RuntimeError(f"handshake refused: {ack}")
            self.stats["handshake_ms"] = round(1000 * (time.perf_counter() - t0))
            self.stats["sessions"] += 1
            self.connected = True
            self.clock = AudioClock()
            # the first words can come while the session opens: they go in after the warm-up
            if self.warmup:
                self.buf[:0] = self.warmup + WARM_GAP
                self.buf_arrived = time.time()
                self.warm_until = time.time() + WARM_MAX_S
            log.info("transcribe session %s open", ack["sessionId"])
            self.partial = self.forced = None
            sender = asyncio.create_task(self._send(ws))
            still = asyncio.create_task(self._watch_still())
            try:
                await self._receive(ws)
            finally:
                sender.cancel()
                still.cancel()
        finally:
            self.connected = False
            await ws.close()

    async def _send(self, ws) -> None:
        """Paced at real time: microphone audio when there is some, silence across gaps."""
        # what is already buffered (the warm-up clip) is due now; paced behind the live microphone
        # it would hold every later turn back by its length
        start = time.time() - len(self.buf) / 2 / SR
        sent_s = 0.0
        frame = int(SR * FRAME_S) * 2
        while True:
            now = time.time()
            if now - start > ROTATE_S or not self.wanted:
                await ws.send(json.dumps({"type": "endStream"}))
                return
            target = now - start
            if self.buf and sent_s < target + AHEAD_S:
                n = min(len(self.buf), frame)
                n -= n % 2
                chunk = bytes(self.buf[:n])
                del self.buf[:n]
                sent_s += n / 2 / SR
                # the chunk's last sample arrived before whatever is still buffered behind it
                self.clock.mark(sent_s, self.buf_arrived - len(self.buf) / 2 / SR)
                await ws.send(chunk)
                continue
            if not self.buf and sent_s < target - GAP_S:
                sent_s += FRAME_S
                self.clock.mark(sent_s, now)
                await ws.send(b"\0" * frame)
                continue
            await asyncio.sleep(FRAME_S / 2)

    async def _watch_still(self) -> None:
        """Our own end of turn, for when the endpointer is slow to close one."""
        while True:
            await asyncio.sleep(0.05)
            if self.partial is None:
                continue
            text, at = self.partial
            now = time.time()
            if now - at >= STILL_S and now - self.voiced_at >= STILL_S:
                self.partial, self.forced = None, _words(text)
                self.stats["forced"] += 1
                self._final(clean(text), None, max(at, self.voiced_at))

    async def _receive(self, ws) -> None:
        async for msg in ws:
            if isinstance(msg, bytes):
                continue
            ev = json.loads(msg)
            kind = ev.get("type")
            if kind == "error":
                raise RuntimeError(ev.get("message"))
            if kind == "speechStart":
                self.turn_start[ev.get("turnId", 0)] = time.time()
            elif kind == "transcript" and not ev.get("final") and ev.get("transcript"):
                text = self._unwarm(ev["transcript"])
                if text:
                    self.latest, self.partial_at = text, time.time()
                    if self.partial is None or self.partial[0] != text:
                        self.partial = (text, time.time())
                    await self.on_heard(Heard(self._joined(text), False, None, None))
            elif kind == "speechComplete":
                text = self._unwarm(clean(ev.get("transcript", "")))
                self.partial = None
                forced, self.forced = self.forced, None
                if forced is not None and _words(text) == forced:
                    continue  # already answered from the partial
                if time.time() < self.warm_until:  # the warm-up turn, and anyone talking over it
                    opened = self.warm_until - WARM_MAX_S
                    self.warm_until = 0.0
                    self.stats["warmup_ms"] = round(1000 * (time.time() - opened))
                else:
                    self.stats["turns"] += 1
                end = self.clock.wall(ev.get("audioProcessedMs", 0) / 1000)
                if text:
                    self._final(text, ev.get("turnId"), end)

    def _final(self, text: str, turn: int | None, end: float) -> None:
        """Pass a final on; one that ends on "my", "the", "um"... waits HANGING_S for the rest of
        the question, which a pause of 0.7-0.9 s splits into a turn of its own."""
        text = self._joined(text)
        self.held = None
        if self._release:
            self._release.cancel()
        loop = asyncio.get_running_loop()
        if _words(text).rsplit(" ", 1)[-1] in HANGING:
            self.held, self.held_at = text, time.time()
            self._release = loop.call_later(HANGING_S, self._let_go, turn, end)
            return
        loop.create_task(self.on_heard(Heard(text, True, turn, end)))  # keep pacing meanwhile

    def _let_go(self, turn: int | None, end: float) -> None:
        """Pass the held start on after HANGING_S of quiet; words still coming join it instead.
        The endpointer can send the next turn's partials before this turn's final."""
        if self.held is None:
            return
        quiet = time.time() - max(self.voiced_at, self.partial_at)
        if quiet >= HANGING_S:
            text = self.held
            if self.partial_at > self.held_at:  # the rest was heard, its final is slow
                text = self._joined(self.latest)
                self.forced = _words(self.latest)
            self.held = None
            asyncio.get_running_loop().create_task(self.on_heard(Heard(text, True, turn, end)))
        else:
            self._release = asyncio.get_running_loop().call_later(
                HANGING_S - quiet, self._let_go, turn, end
            )

    def _joined(self, text: str) -> str:
        """The held start of the question in front of what came after it, said once."""
        if self.held is None or _words(text).startswith(_words(self.held)):
            return text
        head = text[:1].lower() if text[1:2].islower() else text[:1]  # "Keys" but not "I"
        return f"{self.held} {head}{text[1:]}"

    def _unwarm(self, text: str) -> str:
        """During the warm-up turn, the wearer's words without the clip's "Okay, ready" in front."""
        if time.time() >= self.warm_until:
            return text
        t = _WARM_TEXT.sub("", text, count=1)
        return t[:1].upper() + t[1:]


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def clean(text: str) -> str:
    """The model sometimes leaves stray punctuation after the question mark ("keys?:")."""
    t = text.strip()
    while t and t[-1] in ":;,":
        t = t[:-1].rstrip()
    return t
