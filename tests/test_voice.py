import asyncio
import json
import time
from types import SimpleNamespace as NS

import numpy as np

from metasight.providers.transcribe import AudioClock, Transcriber, clean
from metasight.voice import VoiceAsk


class FakeWS:
    """Speaks the realtime protocol: acks the handshake, then replays scripted events (a number
    in the script is a pause, in seconds, before the next event)."""

    def __init__(self, events):
        self.sent: list = []
        self.events = events
        self.q: asyncio.Queue = asyncio.Queue()

    async def send(self, m):
        self.sent.append(m)
        if isinstance(m, str) and "authorization" in m:
            await self.q.put(json.dumps({"sessionId": "s1"}))
            asyncio.get_running_loop().create_task(self._play())

    async def _play(self):
        for e in self.events:
            if isinstance(e, float):
                await asyncio.sleep(e)
            else:
                await self.q.put(json.dumps(e))

    async def recv(self):
        return await self.q.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.q.get()

    async def close(self):
        pass


def test_clean_strips_trailing_marks():
    assert clean("Where are my keys?:") == "Where are my keys?"
    assert clean("  hello ") == "hello"


def test_clock_dates_audio_by_arrival():
    c = AudioClock()
    c.mark(0.08, 10.0)
    c.mark(0.16, 10.5)
    c.mark(0.24, 11.0)
    assert c.wall(0.1) == 10.5  # 0.1 s sits in the chunk ending at 0.16
    assert c.wall(0.24) == 11.0
    assert c.wall(5.0) is None


def test_handshake_asks_for_endpointing_and_keywords():
    t = Transcriber("k", None, ["keys", "glasses case"])
    hs = json.loads(t._handshake())
    assert hs["mode"] == "ENDPOINTING" and hs["audioEncoding"] == "PCM_16KHZ"
    assert hs["keywords"] == ["keys", "glasses case"]
    assert hs["authorization"]["accessToken"] == "Bearer k"


def run_session(events, push_audio=True, warmup=b"", wait=0.4):
    heard = []

    async def on_heard(h):
        heard.append(h)

    async def main():
        ws = FakeWS(events)

        async def connect():
            return ws

        t = Transcriber("k", on_heard, ["keys"], connect=connect, warmup=warmup)
        t.wanted = True
        task = asyncio.create_task(t._session())
        await asyncio.sleep(0.05)
        if push_audio:
            t.push(np.zeros(1600, np.float32))
        await asyncio.sleep(wait)
        t.wanted = False
        await asyncio.sleep(0.1)
        task.cancel()
        return ws, t

    ws, t = asyncio.run(main())
    return heard, ws, t


def test_partials_and_turn_end_reach_the_listener():
    heard, ws, t = run_session(
        [
            {"type": "speechStart", "turnId": 0, "audioProcessedMs": 0},
            {"type": "transcript", "transcript": "where are", "final": False},
            {"type": "transcript", "transcript": "where are my keys", "final": False},
            {"type": "speechEnd", "turnId": 0, "audioProcessedMs": 100},
            {"type": "speechComplete", "turnId": 0, "transcript": "Where are my keys?:"},
        ]
    )
    assert [h.text for h in heard] == ["where are", "where are my keys", "Where are my keys?"]
    assert [h.final for h in heard] == [False, False, True]
    assert t.stats["turns"] == 1


def test_stream_is_paced_and_padded_with_silence():
    _, ws, _ = run_session([], push_audio=False)
    frames = [m for m in ws.sent if isinstance(m, bytes)]
    # 0.45 s of wall time, 80 ms frames, silence once the 0.3 s gap allowance is used up
    assert 1 <= len(frames) <= 3
    assert all(f == b"\0" * len(f) for f in frames)
    assert json.loads(ws.sent[-1]) == {"type": "endStream"}


def test_each_session_opens_with_a_warm_up_turn_that_is_dropped():
    clip = b"\x01\x02" * 1280  # 0.16 s, two frames
    heard, ws, t = run_session(
        [
            {"type": "transcript", "transcript": "okay", "final": False},
            {"type": "speechComplete", "turnId": 0, "transcript": "Okay, ready."},
            {"type": "transcript", "transcript": "where are", "final": False},
            {"type": "speechComplete", "turnId": 1, "transcript": "Where are my keys?"},
        ],
        push_audio=False,
        warmup=clip,
    )
    frames = b"".join(m for m in ws.sent if isinstance(m, bytes))
    assert frames.startswith(clip)
    assert [h.text for h in heard] == ["where are", "Where are my keys?"]
    assert t.stats["turns"] == 1 and t.stats["warmup_ms"] is not None


def test_a_question_spoken_over_the_warm_up_still_gets_through():
    heard, _, _ = run_session(
        [
            {"type": "transcript", "transcript": "Okay, ready, where are", "final": False},
            {
                "type": "speechComplete",
                "turnId": 0,
                "transcript": "Okay, ready, where are my keys?",
            },
        ],
        warmup=b"\x01\x02" * 1280,
    )
    assert [h.text for h in heard] == ["Where are", "Where are my keys?"]


class FakeServer:
    def __init__(self):
        self.calls = []
        self.sent = []
        self.hub = NS(json=self._json)
        self.llm_pool = None
        self.loop = None

    async def _json(self, m):
        self.sent.append(m)

    def known_labels(self):
        return ["keys", "wallet"]

    def _answer_sync(self, text):
        self.calls.append(text)
        time.sleep(0.05)
        return {"type": "answer", "label": "keys" if "key" in text else None, "text": "ok"}

    async def answer(self, text, pending=None, quiet_unknown=False, spoken_end=None):
        res = await pending if pending else self._answer_sync(text)
        if quiet_unknown and res["label"] is None:
            return None
        res["voice"] = {"early": pending is not None}
        return res


def heard(text, final):
    from metasight.providers.transcribe import Heard

    return Heard(text, final, 0 if final else None, time.time() if final else None)


def test_answer_starts_from_the_partial_and_is_reused():
    async def main():
        srv = FakeServer()
        srv.loop = asyncio.get_running_loop()
        v = VoiceAsk(srv, "k", [])
        await v.on_heard(heard("where are", False))
        await v.on_heard(heard("where are my keys", False))
        await v.on_heard(heard("where are my keys and", False))
        await v.on_heard(heard("Where are my keys?", True))
        return srv, v

    srv, v = asyncio.run(main())
    assert srv.calls == ["where are my keys"]  # one early answer, reused at the turn's end
    assert v.last["early"] is True
    assert [m["final"] for m in srv.sent] == [False, False, False, True]


def test_overheard_speech_gets_no_reply():
    async def main():
        srv = FakeServer()
        srv.loop = asyncio.get_running_loop()
        v = VoiceAsk(srv, "k", [])
        await v.on_heard(heard("nice weather today", True))
        await v.on_heard(heard("where is the bathroom?", True))
        return srv, v

    srv, v = asyncio.run(main())
    assert v.last == {}
    assert srv.calls == ["where is the bathroom?"]  # asked, but no known thing: stays quiet


def test_changed_question_drops_the_early_answer():
    async def main():
        srv = FakeServer()
        srv.loop = asyncio.get_running_loop()
        v = VoiceAsk(srv, "k", [])
        await v.on_heard(heard("where is my wallet", False))
        await v.on_heard(heard("Where are my keys?", True))
        return srv, v

    srv, v = asyncio.run(main())
    assert v.last["early"] is False


def test_glimmer_prompt_skips_the_reasoning_turn():
    from metasight.providers.glimmer import prompt

    p = prompt("S", "U")
    assert p.endswith("<|start|>assistant to=user<|message|>")
    assert '# Valid recipients: "user".' in p and "to=self" not in p


def test_fallback_uses_glimmer_when_spark_fails():
    from metasight.providers.glimmer import FallbackLanguage

    class Down:
        name = "muse-spark"

        def complete(self, *a, **k):
            raise ConnectionError("offline")

    class Local:
        name = "muse-glimmer"

        def complete(self, *a, **k):
            return "Your keys are by the sofa."

    f = FallbackLanguage(Down(), Local())
    assert f.complete("s", "u") == "Your keys are by the sofa." and f.name == "muse-glimmer"


def test_offline_glimmer_wraps_spark(monkeypatch):
    from metasight.config import Settings
    from metasight.providers import language
    from metasight.providers.glimmer import FallbackLanguage

    monkeypatch.setattr(language, "secret", lambda *names: "k")
    monkeypatch.setattr(language.OpenAICompatLanguage, "keep_warm", lambda self: None)
    lm = language.make_language(Settings(language="meta", offline="glimmer"))
    assert isinstance(lm, FallbackLanguage) and lm.primary._client.timeout == 4.0
    assert lm.backup.name == "muse-glimmer"


def test_people_and_change_questions_go_to_the_tools():
    from metasight.query import needs_tools

    assert needs_tools("Who is talking?") and needs_tools("what changed since this morning")
    assert needs_tools("is anyone in front of me")
    assert not needs_tools("where are my keys") and not needs_tools("where did I put my wallet")


def test_tool_question_is_not_answered_from_an_early_guess():
    async def main():
        srv = FakeServer()
        srv.loop = asyncio.get_running_loop()
        v = VoiceAsk(srv, "k", [])
        await v.on_heard(heard("where are my keys", False))
        await v.on_heard(heard("Where are my keys, did someone move them?", True))
        return v

    assert asyncio.run(main()).last["early"] is False


def test_a_stalled_endpointer_is_answered_from_the_still_partial(monkeypatch):
    import metasight.providers.transcribe as tr

    monkeypatch.setattr(tr, "STILL_S", 0.1)
    heard, _, t = run_session(
        [
            {"type": "transcript", "transcript": "keys", "final": False},
            0.3,  # the endpointer holds the turn; the microphone is quiet
            {"type": "speechComplete", "turnId": 0, "transcript": "Keys?"},
        ],
        wait=0.5,
    )
    assert [(h.text, h.final) for h in heard] == [("keys", False), ("keys", True)]
    assert t.stats["forced"] == 1


def test_words_after_a_forced_final_still_get_through(monkeypatch):
    import metasight.providers.transcribe as tr

    monkeypatch.setattr(tr, "STILL_S", 0.1)
    heard, _, _ = run_session(
        [
            {"type": "transcript", "transcript": "where did I leave", "final": False},
            0.3,
            {"type": "speechComplete", "turnId": 0, "transcript": "Where did I leave my phone?"},
        ],
        wait=0.5,
    )
    finals = [h.text for h in heard if h.final]
    assert finals == ["where did I leave", "Where did I leave my phone?"]


def test_a_question_split_at_a_pause_after_my_is_joined(monkeypatch):
    import metasight.providers.transcribe as tr

    monkeypatch.setattr(tr, "HANGING_S", 0.3)
    heard, _, _ = run_session(
        [
            {"type": "speechComplete", "turnId": 0, "transcript": "Where are my"},
            0.1,  # the wearer pauses, then finishes the question
            {"type": "transcript", "transcript": "keys", "final": False},
            {"type": "speechComplete", "turnId": 1, "transcript": "Keys?"},
        ],
        wait=0.6,
    )
    assert [h.text for h in heard] == ["Where are my keys", "Where are my keys?"]
    assert [h.final for h in heard] == [False, True]


def test_a_dangling_final_is_let_go_when_nothing_follows(monkeypatch):
    import metasight.providers.transcribe as tr

    monkeypatch.setattr(tr, "HANGING_S", 0.2)
    heard, _, _ = run_session(
        [{"type": "speechComplete", "turnId": 0, "transcript": "Take me to the"}], wait=0.5
    )
    assert [(h.text, h.final) for h in heard] == [("Take me to the", True)]


def test_a_forced_cut_at_my_is_not_said_twice(monkeypatch):
    import metasight.providers.transcribe as tr

    monkeypatch.setattr(tr, "STILL_S", 0.1)
    monkeypatch.setattr(tr, "HANGING_S", 0.3)
    heard, _, _ = run_session(
        [
            {"type": "transcript", "transcript": "where are my", "final": False},
            0.2,
            {"type": "transcript", "transcript": "where are my keys", "final": False},
            {"type": "speechComplete", "turnId": 0, "transcript": "Where are my keys?"},
        ],
        wait=0.6,
    )
    assert [h.text for h in heard if h.final] == ["Where are my keys?"]


def test_the_rest_of_a_split_question_is_taken_from_its_partial_when_its_final_is_slow(
    monkeypatch,
):
    import metasight.providers.transcribe as tr

    monkeypatch.setattr(tr, "HANGING_S", 0.2)
    heard, _, _ = run_session(
        [
            {"type": "speechComplete", "turnId": 0, "transcript": "Where are my"},
            0.05,
            {"type": "transcript", "transcript": "keys", "final": False},
            0.5,  # the endpointer holds the second turn
            {"type": "speechComplete", "turnId": 1, "transcript": "Keys?"},
        ],
        push_audio=False,
        wait=0.3,
    )
    assert [h.text for h in heard if h.final] == ["Where are my keys"]
