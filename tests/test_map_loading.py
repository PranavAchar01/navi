"""Right after a restart, before the home map loads, a question must not get "I haven't seen"."""

import asyncio
import queue
from types import SimpleNamespace as NS

from metasight import server
from metasight.server import Server


def fake(answered):
    async def json(ev, *_):
        pass

    async def set_activity(*_, **__):
        pass

    def answer_sync(text):
        answered.append(text)
        return {
            "type": "answer",
            "label": "keys",
            "text": "Your keys are by the sink.",
            "pin": None,
        }

    return NS(
        onboarding=NS(active=False),
        notes=NS(recording=False),
        hub=NS(json=json),
        known_labels=lambda: ["keys"],
        set_activity=set_activity,
        map_loaded=asyncio.Event(),
        _answer_sync=answer_sync,
        llm_pool=None,
        in_q=queue.Queue(),
        braille=None,
    )


def test_a_question_before_the_map_loads_says_it_is_loading(monkeypatch):
    monkeypatch.setattr(server, "MAP_WAIT_S", 0.05)
    answered = []

    async def run():
        srv = fake(answered)
        srv.loop = asyncio.get_running_loop()
        return await Server.answer(srv, "where are my keys")

    res = asyncio.run(run())
    assert res["text"] == server.MAP_LOADING and answered == []


def test_a_question_waits_for_a_map_that_loads_in_time(monkeypatch):
    monkeypatch.setattr(server, "MAP_WAIT_S", 2.0)
    answered = []

    async def run():
        srv = fake(answered)
        srv.loop = asyncio.get_running_loop()
        stale = srv.loop.create_future()  # started from a partial, on the empty bank
        stale.set_result({"type": "answer", "label": "keys", "text": "I haven't seen your keys."})
        srv.loop.call_later(0.05, srv.map_loaded.set)
        return await Server.answer(srv, "where are my keys", pending=stale)

    res = asyncio.run(run())
    assert res["text"] == "Your keys are by the sink." and answered == ["where are my keys"]
