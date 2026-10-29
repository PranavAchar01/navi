import json
from types import SimpleNamespace as NS

from metasight.pins import Pin, PinStore
from metasight.providers.tools import Tools, agent_answer, side


class Head:
    def to(self, pos):
        return (abs(pos[2]), 30.0)


def store():
    s = PinStore()
    s.pins[1] = Pin(1, "keys", "object", [0, 0, 3.0], ts=900, first_ts=100, seen=3, near=["table"])
    s.pins[2] = Pin(2, "keys", "object", [0, 0, 1.0], ts=990, first_ts=980, seen=1, near=["sofa"])
    s.pins[3] = Pin(3, "mug", "object", [0, 0, 2.0], ts=995, first_ts=990, seen=2, near=["desk"])
    return s


def tools(ev=None):
    return Tools(store(), Head(), lambda: ev, now=lambda: 1000)


def test_where_is_uses_heading_and_age():
    r = tools().where_is("mug")
    assert r["distance_m"] == 2.0 and r["direction"] == "right and ahead" and r["seen_s_ago"] == 5
    assert tools().where_is("wallet") == {"found": False}


def test_what_changed_reports_moved_and_new():
    ch = {c["object"]: c for c in tools().what_changed_since(5)["changes"]}
    assert ch["keys"]["change"] == "moved" and ch["keys"]["was_near"] == ["table"]
    assert ch["mug"]["change"] == "new"
    assert tools().what_changed_since(0.01)["changes"] == []


def test_speaker_direction_in_view_offscreen_and_silent():
    ev = {
        "voice": True,
        "speaker": 1,
        "doa": None,
        "evidence": "lip",
        "tracks": [{"id": 1, "bearing": -40.0, "dist": 2.0, "lost": False, "speaking": 0.8}],
    }
    assert tools(ev).who_is_speaking_direction()["direction"] == "left and ahead"
    off = {"voice": True, "speaker": None, "doa": 100.0, "tracks": [], "evidence": "none"}
    r = tools(off).who_is_speaking_direction()
    assert r["in_view"] is False and r["direction"] == "right"
    assert tools({"voice": False, "tracks": []}).who_is_speaking_direction() == {"speaking": False}
    assert tools(None).people_in_view()["count"] == 0


def test_side_words():
    assert side(0) == "ahead" and side(170) == "behind" and side(-90) == "left"


def test_agent_loop_runs_tool_then_answers():
    calls = []

    def create(**kw):
        calls.append(kw["messages"][-1]["role"])
        if len(calls) == 1:
            c = NS(id="c1", function=NS(name="where_is", arguments=json.dumps({"object": "mug"})))
            return NS(choices=[NS(message=NS(content="", tool_calls=[c]))])
        tool_msg = kw["messages"][-1]["content"]
        return NS(
            choices=[
                NS(
                    message=NS(
                        content="Your mug is " + json.loads(tool_msg)["direction"], tool_calls=None
                    )
                )
            ]
        )

    lang = NS(model="m", extra={}, _client=NS(chat=NS(completions=NS(create=create))))
    assert agent_answer(lang, "where is my mug", tools()) == "Your mug is right and ahead"


def test_empty_change_list_is_stated_and_the_prompt_says_nothing_changed():
    from metasight.providers.tools import SYSTEM

    assert "nothing changed" in SYSTEM
    out = tools().what_changed_since(0.01)
    assert out["changes"] == [] and out["window_min"] == 0.01


def test_a_plan_is_spoken_first_step_first_with_no_second_spark_call():
    from metasight.review import plan

    calls = []

    def create(**kw):
        calls.append(kw)
        c = NS(id="c1", function=NS(name="plan", arguments=json.dumps({"goal": "leave"})))
        return NS(choices=[NS(message=NS(content="", tool_calls=[c]))])

    class Wonder:
        name = "muse-wonder"
        usage = None

        def complete(self, system, user, max_tokens=120):
            return '{"object": "Keys", "say": "Grab your keys."}\n{"object": "wallet", "say": "x"}'

    s = store()
    t = Tools(
        s, Head(), lambda: None, now=lambda: 1000, planner=lambda g: plan(Wonder(), s, g, 1000)
    )
    lang = NS(model="m", extra={}, _client=NS(chat=NS(completions=NS(create=create))))
    assert agent_answer(lang, "get ready to leave", t) == "Grab your keys."
    assert len(calls) == 1 and [i["object"] for i in t.steps] == ["keys"]  # no wallet: not seen
    assert t.steps[0]["near"] == ["table"]


def test_errands_take_the_tool_path():
    from metasight.query import is_question, needs_tools

    assert needs_tools("help me get ready to leave") and is_question("Get ready to leave.")
    assert not needs_tools("where did I leave my keys")
