from types import SimpleNamespace

from metasight.pins import Pin, PinStore
from metasight.review import MapReview, parse


class Fake:
    name = "muse-wonder"

    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200)

    def complete(self, system, user, max_tokens=120):
        self.calls.append(user)
        return self.reply


def pins():
    s = PinStore()
    s.pins[1] = Pin(1, "chair", "object", [0, 0, 3.0], ts=900, first_ts=100, seen=3, near=["desk"])
    s.pins[3] = Pin(3, "person", "object", [0, 0, 2.0], ts=995, first_ts=990, seen=1)
    s.pins[2] = Pin(2, "chair", "object", [0, 0, 1.0], ts=990, first_ts=980, seen=1, near=["door"])
    return s


REPLY = (
    '{"object": "chair", "matters": true, "in_the_way": true, '
    '"say": "The chair is now by the door."}'
)


def test_parse_skips_junk_and_keeps_objects():
    out = parse('ok\n{"object": "mug", "matters": false}\n{bad}\n{"say": "x"}')
    assert out == [{"object": "mug", "matters": False, "in_the_way": False, "say": ""}]


def test_review_reads_changes_since_last_and_prices_the_call():
    lang = Fake(REPLY)
    r = MapReview(lang, pins(), now=lambda: 1000.0)
    r.since = 970.0
    res = r.run()
    assert res["changes"] == 1 and "door" in lang.calls[0]
    assert res["items"][0]["in_the_way"] and r.in_the_way() == {"chair"}
    assert res["usd"] == (1000 * 4 + 200 * 20) / 1e6 == r.spent_usd


def test_review_skips_the_model_when_nothing_changed():
    lang = Fake(REPLY)
    r = MapReview(lang, pins(), now=lambda: 1000.0)
    r.since = 995.0
    res = r.run()
    assert res["changes"] == 0 and lang.calls == [] and r.in_the_way() == set()
