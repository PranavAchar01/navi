from types import SimpleNamespace

from metasight.pins import Pin, PinStore
from metasight.review import MapReview, bank, grounded, nudges, parse, plan


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


def test_grounded_rejects_anything_not_in_the_bank():
    items = [{"object": "Chair", "say": ""}, {"object": "wallet", "say": "Grab your wallet."}]
    kept, invented = grounded(items, bank(pins(), 1000.0))
    assert [i["object"] for i in kept] == ["chair"] and invented == ["wallet"]  # the bank's name


def test_plan_and_nudges_never_name_an_invented_item():
    lang = Fake(
        '{"object": "chair", "say": "Your chair is by the door."}\n'
        '{"object": "wallet", "say": "Grab your wallet."}\n'
        '{"object": "person", "say": "Say bye."}'
    )
    b = bank(pins(), 1000.0)
    assert set(b) == {"chair"} and b["chair"]["near"] == ["door"]  # newest sighting, no people
    for res in (plan(lang, pins(), "get ready to leave", 1000.0), nudges(lang, pins(), 1000.0)):
        assert [i["object"] for i in res["items"]] == ["chair"]
        assert res["invented"] == ["wallet", "person"] and res["usd"] > 0


def tidy_pins():
    s = PinStore()
    s.pins[1] = Pin(1, "door", "object", [0, 0, 0], ts=900, first_ts=100, seen=9, score=0.9)
    s.pins[2] = Pin(2, "door", "object", [1.0, 0, 0], ts=950, first_ts=200, seen=3, score=0.8)
    s.pins[3] = Pin(3, "keys", "object", [3, 0, 3], ts=990, first_ts=990, seen=1, score=0.3)
    s.pins[4] = Pin(4, "desk", "object", [5, 0, 5], ts=990, first_ts=10, seen=20, score=0.9)
    return s


def test_tidy_checks_proposals_before_they_count():
    from metasight.review import apply_tidy, tidy, tidy_candidates

    s = tidy_pins()
    cands = tidy_candidates(s, radius=0.5)
    assert [c["kind"] for c in cands] == ["duplicate?", "ghost?"]
    reply = "\n".join(
        [
            '{"ids": [1, 2], "action": "merge", "why": "one door seen twice"}',
            '{"ids": [3], "action": "drop", "why": "a ghost"}',
            '{"ids": [4], "action": "drop", "why": "not a candidate"}',
            '{"ids": [1], "action": "drop", "why": "well seen, may not be dropped"}',
        ]
    )
    res = tidy(Fake(reply), s)
    assert [i["ids"] for i in res["items"]] == [[1, 2], [3]]
    assert len(res["rejected"]) == 2
    assert apply_tidy(s, res) == {"merged": 1, "dropped": 1}
    assert sorted(s.pins) == [1, 4] and s.pins[1].seen == 12
    assert s.pins[1].pos[0] == 0.25  # weighted by sightings


def test_training_review_is_graded_against_the_lab_rule():
    from metasight.sim_review import review

    summ = {"all": dict.fromkeys(
        ("success", "collisions", "min_clear_m", "time_to_goal_s", "flicker_per_min", "cost"), 1.0
    )}  # fmt: skip
    sim = {
        "arms": {
            "I0": {"hypothesis": "base", "base": None, "summary": summ},
            "I1": {"hypothesis": "a", "base": "I0", "summary": summ,
                   "paired_cost_vs_base": [-2.0, -3.0, -1.0]},
            "I2": {"hypothesis": "b", "base": "I0", "summary": summ,
                   "paired_cost_vs_base": [0.1, -0.5, 0.7]},
        }
    }  # fmt: skip
    reply = "\n".join(
        [
            '{"arm": "I1", "verdict": "keep", "why": "-2.0 with interval [-3.0, -1.0]"}',
            '{"arm": "I2", "verdict": "keep", "why": "cost fell by 7.5"}',
            '{"arm": "I9", "verdict": "keep", "why": "no such arm"}',
            '{"next": "try I1 with I2"}',
        ]
    )
    r = review(Fake(reply), sim)
    assert r["agree"] == "1/2"
    by = {a["arm"]: a for a in r["arms"]}
    assert set(by) == {"I1", "I2"}
    assert by["I2"]["rule"] == "unclear" and by["I2"]["unsupported"] == [7.5]
    assert by["I1"]["unsupported"] == []
    assert r["next"] == "try I1 with I2"
