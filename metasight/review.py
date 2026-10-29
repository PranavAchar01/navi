"""The deep loop: every few minutes a slow model reads what changed on the map and says which
changes matter. Nobody waits on it, so it can be the slow, careful model (Muse Wonder) or the
open one on this Mac (Muse Glimmer).

Input is the change log the pins already keep (`Tools.what_changed_since`: moved or new
objects, with what they were near). Output, per change: does it matter, is it something to walk
into, and one short sentence the wearer could be told. Nothing is called when nothing changed.

The same model also makes plans ("get ready to leave" -> keys, then the door) and nudges (a
thing not seen for a long time). Both only ever name things in the memory bank: `grounded`
drops anything else before it reaches the wearer.
"""

from __future__ import annotations

import json
import re
import time

from .providers.tools import Tools

PRICE = {  # $ per 1M tokens, https://dev.meta.ai/docs/pricing-rate-limits
    "muse-wonder": (4.0, 20.0),
    "muse-spark": (1.25, 4.25),
}

SYSTEM = (
    "You review changes to a blind person's home map. For each change, reply with one JSON "
    "object per line, nothing else: "
    '{"object": str, "matters": bool, "in_the_way": bool, "say": str}. '
    "matters: the person would want to know (a thing they use moved, something new where "
    "they walk). in_the_way: it could now be walked into (furniture, a bag, a box on a floor "
    "path). say: one short sentence, or empty when it does not matter."
)


PLAN_SYSTEM = (
    "You plan a short errand for a blind person at home. You get the request and the things "
    "in their home memory, with what each was last seen near. Reply with one JSON object per "
    'step, in order, nothing else: {"object": str, "say": str}. object must be copied exactly '
    "from the memory list; if something the request needs is not in it, leave it out. say: one "
    "short sentence for that step."
)

NUDGE_SYSTEM = (
    "You help a blind person keep track of their home. You get the things in their home memory "
    "and how many hours since each was last seen. Pick at most 3 worth a glance next time they "
    "walk past (things they carry or use, not walls or doors). Reply with one JSON object per "
    'line, nothing else: {"object": str, "say": str}. object must be copied exactly from the '
    "list. say: one short sentence."
)


def _objects(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d.get("object"), str):
            out.append(d)
    return out


def parse(text: str) -> list[dict]:
    return [
        {
            "object": d["object"],
            "matters": bool(d.get("matters")),
            "in_the_way": bool(d.get("in_the_way")),
            "say": str(d.get("say") or ""),
        }
        for d in _objects(text)
    ]


def bank(pins, now: float) -> dict[str, dict]:
    """Every object label in memory, newest sighting first: where it was and how long ago."""
    out: dict[str, dict] = {}
    for p in sorted(pins.pins.values(), key=lambda p: -p.ts):
        if p.kind == "object" and p.label != "person" and p.label not in out:
            out[p.label] = {"near": p.near, "hours": round((now - p.ts) / 3600, 1)}
    return out


def grounded(items: list[dict], labels) -> tuple[list[dict], list[str]]:
    """Keep only items naming a thing in the bank; return the rest by name, to count them."""
    known = {x.lower() for x in labels}
    kept = [i for i in items if i["object"].lower() in known]
    return kept, [i["object"] for i in items if i["object"].lower() not in known]


def _usd(lang) -> float | None:
    usage = getattr(lang, "usage", None)
    if usage is None or lang.name not in PRICE:
        return None
    pin, pout = PRICE[lang.name]
    return (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6


def _ask(lang, kind: str, system: str, user: str, labels, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    raw = lang.complete(system, user, max_tokens=max_tokens)
    items = [{"object": d["object"], "say": str(d.get("say") or "")} for d in _objects(raw)]
    kept, invented = grounded(items, labels)
    res = {"type": kind, "items": kept, "invented": invented, "raw": raw, "model": lang.name}
    res["ms"] = round(1000 * (time.perf_counter() - t0))
    if (usd := _usd(lang)) is not None:
        res["usd"] = usd
    return res


def plan(lang, pins, request: str, now: float) -> dict:
    """An ordered list of things to pick up or go to for `request`, all from the bank."""
    b = {k: v["near"] for k, v in bank(pins, now).items()}
    user = json.dumps({"request": request, "memory": b})
    return _ask(lang, "plan", PLAN_SYSTEM, user, b, max_tokens=600)


def nudges(lang, pins, now: float) -> dict:
    """Up to three things not seen for a while that are worth a glance."""
    b = {k: v["hours"] for k, v in bank(pins, now).items()}
    return _ask(lang, "nudges", NUDGE_SYSTEM, json.dumps(b), b, max_tokens=600)


class MapReview:
    def __init__(self, lang, pins, every_s: float = 600.0, now=time.time):
        self.lang, self.pins, self.every_s, self.now = lang, pins, every_s, now
        self.since = now()
        self.last: dict | None = None
        self.spent_usd = 0.0

    def changes(self, since: float) -> list[dict]:
        minutes = max(0.0, (self.now() - since) / 60)
        log = Tools(self.pins, None, lambda: None, self.now).what_changed_since(minutes)
        # people are anonymous tracks, never map furniture: they stay out of the review
        return [c for c in log["changes"] if c["object"] != "person"]

    def run(self) -> dict:
        """One review of everything since the last one. Blocking: call it off the event loop."""
        t0, start = time.perf_counter(), self.now()
        changes = self.changes(self.since)
        self.since = start
        res = {"type": "map_review", "ts": start, "changes": len(changes), "items": []}
        if changes:
            text = self.lang.complete(
                SYSTEM, json.dumps(changes), max_tokens=300 + 60 * len(changes)
            )
            res["items"] = parse(text)
            res["model"] = self.lang.name
            if (usd := _usd(self.lang)) is not None:
                res["usd"] = usd
                self.spent_usd += usd
        res["ms"] = round(1000 * (time.perf_counter() - t0))
        self.last = res
        return res

    def in_the_way(self) -> set[str]:
        """Labels the last review called a new thing to walk into: for the reflex's priors."""
        return {i["object"] for i in (self.last or {}).get("items", []) if i["in_the_way"]}
