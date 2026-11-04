"""The deep loop: every few minutes a slow model reads what changed on the map and says which
changes matter. Nobody waits on it, so it can be the slow, careful model (Muse Wonder) or the
open one on this Mac (Muse Glimmer).

Input is the change log the pins already keep (`Tools.what_changed_since`: moved or new
objects, with what they were near). Output, per change: does it matter, is it something to walk
into, and one short sentence the wearer could be told. Nothing is called when nothing changed.

It also tidies the map (`tidy`): pins the geometric merge leaves apart but that look like one
object seen twice, or a weak one-off sighting next to a look-alike, go to the model, and only
checked proposals are applied (`apply_tidy`).

The same model also makes plans ("get ready to leave" -> keys, then the door) and nudges (a
thing not seen for a long time). Both only ever name things in the memory bank: `grounded`
drops anything else before it reaches the wearer.
"""

from __future__ import annotations

import json
import math
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
        if p.kind == "object" and p.label not in ("person", *out) and pins.stands(p):
            out[p.label] = {"near": p.near, "hours": round((now - p.ts) / 3600, 1)}
    return out


def grounded(items: list[dict], labels) -> tuple[list[dict], list[str]]:
    """Keep only items naming a thing in the bank; return the rest by name, to count them."""
    known = {x.lower(): x for x in labels}
    kept = [
        {**i, "object": known[i["object"].lower()]} for i in items if i["object"].lower() in known
    ]
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


TIDY_SYSTEM = (
    "You tidy a blind person's home map. Each candidate is a group of pins a camera made; drift "
    "and misreads leave duplicates (one object pinned twice) and ghosts (a weak single sighting "
    "of something that is not there, often a look-alike next to it). For each candidate reply "
    'with one JSON object per line, nothing else: {"ids": [int], "action": "merge"|"drop"|'
    '"keep", "why": str}. merge: the pins are one object. drop: only for a single weak pin that '
    "is a ghost. keep: anything you are unsure about. why: one short sentence."
)
DUP_REACH_M = 2.0  # same label further apart than the merge radius, but close enough to be one
GHOST_SCORE = 0.5  # a pin seen once below this score is a ghost candidate


def tidy_candidates(pins, radius: float) -> list[dict]:
    """Groups the geometric merge leaves alone but a reader might not: same-label pins between
    `radius` and DUP_REACH_M apart, and weak single sightings."""
    obj = [p for p in pins.pins.values() if p.kind == "object" and p.label != "person"]

    def card(p) -> dict:
        return {
            "id": p.id,
            "label": p.label,
            "seen": p.seen,
            "score": round(p.score, 2),
            "near": p.near[:4],
            "pos": [round(v, 2) for v in p.pos],
        }

    out, used = [], set()
    for i, a in enumerate(obj):
        group = [a] + [
            b
            for b in obj[i + 1 :]
            if b.label == a.label and radius < math.dist(a.pos, b.pos) <= DUP_REACH_M
        ]
        if len(group) > 1 and not used & {p.id for p in group}:
            used |= {p.id for p in group}
            out.append({"kind": "duplicate?", "pins": [card(p) for p in group]})
    for p in obj:
        if p.id not in used and p.seen == 1 and p.score < GHOST_SCORE:
            out.append({"kind": "ghost?", "pins": [card(p)]})
    return out


def tidy(lang, pins, radius: float = 0.5) -> dict:
    """Ask the deep model which candidate groups are one object or a ghost. Proposals are
    checked: ids must come from one candidate, `drop` only a single weak pin, `merge` only
    pins of one label. Nothing is changed here; `apply_tidy` does that."""
    t0 = time.perf_counter()
    cands = tidy_candidates(pins, radius)
    res: dict = {"type": "map_tidy", "candidates": len(cands), "items": [], "rejected": []}
    if not cands:
        res["ms"] = 0
        return res
    raw = lang.complete(TIDY_SYSTEM, json.dumps(cands), max_tokens=200 + 70 * len(cands))
    groups = [{c["id"] for c in g["pins"]} for g in cands]
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        ids = {int(x) for x in d.get("ids", []) if isinstance(x, int | float)}
        act = d.get("action")
        ok = bool(ids) and any(ids <= g for g in groups) and act in ("merge", "drop", "keep")
        if ok and act == "drop":
            ok = len(ids) == 1 and all(
                pins.pins[i].seen == 1 and pins.pins[i].score < GHOST_SCORE for i in ids
            )
        if ok and act == "merge":
            ok = len(ids) > 1 and len({pins.pins[i].label for i in ids}) == 1
        (res["items"] if ok else res["rejected"]).append(
            {"ids": sorted(ids), "action": act, "why": str(d.get("why") or "")}
        )
    res |= {"raw": raw, "model": lang.name, "ms": round(1000 * (time.perf_counter() - t0))}
    if (usd := _usd(lang)) is not None:
        res["usd"] = usd
    return res


def apply_tidy(pins, res: dict) -> dict:
    """Carry out checked proposals: fold merged pins into the most seen one, remove ghosts."""
    merged = dropped = 0
    for it in res["items"]:
        ids = [i for i in it["ids"] if i in pins.pins]
        if it["action"] == "drop" and len(ids) == 1:
            del pins.pins[ids[0]]
            dropped += 1
        elif it["action"] == "merge" and len(ids) > 1:
            group = sorted((pins.pins[i] for i in ids), key=lambda p: -p.seen)
            host, n = group[0], sum(p.seen for p in group)
            host.pos = [sum(p.pos[k] * p.seen for p in group) / n for k in range(len(host.pos))]
            host.seen, host.ts = n, max(p.ts for p in group)
            host.first_ts = min(p.first_ts for p in group)
            for p in group[1:]:
                del pins.pins[p.id]
            merged += len(group) - 1
    return {"merged": merged, "dropped": dropped}
