"""The deep loop: every few minutes a slow model reads what changed on the map and says which
changes matter. Nobody waits on it, so it can be the slow, careful model (Muse Wonder) or the
open one on this Mac (Muse Glimmer).

Input is the change log the pins already keep (`Tools.what_changed_since`: moved or new
objects, with what they were near). Output, per change: does it matter, is it something to walk
into, and one short sentence the wearer could be told. Nothing is called when nothing changed.
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


def parse(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d.get("object"), str):
            out.append(
                {
                    "object": d["object"],
                    "matters": bool(d.get("matters")),
                    "in_the_way": bool(d.get("in_the_way")),
                    "say": str(d.get("say") or ""),
                }
            )
    return out


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
            usage = getattr(self.lang, "usage", None)
            if usage is not None and self.lang.name in PRICE:
                pin, pout = PRICE[self.lang.name]
                res["usd"] = (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6
                self.spent_usd += res["usd"]
        res["ms"] = round(1000 * (time.perf_counter() - t0))
        self.last = res
        return res

    def in_the_way(self) -> set[str]:
        """Labels the last review called a new thing to walk into: for the reflex's priors."""
        return {i["object"] for i in (self.last or {}).get("items", []) if i["in_the_way"]}
