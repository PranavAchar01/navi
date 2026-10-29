"""Tool calls for the language model (Muse Spark): it asks the pins, the map and the people
tracks for facts instead of being handed a pre-digested paragraph.

Tools read state and never change it. People are anonymous track numbers: nothing here can name
or recognise anyone.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "where_is",
            "description": "Where an object was last seen: distance and direction from the "
            "wearer now, how long ago, and the landmarks near it.",
            "parameters": {
                "type": "object",
                "properties": {"object": {"type": "string", "description": "e.g. keys"}},
                "required": ["object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "what_changed_since",
            "description": "Objects that appeared or moved to a new place in the last N minutes.",
            "parameters": {
                "type": "object",
                "properties": {"minutes": {"type": "number"}},
                "required": ["minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "who_is_speaking_direction",
            "description": "Which way the current voice comes from relative to where the "
            "wearer faces, and whether that person is in view.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "people_in_view",
            "description": "How many people are in view, with direction and distance each "
            "(anonymous numbers, no names).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM = (
    "You answer a wearer's question about their surroundings in one short spoken sentence. "
    "Use the tools for facts; never guess. Distances in metres, directions as left/right/ahead/"
    "behind. People have no names here: say 'someone'. A tool that answers with an empty "
    "changes list means nothing changed: say so. Say you do not know only when a tool fails "
    "or has no data."
)


def side(bearing: float) -> str:
    a = abs(bearing)
    if a < 20:
        return "ahead"
    if a > 135:
        return "behind"
    return ("right" if bearing > 0 else "left") + (" and ahead" if a < 70 else "")


class Tools:
    """`pins` is a PinStore, `heading` a Heading, `people` returns the latest people event."""

    def __init__(self, pins, heading, people: Callable[[], dict | None], now=time.time):
        self.pins, self.heading, self.people, self.now = pins, heading, people, now

    def where_is(self, object: str) -> dict:
        pin = self.pins.latest(object.lower().strip())
        if pin is None:
            return {"found": False}
        out = {
            "found": True,
            "seen_s_ago": round(self.now() - pin.ts),
            "near": pin.near[:3],
            "sightings": pin.seen,
        }
        rel = self.heading.to(pin.pos) if pin.placed else None
        if rel is not None:
            out.update(
                distance_m=round(rel[0], 1), bearing_deg=round(rel[1]), direction=side(rel[1])
            )
        return out

    def _rel(self, pin) -> dict:
        rel = self.heading.to(pin.pos) if pin.placed else None
        if rel is None:
            return {}
        return {"distance_m": round(rel[0], 1), "direction": side(rel[1])}

    def what_changed_since(self, minutes: float) -> dict:
        cut = self.now() - 60 * minutes
        by: dict[str, list] = {}
        for p in self.pins.pins.values():
            if p.kind == "object":
                by.setdefault(p.label, []).append(p)
        events = []
        for label, ps in by.items():
            ps.sort(key=lambda p: p.ts)
            new = ps[-1]
            if len(ps) > 1 and new.first_ts >= cut and ps[-2].ts < new.first_ts:
                events.append(
                    {
                        "object": label,
                        "change": "moved",
                        "now_near": new.near[:2],
                        "was_near": ps[-2].near[:2],
                    }
                )
            elif len(ps) == 1 and new.first_ts >= cut:
                events.append({"object": label, "change": "new", "near": new.near[:2]})
        return {"window_min": minutes, "changes": events}

    def who_is_speaking_direction(self) -> dict:
        ev = self.people()
        if not ev or not ev.get("voice"):
            return {"speaking": False}
        spk = next((t for t in ev["tracks"] if t["id"] == ev.get("speaker")), None)
        if spk is not None:
            return {
                "speaking": True,
                "in_view": True,
                "bearing_deg": spk["bearing"],
                "direction": side(spk["bearing"]),
                "distance_m": spk["dist"],
                "evidence": ev["evidence"],
            }
        if ev.get("doa") is not None:
            return {
                "speaking": True,
                "in_view": False,
                "bearing_deg": ev["doa"],
                "direction": side(ev["doa"]),
                "evidence": "microphones",
            }
        return {"speaking": True, "in_view": None, "direction": "unknown"}

    def people_in_view(self) -> dict:
        ev = self.people() or {"tracks": []}
        ts = [t for t in ev["tracks"] if not t["lost"]]
        return {
            "count": len(ts),
            "people": [
                {
                    "id": t["id"],
                    "direction": side(t["bearing"]),
                    "distance_m": t["dist"],
                    "speaking": t["speaking"] >= 0.35,
                }
                for t in ts
            ],
        }

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None) if name in {t["function"]["name"] for t in TOOLS} else None
        if fn is None:
            return json.dumps({"error": "unknown tool"})
        try:
            return json.dumps(fn(**args), default=lambda o: None if not math.isfinite(o) else o)
        except TypeError:
            return json.dumps({"error": "bad arguments"})


def agent_answer(lang, question: str, tools: Tools, rounds: int = 3) -> str:
    """Let the model call tools until it answers. `lang` is an OpenAICompatLanguage."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(rounds):
        r = lang._client.chat.completions.create(
            model=lang.model,
            messages=msgs,
            tools=TOOLS,
            max_tokens=400,
            temperature=0.2,
            extra_body=lang.extra,
        )
        m = r.choices[0].message
        if not m.tool_calls:
            return (m.content or "").strip()
        msgs.append(
            {
                "role": "assistant",
                "content": m.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                    }
                    for c in m.tool_calls
                ],
            }
        )
        for c in m.tool_calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            msgs.append(
                {"role": "tool", "tool_call_id": c.id, "content": tools.call(c.function.name, args)}
            )
    return ""
