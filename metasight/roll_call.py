"""Room roll call: what is around you, from memory, as a clock face, each name with a pluck from
where the thing really is.

"Bedroom. Bed at two o'clock, two metres. Desk at ten o'clock, one metre." The map's pins are
the source, so things behind the wearer are named too (Soundscape's "Around Me" does this for
street places, Seeing AI for what the camera sees now). Only pins seen many times and placed in
this walk's frame count; duplicates of one label close together become one; the furniture you
find your way by goes first; people are left out (they move). The list runs clockwise from
straight ahead.

Pure functions: the server says the line and sends the items (with where each name sits in the
text, so the phone can pluck as the name is spoken); the renderer here makes the same thing as
a stereo track for the video.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from . import spatial_sound as ss

MIN_SEEN = 20  # sightings before a pin is trusted enough to name
MIN_SCORE = 0.5
SAME_M = 1.0  # one label closer than this is one thing
ON_TOP_M = 0.6  # a laptop on the bed is not named apart from the bed
MAX_ITEMS = 6
# What you find your way around a room by, most useful first. Anything else fills the rest.
LAYOUT = (
    "door", "bed", "desk", "chair", "table", "sofa", "armchair", "dresser", "cabinet",
    "nightstand", "shelf", "mini fridge", "trash can", "stool", "tv",
)  # fmt: skip
SKIP = {"person", "wall", "window", "curtain", "rug", "picture frame", "mirror"}
NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
HOURS = (
    "twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven",
)  # fmt: skip


@dataclass(frozen=True)
class Item:
    label: str
    pos: tuple[float, float, float]  # map frame
    dist_m: float  # along the floor
    bearing_deg: float  # clockwise from straight ahead, -180..180


def _get(p, k, default=None):
    return p.get(k, default) if isinstance(p, dict) else getattr(p, k, default)


def pick(
    pins: Iterable,
    to: Callable[[np.ndarray], tuple[float, float] | None],
    k: int = MAX_ITEMS,
    min_seen: int = MIN_SEEN,
) -> list[Item]:
    """The few things worth naming, clockwise from straight ahead. `to` is heading.to: map
    point to (floor metres, bearing degrees). Pins are Pin objects or their JSON."""
    keep: list[dict] = []
    for p in sorted(pins, key=lambda p: -_get(p, "seen", 0)):
        label = _get(p, "label", "").lower()
        if (
            label in SKIP
            or not _get(p, "placed", True)
            or _get(p, "seen", 0) < min_seen
            or _get(p, "score", 0.0) < MIN_SCORE
        ):
            continue
        pos = np.asarray(_get(p, "pos"), float)
        if any(q["label"] == label and np.linalg.norm(q["pos"] - pos) < SAME_M for q in keep):
            continue
        keep.append({"label": label, "pos": pos, "seen": _get(p, "seen", 0)})

    def useful(q):
        return (LAYOUT.index(q["label"]) if q["label"] in LAYOUT else len(LAYOUT), -q["seen"])

    chosen = []
    for q in sorted(keep, key=useful):
        rel = to(q["pos"])
        if rel is None:
            return []  # no pose: nothing can be placed
        if len(chosen) < k and all(
            np.linalg.norm(q["pos"] - np.asarray(c.pos)) >= ON_TOP_M for c in chosen
        ):
            chosen.append(Item(q["label"], tuple(q["pos"].tolist()), rel[0], rel[1]))
    return sorted(chosen, key=lambda it: it.bearing_deg % 360)


def clock(bearing_deg: float) -> str:
    """Twelve is straight ahead, three is right, six behind."""
    return HOURS[round((bearing_deg % 360) / 30) % 12] + " o'clock"


def metres(d: float) -> str:
    """Spoken to the nearest half metre: "half a metre", "one metre", "two and a half metres"."""
    h = max(1, round(d * 2))
    whole, half = divmod(h, 2)
    if whole == 0:
        return "half a metre"
    word = NUMBERS[whole] if whole < len(NUMBERS) else str(whole)
    if half:
        return f"{word} and a half metres"
    return f"{word} metre" + ("" if whole == 1 else "s")


def line(items: list[Item], room: str | None = None) -> tuple[str, list[tuple[int, int]]]:
    """The spoken roll call and, for each item, (start, length) of its name in the text, so the
    phone can pluck from the item's direction as the name is spoken."""
    text = f"{room.capitalize()}. " if room else ""
    spans = []
    if not items:
        return text + "I don't know this room well enough yet.", []
    for it in items:
        name = it.label.capitalize()
        spans.append((len(text), len(name)))
        text += f"{name} at {clock(it.bearing_deg)}, {metres(it.dist_m)}. "
    return text.strip(), spans


def message(items: list[Item], room: str | None = None) -> dict:
    """The websocket message for the phone: the text to say and each name's place in it."""
    text, spans = line(items, room)
    return {
        "type": "rollcall",
        "text": text,
        "items": [
            {
                "label": it.label,
                "bearing": round(it.bearing_deg, 1),
                "dist": round(it.dist_m, 2),
                "start": s,
                "length": n,
            }
            for it, (s, n) in zip(items, spans, strict=True)
        ],
    }


def render(
    items: list[Item],
    head: np.ndarray,
    fwd: np.ndarray,
    up: np.ndarray,
    voice: list[np.ndarray] | None = None,
    gap_s: float = 1.6,
    lead_s: float = 0.4,
    method: str = "hrtf",
) -> np.ndarray:
    """Stereo track for the video. `voice` is one mono clip per sentence at ss.SAMPLE_RATE (the
    room name first when there is one, then one per item); each item's pluck starts with its
    sentence, from the item's true direction. Without voice, plucks are `gap_s` apart."""
    starts, t = [], lead_s
    clips = list(voice) if voice else []
    room_clip = clips.pop(0) if len(clips) == len(items) + 1 else None
    if room_clip is not None:
        t += len(room_clip) / ss.SAMPLE_RATE
    for i in range(len(items)):
        starts.append(t)
        t += (len(clips[i]) / ss.SAMPLE_RATE + 0.15) if clips else gap_s
    out = np.zeros((int((t + 0.5) * ss.SAMPLE_RATE), 2))
    if room_clip is not None:
        n0 = int(lead_s * ss.SAMPLE_RATE)
        out[n0 : n0 + len(room_clip)] += room_clip[:, None] * 0.7
    for i, (it, t0) in enumerate(zip(items, starts, strict=True)):
        n0 = int(t0 * ss.SAMPLE_RATE)
        d = ss.relative(head, fwd, up, np.asarray(it.pos))
        st = ss.place(ss.beacon_sound(False) * ss.beacon_gain(d.ground_m), d, method)
        out[n0 : n0 + len(st)] += st[: len(out) - n0]
        if clips:
            c = clips[i]
            m = int(0.12 * ss.SAMPLE_RATE)  # the name lands just after the pluck
            out[n0 + m : n0 + m + len(c)] += c[: len(out) - n0 - m, None] * 0.7
    return np.tanh(out).astype(np.float32)
