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
from .belt import MODES
from .belt_modes import talk_cmd

MIN_SEEN = 20  # sightings before a pin is trusted enough to name
FURNITURE_MIN_SEEN = 10  # furniture stays put, so fewer sightings will do
MIN_SCORE = 0.5
SAME_M = 1.0  # one label closer than this is one thing
TWIN_SHARE = 0.5  # a second thing of one kind needs half the first one's sightings
ON_TOP_M = 0.6  # a laptop on the bed is not named apart from the bed
MAX_ITEMS = 6
FILL_RANGE_M = 2.0  # a small thing (not LAYOUT) is only named this close: a far bottle is noise
MAX_PHRASES = 5  # things at one clock hour share a phrase; more than five phrases is a lecture
# What you find your way around a room by, most useful first. Anything else fills the rest.
LAYOUT = (
    "door", "bed", "desk", "monitor", "chair", "table", "sofa", "armchair", "dresser", "cabinet",
    "nightstand", "shelf", "couch", "mini fridge", "trash can", "stool", "tv",
)  # fmt: skip
FURNITURE = {"bed", "desk", "chair", "table", "sofa", "couch", "armchair", "dresser", "stool"}
ALIASES = {"dining table": "table"}  # the detector's word, then what a person says
SAID = {"tv": "TV"}
# What the detector's word means in a given room, checked on each object's best frame
# (2026-11-02): the bedroom's "tv" is the monitors on the desk (the twin has no desk of its own,
# and the desk is what a walker meets, so it is said as the desk), its "couch" is the bed's
# headboard, its "dining table" is the white folding table. None drops the word. Pranav to
# confirm.
ROOM_NAMES: dict[str, dict[str, str | None]] = {
    "bedroom": {"tv": "desk", "couch": None, "sofa": None},
}
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


def _seen(p) -> int:
    """Pins count sightings in `seen`, the room twin's objects in `views`."""
    return int(_get(p, "seen", None) or _get(p, "views", 0) or 0)


def _score(p) -> float:
    s = _get(p, "score", None)
    return float(_get(p, "conf", 0.0) if s is None else s)


def pick(
    pins: Iterable,
    to: Callable[[np.ndarray], tuple[float, float] | None],
    k: int = MAX_ITEMS,
    min_seen: int = MIN_SEEN,
    twin: Iterable = (),
    room: str | None = None,
) -> list[Item]:
    """The few things worth naming, clockwise from straight ahead. `to` is heading.to: map
    point to (floor metres, bearing degrees). Pins are Pin objects or their JSON; `twin` adds
    the room twin's objects (objects-s1.json: label, pos, views, conf), which see a chair over
    many walks where one walk's pins may not. Both must be in the frame `to` reads. `room`
    applies that room's spoken names (ROOM_NAMES)."""
    names = ROOM_NAMES.get((room or "").lower(), {})
    keep: list[dict] = []
    for p in sorted([*pins, *twin], key=lambda p: -_seen(p)):
        label = _get(p, "label", "").lower()
        label = ALIASES.get(label, label)
        label = names.get(label, label)
        if label is None:
            continue
        bar = FURNITURE_MIN_SEEN if label in FURNITURE else min_seen
        if label in SKIP or not _get(p, "placed", True) or _seen(p) < bar or _score(p) < MIN_SCORE:
            continue
        pos = np.asarray(_get(p, "pos"), float)
        same = [q for q in keep if q["label"] == label]
        if any(np.linalg.norm(q["pos"] - pos) < SAME_M for q in same):
            continue
        if same and _seen(p) < TWIN_SHARE * same[0]["seen"]:
            continue  # a faint second bed is the map's noise, not a second bed
        keep.append({"label": label, "pos": pos, "seen": _seen(p)})

    def useful(q):
        return (LAYOUT.index(q["label"]) if q["label"] in LAYOUT else len(LAYOUT), -q["seen"])

    chosen = []
    for q in sorted(keep, key=useful):
        rel = to(q["pos"])
        if rel is None:
            return []  # no pose: nothing can be placed
        if q["label"] not in LAYOUT and rel[0] > FILL_RANGE_M:
            continue
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


def phrases(items: list[Item], cap: int = MAX_PHRASES) -> list[list[Item]]:
    """Items (already clockwise) grouped by clock hour, at most `cap` groups: "Desk and basket
    at eight o'clock" is one phrase, not two."""
    groups: list[list[Item]] = []
    for it in items:
        if groups and clock(groups[-1][0].bearing_deg) == clock(it.bearing_deg):
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups[:cap]


def sentence(group: list[Item], at: int = 0) -> tuple[str, list[tuple[int, int]]]:
    """One phrase, "Basket and desk at eight o'clock, one and a half metres." (the nearest's
    distance), and each name's (start, length) counted from `at`."""
    text, spans = "", []
    for i, it in enumerate(group):
        name = SAID.get(it.label) or (it.label.capitalize() if i == 0 else it.label)
        if i:
            text += " and " if i == len(group) - 1 else ", "
        spans.append((at + len(text), len(name)))
        text += name
    near = min(it.dist_m for it in group)
    return f"{text} at {clock(group[0].bearing_deg)}, {metres(near)}.", spans


def line(items: list[Item], room: str | None = None) -> tuple[str, list[tuple[int, int]]]:
    """The spoken roll call and, for each spoken item, (start, length) of its name in the text,
    so the phone can pluck from the item's direction as the name is spoken."""
    text = f"{room.capitalize()}. " if room else ""
    spans: list[tuple[int, int]] = []
    if not items:
        return text + "I don't know this room well enough yet.", []
    for group in phrases(items):
        said, sp = sentence(group, len(text))
        spans += sp
        text += said + " "
    return text.strip(), spans


def spoken(items: list[Item]) -> list[Item]:
    """The items the line actually names (the phrase cap can drop the last few)."""
    return [it for g in phrases(items) for it in g]


def message(items: list[Item], room: str | None = None) -> dict:
    """The websocket message for the phone: the text to say and each name's place in it."""
    text, spans = line(items, room)
    items = spoken(items)
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
    """Stereo track for the video. `voice` is one mono clip per spoken sentence at
    ss.SAMPLE_RATE: the room name first, then one per phrase (`phrases`). Each name's pluck
    plays from the item's true direction where the name falls in its clip (by its place in the
    sentence). Without voice, phrases are `gap_s` apart."""
    groups = phrases(items)
    clips = list(voice) if voice else []
    room_clip = clips.pop(0) if len(clips) == len(groups) + 1 else None
    t = lead_s + (len(room_clip) / ss.SAMPLE_RATE if room_clip is not None else 0.0)
    plan = []  # (phrase start s, phrase length s, group, clip)
    for i, g in enumerate(groups):
        c = clips[i] if clips else None
        length = len(c) / ss.SAMPLE_RATE if c is not None else gap_s
        plan.append((t, length, g, c))
        t += length + (0.15 if c is not None else 0.0)
    out = np.zeros((int((t + 0.5) * ss.SAMPLE_RATE), 2))
    if room_clip is not None:
        n0 = int(lead_s * ss.SAMPLE_RATE)
        out[n0 : n0 + len(room_clip)] += room_clip[:, None] * 0.7
    m = int(0.12 * ss.SAMPLE_RATE)  # the voice starts just after the first pluck
    for t0, length, g, c in plan:
        said, spans = sentence(g)
        for it, (start, _) in zip(g, spans, strict=True):
            n0 = int((t0 + length * start / len(said)) * ss.SAMPLE_RATE)
            d = ss.relative(head, fwd, up, np.asarray(it.pos))
            st = ss.place(ss.beacon_sound(False) * ss.beacon_gain(d.ground_m), d, method)
            out[n0 : n0 + len(st)] += st[: len(out) - n0]
        if c is not None:
            n0 = int(t0 * ss.SAMPLE_RATE) + m
            out[n0 : n0 + len(c)] += c[: len(out) - n0, None] * 0.7
    return np.tanh(out).astype(np.float32)


SPEECH_CHARS_PER_S = 14.0  # iOS AVSpeechUtteranceDefaultSpeechRate reads about this fast


def taps(msg: dict, chars_per_s: float = SPEECH_CHARS_PER_S) -> list[tuple[float, float]]:
    """(seconds after the line starts, bearing) for one belt tap per name, estimated from where
    the name sits in the text. The phone plucks on the real word timing; the belt is on the Mac
    and only has this estimate."""
    return [(it["start"] / chars_per_s, it["bearing"]) for it in msg["items"]]


TAP_S = 0.12  # one short tap per name
TAP_LEVEL = 200


class BeltTaps:
    """The roll call's taps on the belt, laid over each frame's command like the Talk cue: a
    tap only takes a motor nothing else holds, and none plays while any cue is solid (walking
    into something outranks naming the room)."""

    def __init__(self) -> None:
        self.due: list[tuple[float, float]] = []  # (wall time, bearing)

    def start(self, msg: dict, now: float) -> None:
        self.due = [(now + t, b) for t, b in taps(msg)]

    def overlay(self, cmd: list[tuple[int, int]], now: float) -> list[tuple[int, int]]:
        self.due = [d for d in self.due if now < d[0] + TAP_S]
        live = [b for t, b in self.due if t <= now]
        if not live or any(mode == MODES["solid"] for mode, _ in cmd):
            return cmd
        out = list(cmd)
        for bearing in live:
            for motor, (mode, amp) in enumerate(talk_cmd(bearing, TAP_LEVEL)):
                if mode != MODES["off"] and out[motor][0] == MODES["off"]:
                    out[motor] = (mode, amp)
        return out
