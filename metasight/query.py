"""Turn "where are my keys" into a place in the map and one sentence."""

from __future__ import annotations

import re

import numpy as np

from .geometry import Relative, relative_to
from .pins import Pin

ALIASES = {
    "keys": ["keys", "key", "car keys", "house keys", "my keys", "keychain", "key ring"],
    "wallet": ["wallet", "billfold"],
    "phone": ["phone", "cell phone", "cellphone", "iphone", "mobile"],
    "glasses case": ["glasses case", "glasses", "sunglasses", "spectacles"],
    "remote control": ["remote", "remote control", "clicker", "tv remote"],
    "backpack": ["backpack", "bag", "rucksack"],
    "water bottle": ["water bottle", "bottle", "my bottle", "flask", "hydro flask", "tumbler"],
}
_ASK = re.compile(
    r"\b(where(?:'s| is| are| did i (?:leave|put))?|find|locate|have you seen|did you see|"
    r"do you know where)\b"
)


def parse_query(text: str, known: list[str] | None = None) -> str | None:
    """The watch-list label a question is about, or None. Longest alias wins."""
    t = re.sub(r"[^a-z' ]", " ", text.lower())
    pool = dict(ALIASES)
    for label in known or []:
        pool.setdefault(label, [label])
    best, best_len = None, 0
    for label, words in pool.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", t) and len(w) > best_len:
                best, best_len = label, len(w)
    return best


def is_question(text: str) -> bool:
    return bool(_ASK.search(text.lower())) or text.strip().endswith("?")


def direction_phrase(bearing_deg: float) -> str:
    b = (bearing_deg + 180) % 360 - 180
    a = abs(b)
    side = "right" if b > 0 else "left"
    if a < 22.5:
        return "straight ahead"
    if a < 67.5:
        return f"ahead of you to the {side}"
    if a < 112.5:
        return f"on your {side}"
    if a < 157.5:
        return f"behind you to the {side}"
    return "behind you"


def distance_phrase(m: float) -> str:
    if m < 1.0:
        return "less than a metre"
    if m < 1.5:
        return "about a metre"
    r = round(m * 2) / 2
    n = f"{r:g}" if r != int(r) else str(int(r))
    return f"about {n} metres"


def time_ago(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 45:
        return "just now"
    m = round(s / 60)
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''} ago"
    h = round(s / 3600)
    return f"{h} hour{'s' if h != 1 else ''} ago"


def template_answer(
    label: str, pin: Pin | None, pose: np.ndarray | None, now: float, stand_in: bool = False
) -> str:
    """`stand_in`: `pin` is a look-alike (a cup for the water bottle), not a confirmed sighting."""
    noun = f"your {label}"
    was = "were" if label.endswith("s") else "was"
    if pin is None:
        return f"I haven't seen {noun} yet. Walk past where you left them and ask again."
    ago = time_ago(now - pin.ts)
    where = f" near the {pin.near[0]}" if pin.near else ""
    if stand_in:
        noun = f"I'm not sure I saw {noun}, but a {pin.label}"
        was = "was"
    if pose is None:
        return f"{noun[0].upper() + noun[1:]} {was} last seen{where}, {ago}."
    rel: Relative = relative_to(np.array(pin.pos), pose)
    if rel.horizontal_m < 1.0:
        place = "right next to you"
    else:
        place = f"{distance_phrase(rel.horizontal_m)} {direction_phrase(rel.bearing_deg)}"
    return f"{noun[0].upper() + noun[1:]} {was}{where}, {place}, seen {ago}."


def live_answer(label: str, dist_m: float, bearing_deg: float) -> str:
    """The camera sees it right now: that beats any remembered sighting, however recent."""
    was = "are" if label.endswith("s") else "is"
    if dist_m < 1.0:
        place = "right in front of you"
    else:
        place = f"{distance_phrase(dist_m)} {direction_phrase(bearing_deg)}"
    return f"Your {label} {was} {place}, in view now."


def camera_answer(label: str, seconds: float) -> str:
    """Seen by the camera this walk, but not yet placed on the map."""
    when = time_ago(seconds)
    when = "a moment ago" if when == "just now" else when
    return f"I saw your {label} {when}, but I don't have its place on the map yet."


def facts_for_language(label: str, pin: Pin, pose: np.ndarray | None, now: float) -> str:
    """Compact JSON-ish facts for the language model: it phrases, it never invents."""
    rel = relative_to(np.array(pin.pos), pose) if pose is not None else None
    bits = [
        f"object: {label}",
        f"seen: {time_ago(now - pin.ts)}",
    ]
    if pin.near:  # "near: nothing" gets read out word for word by the short prompt
        bits.append(f"near: {', '.join(pin.near)}")
    if rel:
        bits += [
            f"distance: {distance_phrase(rel.horizontal_m)}",
            f"direction: {direction_phrase(rel.bearing_deg)}",
            f"height: {'below eye level' if rel.dy_m < -0.3 else 'around eye level'}",
        ]
    return "\n".join(bits)


# Kept this short on purpose: Muse Spark reasons even at "minimal", and every extra rule here
# bought ~200 more reasoning tokens. The longer prompt took 1.2 to 1.8 s, this one 0.6 to 0.9 s
# (bench/results/spark-prompt-length.json); the facts already carry the place and the time.
SYSTEM = "Answer in ONE short spoken sentence using only the facts given."


def valid_sentence(text: str) -> bool:
    t = text.strip()
    return bool(t) and "\n" not in t and len(t) <= 240
