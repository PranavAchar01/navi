"""First run: learn the home by walking it, name a few things, feel what each mode does.

Voice-led and short. The server owns the state; the app speaks each `say` line, shows the
progress and sends "start" or "skip". Progress is the floor the wearer has actually walked
(from the memory worker's camera trail), counted in 0.5 m cells, since the worker keeps no
per-room notion: the lines say so ("a good first look"), not "the kitchen is done".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .belt import MODES
from .modes import ALL, LINES
from .pan import pan_gains
from .query import ALIASES

CELL_M = 0.5
# Walked floor (m2, counted in whole cells) at which a line is said, and where the walk is done.
MILESTONES = ((2.0, "Good. Keep going."), (6.0, "Nice and slow. That's it."))
WALK_DONE_M2 = 12.0
MIN_ITEMS = 1
MAX_ITEMS = 6

WALK_LINE = "Let's learn your home. Walk slowly through each room."
WALK_DONE_LINE = "That's a good first look at your home."
NAME_LINE = "Now tell me a few things you care about. Say, these are my keys."
NAME_DONE_LINE = "Lovely. I'll look out for those."
BELT_LINE = "Last, feel the belt. One pattern for each mode."
END_LINE = "All set. I'm here when you need me."

_NAMING = re.compile(
    r"\b(?:these are|this is|that is|that's|those are|here are|here is|remember|this one is)\s+"
    r"(?:my|the)\s+([a-z][a-z' ]{1,28}?)(?:\s+(?:here|there|please))?\s*$"
)
_DONE = re.compile(r"\b(that's all|that is all|that's it|i'm done|im done|done|finished|skip)\b")


def item_label(text: str) -> str | None:
    """The thing named in "these are my keys", as the watch-list word, else None."""
    t = re.sub(r"[^a-z' ]", " ", text.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    m = _NAMING.search(t)
    if not m:
        return None
    said = m.group(1).strip()
    for label, words in ALIASES.items():
        if said == label or said in words:
            return label
    return said


# A pattern is segments (start s, end s, bearing deg from the front, belt mode, intensity). These
# are stand-ins in the belt's own vocabulary; set PATTERNS[mode] to replace one with the final one.
PATTERNS: dict[str, list[tuple[float, float, float, str, int]]] = {
    "guide": [(0.0, 0.4, 0, "slow", 200), (0.7, 1.1, 0, "slow", 200)],
    "find": [(0.0, 0.6, 45, "slow", 170), (0.9, 1.5, 45, "slow", 170)],
    "talk": [(0.0, 0.5, -60, "slow", 110), (0.7, 1.2, 60, "slow", 110)],
    "explore": [(i * 0.35, i * 0.35 + 0.3, 90 * i, "slow", 100) for i in range(4)],
}


def pattern_cmd(segments, t: float) -> list[tuple[int, int]]:
    """The eight motor commands of a pattern `t` seconds in (all off outside its segments)."""
    cmd = [(MODES["off"], 0)] * 8
    for start, end, bearing, mode, level in segments:
        if start <= t < end:
            a, ga, b, gb = pan_gains(bearing)
            for motor, gain in ((a, ga), (b, gb)):
                if gain > 0.05:
                    cmd[motor] = (MODES[mode], int(level * gain))
    return cmd


def pattern_length(segments) -> float:
    return max((s[1] for s in segments), default=0.0)


START_PHRASES = (
    "let's set up",
    "lets set up",
    "set up my home",
    "setup my home",
    "learn my home",
    "learn my house",
    "learn my place",
)


def wants_setup(text: str) -> bool:
    """The wearer asked to start the first-run setup by voice."""
    t = " ".join(text.lower().replace("\u2019", "'").split())
    return any(p in t for p in START_PHRASES)


@dataclass
class Onboarding:
    step: str = "idle"  # idle, walk, name, belt, done
    cells: frozenset = frozenset()
    items: tuple = ()
    said: int = 0  # milestones already spoken
    demo_mode: str | None = None
    demo_t0: float = 0.0

    @property
    def active(self) -> bool:
        return self.step in ("walk", "name", "belt")

    @property
    def area_m2(self) -> float:
        return len(self.cells) * CELL_M * CELL_M

    def event(self, say: str | None = None, **extra) -> dict:
        ev = {
            "type": "onboarding",
            "step": self.step,
            "progress": round(min(1.0, self.area_m2 / WALK_DONE_M2), 2),
            "area_m2": round(self.area_m2, 1),
            "items": list(self.items),
            **extra,
        }
        if say:
            ev["say"] = say
        return ev

    def start(self) -> dict:
        self.__init__()
        self.step = "walk"
        return self.event(WALK_LINE)

    def skip(self) -> dict:
        self.step, self.demo_mode = "done", None
        return self.event()

    def trail(self, points) -> list[dict]:
        """New camera positions [x, y, z, ts] from the map worker. Floor is x and z."""
        if self.step != "walk":
            return []
        before = len(self.cells)
        self.cells = self.cells | {(int(p[0] // CELL_M), int(p[2] // CELL_M)) for p in points}
        if len(self.cells) == before:
            return []
        if self.area_m2 >= WALK_DONE_M2:
            self.step = "name"
            return [self.event(WALK_DONE_LINE), self.event(NAME_LINE)]
        if self.said < len(MILESTONES) and self.area_m2 >= MILESTONES[self.said][0]:
            line = MILESTONES[self.said][1]
            self.said += 1
            return [self.event(line)]
        return [self.event()]

    def hear(self, text: str) -> tuple[bool, list[dict], str | None]:
        """(handled, events, label to add to the watch list) for something the wearer said."""
        t = text.lower()
        if self.step == "walk" and _DONE.search(t):
            self.step = "name"
            return True, [self.event(WALK_DONE_LINE), self.event(NAME_LINE)], None
        if self.step != "name":
            return False, [], None
        label = item_label(text)
        if label and len(self.items) < MAX_ITEMS:
            if label not in self.items:
                self.items = (*self.items, label)
            line = f"Got it, {label}."
            if len(self.items) >= MAX_ITEMS:
                self.step = "belt"
                return True, [self.event(line), self.event(NAME_DONE_LINE)], label
            return True, [self.event(line)], label
        if _DONE.search(t) and len(self.items) >= MIN_ITEMS:
            self.step = "belt"
            return True, [self.event(NAME_DONE_LINE)], None
        return False, [], None

    def belt_events(self) -> list[tuple[dict, str]]:
        """What the belt intro says and plays, in order: (event, mode whose pattern plays)."""
        out = [(self.event(BELT_LINE, mode=None), "")]
        out += [(self.event(LINES[m], mode=m), m) for m in ALL]
        return out

    def finish(self) -> dict:
        self.step, self.demo_mode = "done", None
        return self.event(END_LINE)

    def demo_cmd(self, now: float) -> list[tuple[int, int]] | None:
        """The belt override while a mode's pattern plays, else None."""
        if self.demo_mode is None:
            return None
        return pattern_cmd(PATTERNS[self.demo_mode], now - self.demo_t0)
