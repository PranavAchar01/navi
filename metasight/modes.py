"""The four things the wearer can be doing: Guide, Find, Talk, Explore.

One mode is active at a time. It changes how loud the safety cues are, whether the belt turns
the wearer toward a speaker, and what the glasses say when it changes. The mode is picked by
voice ("find mode"), implied by a question ("who is talking" means Talk), or set by the app.
Not to be confused with `Server.mode`, which is the home/outside map setting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .belt import FLAG_DROP
from .belt import MODES as CUE_MODES

GUIDE, FIND, TALK, EXPLORE = "guide", "find", "talk", "explore"
ALL = (GUIDE, FIND, TALK, EXPLORE)
FLAG_MODE_SHIFT = 2  # belt packet flags: bits 0 and 1 are drop and found
FLAG_MODE_MASK = 3

LINES = {
    GUIDE: "Guide. I'll watch the way ahead.",
    FIND: "Find. Tell me what you are looking for.",
    TALK: "Talk. I'll point you toward whoever is speaking.",
    EXPLORE: "Explore. I'll tell you what is around you.",
}


@dataclass(frozen=True)
class Profile:
    obstacle_scale: float  # non-urgent proximity cues are scaled by this
    speaker_cues: bool  # the belt and the beep turn the wearer toward a speaker


PROFILES = {
    GUIDE: Profile(1.0, False),
    FIND: Profile(1.0, False),
    TALK: Profile(0.6, True),
    EXPLORE: Profile(0.85, False),
}

_NAMES = "|".join(ALL)
_EXPLICIT = re.compile(
    rf"\b(?:(?:switch|change|go|turn|put me)\s+(?:in)?to\s+|start\s+|enter\s+)?"
    rf"({_NAMES})\s+mode\b|\b(?:switch|change|go)\s+to\s+({_NAMES})\b"
)
_TALK = re.compile(r"\b(who is|who's|who was)\s+(talking|speaking)\b|\bwho said\b|\bwhere.*voice\b")
_EXPLORE = re.compile(
    r"\bwhat(?:'s| is)? (?:around|near|in front|nearby)\b|\bdescribe\b|\bwhat do you see\b"
)


def command(text: str) -> str | None:
    """The mode a spoken command names ("find mode", "switch to talk"), else None."""
    m = _EXPLICIT.search(text.lower())
    return (m.group(1) or m.group(2)) if m else None


def implied(text: str, finds_item: bool) -> str | None:
    """The mode a question implies. `finds_item`: it names a thing to look for."""
    t = text.lower()
    if _TALK.search(t):
        return TALK
    if _EXPLORE.search(t):
        return EXPLORE
    if finds_item:
        return FIND
    return None


class Activity:
    def __init__(self) -> None:
        self.mode = GUIDE

    @property
    def profile(self) -> Profile:
        return PROFILES[self.mode]

    def set(self, mode: str, cause: str, speak: bool = True) -> dict | None:
        """Switch; the event to broadcast, or None when already there."""
        if mode not in ALL or mode == self.mode:
            return None
        prev, self.mode = self.mode, mode
        ev = {
            "type": "activity",
            "mode": mode,
            "prev": prev,
            "cause": cause,
            "speaker_cues": PROFILES[mode].speaker_cues,
            "obstacle_scale": PROFILES[mode].obstacle_scale,
        }
        if speak:
            ev["say"] = LINES[mode]
        return ev

    def event(self) -> dict:
        p = self.profile
        return {
            "type": "activity",
            "mode": self.mode,
            "speaker_cues": p.speaker_cues,
            "obstacle_scale": p.obstacle_scale,
        }

    def flag_bits(self) -> int:
        return (ALL.index(self.mode) & FLAG_MODE_MASK) << FLAG_MODE_SHIFT

    def calm(self, cmd: list[tuple[int, int]], flags: int = 0) -> list[tuple[int, int]]:
        """Scale down the cues that are not urgent. Solid (about to touch) and a drop stay full."""
        k = self.profile.obstacle_scale
        if k >= 1.0:
            return cmd
        if flags & FLAG_DROP:
            return cmd
        return [(m, i) if m == CUE_MODES["solid"] else (m, int(i * k)) for m, i in cmd]
