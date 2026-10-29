"""Object pins: where things were last seen in the metric map."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

STRUCTURE = frozenset(
    {"door", "window", "stairs", "wall", "railing", "pole", "cabinet", "counter", "sink"}
    | {"toilet", "oven", "fridge", "refrigerator", "microwave", "shelf", "curtain"}
)
FURNITURE = frozenset(
    {"bed", "dresser", "nightstand", "desk", "table", "dining table", "chair", "armchair"}
    | {"sofa", "couch", "stool", "bench", "tv", "lamp", "mirror", "rug", "plant", "fan"}
    | {"speaker", "mini fridge", "picture frame", "computer monitor", "potted plant"}
)


def layer_of(label: str, kind: str) -> str:
    """structure (walls, doors, stairs, fixed appliances), furniture (moves rarely), or carried
    (things a person picks up). Doors and the rest of structure are the map's landmarks."""
    name = label.lower()
    if name in STRUCTURE:
        return "structure"
    if kind == "landmark" or name in FURNITURE:
        return "furniture"
    return "carried"


@dataclass
class Pin:
    id: int
    label: str
    kind: str  # object | landmark
    pos: list[float]  # map frame, metres
    ts: float  # epoch seconds, last seen
    first_ts: float
    seen: int = 1
    score: float = 0.0
    near: list[str] = field(default_factory=list)  # landmark labels within ~1.5 m
    thumb: str = ""  # base64 jpeg
    source: str = ""
    layer: str = ""
    placed: bool = True  # False: a home-map pin not yet placed in this walk's frame

    def to_json(self, include_thumb: bool = True) -> dict:
        d = asdict(self)
        if not include_thumb:
            d.pop("thumb")
        return d


SURE_SCORE = 0.3  # a personal item's local sighting this sure stands without SAM 3.1
SAM_SOURCE = "sam-3.1"  # the source of a SAM 3.1 sighting (Sam31Finder.name)


def confirmed(score: float, source: str) -> bool:
    """A personal-item sighting that stands: SAM 3.1 named it, or the local model was sure.
    (SAM's own floor is 0.4, so a SAM sighting folded into a weak pin lifts it past this.)"""
    return score >= SURE_SCORE or source == SAM_SOURCE


MOVED_AFTER_S = 600.0  # a lone sighting this much newer than the firm pin means it moved
MOVED_SCORE = 0.3  # ...as long as the detector was fairly sure
LONE_REACH = 2.0  # a lone sighting from the same visit folds in from this many merge radii


class PinStore:
    """Merges repeat sightings: the same label within `radius` metres is the same thing."""

    def __init__(self, radius: float = 0.5, landmark_radius: float = 1.3, items=()):
        self.radius, self.landmark_radius = radius, landmark_radius
        self.pins: dict[int, Pin] = {}
        self._next = 1
        # Personal items whose pins must be confirmed to be answered or routed to. A rule at
        # read time: a saved weak pin (the keyboard as "keys") stays in the house untouched.
        self.items = frozenset(items)

    def stands(self, p: Pin) -> bool:
        return p.label not in self.items or confirmed(p.score, p.source)

    def observe(
        self,
        label: str,
        kind: str,
        pos: list[float],
        ts: float,
        score: float = 0.0,
        near: list[str] | None = None,
        thumb: str = "",
        source: str = "",
    ) -> Pin:
        best, best_d = None, math.inf
        for p in self.pins.values():
            if p.label == label and p.kind == kind:
                d = math.dist(p.pos, pos)
                if d < best_d:
                    best, best_d = p, d
        if best is not None and best_d <= (
            self.radius if kind == "object" else self.landmark_radius
        ):
            k = min(0.5, 1.0 / (best.seen + 1))
            best.pos = [a + k * (b - a) for a, b in zip(best.pos, pos, strict=True)]
            best.ts, best.seen = ts, best.seen + 1
            if score >= best.score * 0.8 and thumb:
                best.thumb = thumb
            best.score = max(best.score, score)
            if near:
                best.near = near
            return best
        pin = Pin(
            self._next,
            label,
            kind,
            list(pos),
            ts,
            ts,
            1,
            score,
            near or [],
            thumb,
            source,
            layer_of(label, kind),
        )
        self.pins[pin.id] = pin
        self._next += 1
        return pin

    def latest(self, label: str, kind: str = "object") -> Pin | None:
        """Where it was last seen, so a moved object answers with its new place: the newest pin
        with two sightings, or one the crop check scored firmly (score >= 0.5), unless a fair
        lone sighting came long after it. Only when no pin of the label is firm does the
        most-seen lone sighting stand in. A personal item's pin that does not stand is skipped:
        better "not seen" than a walk to the keyboard."""
        cands = [
            p for p in self.pins.values() if p.label == label and p.kind == kind and self.stands(p)
        ]
        firm = [p for p in cands if p.seen >= 2 or p.score >= 0.5]
        if firm:
            best = max(firm, key=lambda p: p.ts)
            # A moved object is often seen once in its new place before the old place is
            # forgotten: a fair sighting long after the firm pin is where it is now.
            fair = [p for p in cands if p.score >= MOVED_SCORE]
            fresh = max(fair, key=lambda p: p.ts, default=best)
            if fresh.ts - best.ts > MOVED_AFTER_S:
                return fresh
            return best
        return max(cands, key=lambda p: (p.seen, p.ts)) if cands else None

    def glimpse(self, label: str, after_ts: float, kind: str = "object") -> Pin | None:
        """The newest weak lone sighting long after `after_ts`: too unsure to move the pin, but
        fresher than an hours-old memory, so worth offering as a maybe."""
        weak = [
            p
            for p in self.pins.values()
            if p.label == label
            and p.kind == kind
            and p.seen == 1
            and p.score < MOVED_SCORE
            and p.ts - after_ts > MOVED_AFTER_S
            and self.stands(p)
        ]
        return max(weak, key=lambda p: p.ts, default=None)

    def merge(self, radius: float, landmark_radius: float | None = None) -> None:
        """Fold pins of one label that sit within `radius` of each other into the better seen
        one (position averaged by sightings). Map drift splits one object into several pins
        that the live radius, tuned for a single window, does not join."""
        lm = radius * 2.6 if landmark_radius is None else landmark_radius
        order = sorted(self.pins.values(), key=lambda p: -p.seen)
        keep: list[Pin] = []

        def reach(k: Pin, p: Pin) -> float:
            r = radius if p.kind == "object" else lm
            # A weak lone sighting from the same visit is drift, not a second one; a lone one
            # long after is kept, since `latest` reads it as the object having moved.
            lone = p.kind == "object" and p.seen == 1 and p.score < 0.5
            return r * LONE_REACH if lone and abs(p.ts - k.ts) <= MOVED_AFTER_S else r

        for p in order:
            host = next(
                (
                    k
                    for k in keep
                    if k.label == p.label
                    and k.kind == p.kind
                    and math.dist(k.pos, p.pos) <= reach(k, p)
                ),
                None,
            )
            if host is None:
                keep.append(p)
                continue
            n = host.seen + p.seen
            host.pos = [
                (a * host.seen + b * p.seen) / n for a, b in zip(host.pos, p.pos, strict=True)
            ]
            host.seen, host.first_ts = n, min(host.first_ts, p.first_ts)
            if p.ts > host.ts:
                host.ts = p.ts
            if p.score > host.score:
                host.score, host.thumb = p.score, p.thumb or host.thumb
            host.near = host.near or p.near
        self.pins = {p.id: p for p in keep}

    def labels(self, kind: str = "object") -> list[str]:
        return sorted({p.label for p in self.pins.values() if p.kind == kind})

    def replace_all(self, pins: list[dict]) -> None:
        self.pins = {p["id"]: Pin(**p) for p in pins}
        for p in self.pins.values():
            p.layer = p.layer or layer_of(p.label, p.kind)
        self._next = max(self.pins, default=0) + 1
