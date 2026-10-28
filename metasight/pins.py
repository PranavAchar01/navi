"""Object pins: where things were last seen in the metric map."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


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

    def to_json(self, include_thumb: bool = True) -> dict:
        d = asdict(self)
        if not include_thumb:
            d.pop("thumb")
        return d


class PinStore:
    """Merges repeat sightings: the same label within `radius` metres is the same thing."""

    def __init__(self, radius: float = 0.5, landmark_radius: float = 1.3):
        self.radius, self.landmark_radius = radius, landmark_radius
        self.pins: dict[int, Pin] = {}
        self._next = 1

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
        pin = Pin(self._next, label, kind, list(pos), ts, ts, 1, score, near or [], thumb, source)
        self.pins[pin.id] = pin
        self._next += 1
        return pin

    def latest(self, label: str, kind: str = "object") -> Pin | None:
        """The pin of this label seen most often (a stray detection loses to the real thing),
        the newest among equals."""
        cands = [p for p in self.pins.values() if p.label == label and p.kind == kind]
        firm = [p for p in cands if p.seen >= 2 or p.score >= 0.5]
        pool = firm or cands
        return max(pool, key=lambda p: (p.seen, p.ts)) if pool else None

    def merge(self, radius: float, landmark_radius: float | None = None) -> None:
        """Fold pins of one label that sit within `radius` of each other into the better seen
        one (position averaged by sightings). Map drift splits one object into several pins
        that the live radius, tuned for a single window, does not join."""
        lm = radius * 2.6 if landmark_radius is None else landmark_radius
        order = sorted(self.pins.values(), key=lambda p: -p.seen)
        keep: list[Pin] = []
        for p in order:
            r = radius if p.kind == "object" else lm
            host = next(
                (
                    k
                    for k in keep
                    if k.label == p.label and k.kind == p.kind and math.dist(k.pos, p.pos) <= r
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
        self._next = max(self.pins, default=0) + 1
