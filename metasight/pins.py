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

    def __init__(self, radius: float = 0.6):
        self.radius = radius
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
        if best is not None and best_d <= self.radius:
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
        cands = [p for p in self.pins.values() if p.label == label and p.kind == kind]
        return max(cands, key=lambda p: p.ts) if cands else None

    def labels(self, kind: str = "object") -> list[str]:
        return sorted({p.label for p in self.pins.values() if p.kind == kind})

    def replace_all(self, pins: list[dict]) -> None:
        self.pins = {p["id"]: Pin(**p) for p in pins}
        self._next = max(self.pins, default=0) + 1
