"""Which camera frames the map gets: a frame is kept when it adds something, not because its
time slot came up.

A frame adds something when the view has moved on from the last kept one: the head turned (the
picture slid sideways by more than `turn_deg` worth of field of view) or the scene no longer
lines up with it (the wearer stepped forward, or something in it changed). Standing still adds
nothing. The map also needs overlap between neighbours to stitch them, so a view is kept after
`max_gap_s` even if it barely moved, and never more often than `min_gap_s`. Among the frames that
qualify within one short wait, the sharpest wins: walking blurs most of them.

It reads a small grey thumbnail only, so it needs no pose and costs well under a millisecond.
"""

from __future__ import annotations

import cv2
import numpy as np

THUMB = (64, 48)


def thumb(gray: np.ndarray) -> np.ndarray:
    t = cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
    return (t - t.mean()) / (t.std() + 1e-6)


def moved(a: np.ndarray, b: np.ndarray, hfov_deg: float) -> tuple[float, float]:
    """(turn in degrees, how well b still lines up with a after undoing the turn, 0..1)."""
    win = cv2.createHanningWindow(THUMB, cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a.copy(), b.copy(), win)  # it windows its inputs in place
    turn = abs(dx) / THUMB[0] * hfov_deg
    sx, sy = int(round(dx)), int(round(dy))
    h, w = a.shape
    if abs(sx) >= w - 8 or abs(sy) >= h - 8:
        return turn, 0.0
    # compare only where the two views overlap once the slide is undone
    pa = a[max(0, -sy) : h + min(0, -sy), max(0, -sx) : w + min(0, -sx)]
    pb = b[max(0, sy) : h + min(0, sy), max(0, sx) : w + min(0, sx)]
    pa, pb = (pa - pa.mean()) / (pa.std() + 1e-6), (pb - pb.mean()) / (pb.std() + 1e-6)
    match = float((pa * pb).mean())
    return turn, max(0.0, match)


class KeyframePolicy:
    def __init__(
        self,
        hfov_deg: float,
        turn_deg: float = 15.0,
        min_match: float = 0.75,
        min_gap_s: float = 0.4,
        max_gap_s: float = 4.0,
        settle_s: float = 0.3,
    ):
        self.hfov, self.turn_deg, self.min_match = hfov_deg, turn_deg, min_match
        self.min_gap, self.max_gap, self.settle = min_gap_s, max_gap_s, settle_s
        self.last: np.ndarray | None = None
        self.last_t = -1e9
        self.since: float | None = None  # when the current view first qualified
        self.best = None  # (sharp, ts, payload, thumb)
        self.why = ""

    def novel(self, t: np.ndarray, now: float) -> str:
        if self.last is None:
            return "first"
        if now - self.last_t >= self.max_gap:
            return "gap"
        turn, match = moved(self.last, t, self.hfov)
        if turn >= self.turn_deg:
            return "turn"
        if match < self.min_match:
            return "new view"
        return ""

    def offer(self, gray: np.ndarray, sharp: float, now: float, payload=None):
        """Feed every frame. Returns the payload of the frame to keep, or None."""
        if now - self.last_t < self.min_gap:
            return None
        t = thumb(gray)
        why = self.novel(t, now)
        if not why and self.since is None:
            return None
        if self.since is None:
            self.since, self.why = now, why
        if self.best is None or sharp > self.best[0]:
            self.best = (sharp, now, payload, t)
        if why != "first" and now - self.since < self.settle:
            return None
        _, ts, kept, self.last = self.best
        self.last_t, self.since, self.best = ts, None, None
        return kept
