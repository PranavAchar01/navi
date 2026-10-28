"""Which way the head points between map poses.

Map poses arrive every 2 to 3 s, describe a keyframe taken seconds earlier, and their
orientation is noisy, so pointing the belt at a remembered item from the last pose alone swings
the cue around. The head's own turning is tracked continuously instead: the gyro when the client
sends one (glasses, phone), otherwise the sideways image shift of the live camera. Each map pose
is matched against the turn at its own keyframe time, which gives the offset between the turn
and the map; the heading is the turn now plus that offset. A new pose moves the offset a step
at a time instead of replacing it, and a pose that looks at the floor or ceiling is ignored for
heading.

The image shift measures the turn only up to a lens-dependent factor (the ultra-wide room
camera read about 1.3x the map's turn), so that factor is learned from pairs of map poses.
"""

from __future__ import annotations

import bisect
import math
from collections import deque

import cv2
import numpy as np

GYRO_FRESH_S = 0.5  # gyro samples newer than this win over the image shift
MAX_PITCH_FWD_Y = 0.8  # map forward vector more vertical than this has no usable heading
MAP_BLEND = 0.35  # share of the gap to a new map heading closed per pose once tracking
HISTORY_S = 20.0  # how far back a keyframe time can be matched
SHIFT_W = 160  # image width the shift is measured at
MIN_RESPONSE = 0.08  # phase correlation peak below this is texture-free noise
GAIN_MIN_TURN = math.radians(20)  # pose pairs with less image turn than this say little
GAIN_RANGE = (0.5, 1.5)
JUMP_M = 1.5  # a map position this far from the last one is a relocalisation: snap, don't blend
SIGHT_BLEND = 0.3  # share of the gap to the camera's bearing closed per frame the item is seen


def wrap(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def image_yaw(prev: np.ndarray, cur: np.ndarray, hfov_deg: float) -> float | None:
    """Head yaw to the left (radians) between two grey frames, from their horizontal shift at
    the image centre. Turning left moves the scene right in the image (positive dx)."""
    (dx, _dy), resp = cv2.phaseCorrelate(prev, cur)
    if resp < MIN_RESPONSE:
        return None
    f = 0.5 * prev.shape[1] / math.tan(math.radians(hfov_deg) / 2)
    return math.atan2(dx, f)


class Heading:
    def __init__(self, hfov_deg: float):
        self.hfov_deg = hfov_deg
        self.pos: np.ndarray | None = None
        self.turn = 0.0  # head turn to the left since start, radians, never wrapped
        self.raw = 0.0  # the image-shift part of `turn` before the gain
        self.hist: deque[tuple[float, float, float]] = deque()  # (time, turn, raw)
        self.gain = 1.0  # map turn per image-shift turn
        self.fit = [0.0, 0.0]  # running sums of raw*map and raw*raw over pose pairs
        self.last_anchor: tuple[float, float] | None = None  # (map heading, raw) at keyframe
        self.offset: float | None = None  # map heading minus turn
        self.gyro_t = -math.inf
        self.prev: np.ndarray | None = None

    @property
    def yaw(self) -> float | None:
        """Map-frame heading, radians, atan2(fwd_x, fwd_z)."""
        return None if self.offset is None else wrap(self.turn + self.offset)

    def _add(self, d: float, now: float, raw: float = 0.0) -> None:
        self.turn += d
        self.raw += raw
        self.hist.append((now, self.turn, self.raw))
        while self.hist and now - self.hist[0][0] > HISTORY_S:
            self.hist.popleft()

    def at(self, t: float | None) -> tuple[float, float]:
        """(turn, raw) at time `t`, or now."""
        if t is None or not self.hist:
            return self.turn, self.raw
        i = bisect.bisect_right(self.hist, t, key=lambda h: h[0])
        _, turn, raw = self.hist[max(0, i - 1)]
        return turn, raw

    # ---------------------------------------------------------------- turning
    def gyro(self, rate: np.ndarray, up: np.ndarray, dt: float, now: float) -> None:
        """`rate` in rad/s and `up` in the same device axes; turning left is positive."""
        self.gyro_t = now
        if 0 < dt < 0.2:
            self._add(float(rate @ up) * dt, now)

    def shift_gray(self, bgr: np.ndarray) -> np.ndarray:
        """Grey, small, float frame for `image_turn` (cheap enough for the safety thread)."""
        h = round(bgr.shape[0] * SHIFT_W / bgr.shape[1])
        small = cv2.resize(bgr, (SHIFT_W, h), interpolation=cv2.INTER_AREA)
        return np.float32(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

    def image_turn(self, gray: np.ndarray, now: float) -> None:
        prev, self.prev = self.prev, gray
        if prev is None or prev.shape != gray.shape or now - self.gyro_t < GYRO_FRESH_S:
            return
        d = image_yaw(prev, gray, self.hfov_deg)
        if d is not None:
            self._add(self.gain * d, now, d)

    # ---------------------------------------------------------------- map
    def anchor(self, pos: np.ndarray, fwd: np.ndarray, t: float | None = None) -> bool:
        """Take a map pose for the keyframe at time `t`. False when its heading was unusable
        (the position is kept)."""
        new_pos = np.asarray(pos, dtype=float)
        jumped = self.pos is not None and float(np.linalg.norm(new_pos - self.pos)) > JUMP_M
        self.pos = new_pos
        f = np.asarray(fwd, dtype=float)
        f = f / (np.linalg.norm(f) or 1.0)
        if abs(f[1]) > MAX_PITCH_FWD_Y:
            return False
        heading = math.atan2(f[0], f[2])
        turn, raw = self.at(t)
        self._learn_gain(heading, raw)
        new = heading - turn
        if self.offset is None or jumped:
            self.offset = wrap(new)
        else:
            self.offset = wrap(self.offset + MAP_BLEND * wrap(new - self.offset))
        return True

    def box_bearing(self, box: tuple[float, float, float, float]) -> float:
        """Bearing in degrees (positive to the right) of a box's centre, box in 0..1 units."""
        cx = 0.5 * (box[0] + box[2])
        return math.degrees(math.atan((2 * cx - 1) * math.tan(math.radians(self.hfov_deg) / 2)))

    def sight(self, point: np.ndarray, bearing_deg: float) -> None:
        """The camera sees `point` at `bearing_deg`: turn the heading part of the way to agree."""
        rel = self.to(point)
        if rel is None or self.offset is None:
            return
        # A larger heading (turned left) puts the point further right.
        self.offset = wrap(self.offset + SIGHT_BLEND * math.radians(bearing_deg - rel[1]))

    def to(self, point: np.ndarray) -> tuple[float, float] | None:
        """(horizontal metres, bearing degrees, positive to the right) from the head to `point`."""
        yaw = self.yaw
        if self.pos is None or yaw is None:
            return None
        v = np.asarray(point, dtype=float) - self.pos
        fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        bearing = math.degrees(math.atan2(float(v @ right), float(v @ fwd)))
        return float(math.hypot(v[0], v[2])), bearing

    def _learn_gain(self, heading: float, raw: float) -> None:
        prev, self.last_anchor = self.last_anchor, (heading, raw)
        if prev is None:
            return
        d_raw = raw - prev[1]
        if abs(d_raw) < GAIN_MIN_TURN:
            return
        # The map turn, unwrapped next to the current guess (turns over 180 degrees per pair).
        d_map = self.gain * d_raw + wrap(heading - prev[0] - self.gain * d_raw)
        self.fit[0] += d_raw * d_map
        self.fit[1] += d_raw * d_raw
        self.gain = min(max(self.fit[0] / self.fit[1], GAIN_RANGE[0]), GAIN_RANGE[1])
