"""Spatial sound, the second channel next to the belt: the item being found plays from where it
is, and the nearest hazard ticks from where it is.

Two sounds, both sparse. The beacon is a soft pluck that plays only while a "where is my X"
guide runs. It repeats faster as the item gets closer and turns into a two-note figure when
the wearer faces it, as Soundscape's beacon does, then plays a short chime on arrival. A hazard
ticks only inside 2 m, and its rate and timbre follow the phone's beeps (ios/.../BeepCue.swift):
a faint tick from 2 to 1 m, a soft beep to 0.5 m, then a clear rapid beep.

Placement uses the MIT KEMAR compact head-related impulse responses (Gardner and Martin 1994,
free to use with credit; metasight/hrtf/README.md). method="pan" is the fallback with no
dataset: Woodworth time difference, a head-shadow low-pass on the far ear and a dull back.

This module is the reference for the rules (the phone mirrors them in BeaconCue.swift, the
unit tests pin them) and the offline renderer the simulation mixes into its video:
`render(times, positions, forwards, ups, target=..., hazards=[...])` gives stereo samples.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, lfilter, resample_poly

SAMPLE_RATE = 44_100  # the KEMAR set's rate
STEP_S = 0.04  # the phone's beat loop runs every 40 ms; the renderer decides on the same grid
HEAD_RADIUS_M = 0.0875
SPEED_OF_SOUND = 343.0

ARRIVE_M = 0.8  # the server ends a guide here (server.py _guide)
ALIGNED_DEG = 10.0  # pan.GUIDE_ALIGNED_DEG: facing the item
BEACON_FAR = (4.0, 1.4)  # (distance m, seconds between plucks)
BEACON_NEAR = (ARRIVE_M, 0.45)
BEACON_GAIN = 0.5
BEACON_HZ = 523.25  # C5; the aligned figure adds the G above
TICK_RANGE_M = 2.0  # hazards are silent at 2 m or more, as on the phone
# A warning that holds (no stronger band, any direction) settles to one beep a second after 3 s:
# the belt still holds it, and a stop beep five times a second for half a minute at a desk
# is the constant noise the wearer turns off.
SETTLE_AFTER_S = 3.0
SETTLED_PERIOD_S = 1.0

HRTF = Path(__file__).parent / "hrtf" / "mit_kemar_compact.npz"
# A real head's delay at our sounds' frequencies (0.5 to 2 kHz) runs above Woodworth's
# high-frequency figure; at 1.0 the simulated listener heard the pluck 13 deg toward the
# middle, at 1.15 7 deg; 1.25 overshoots into the stop beep's ambiguity (eval_localisation).
PAN_ITD_SCALE = 1.15
HRTF_GAIN = 0.78  # the set's front response is 0.96: this puts the front level with the pan


@dataclass(frozen=True)
class Direction:
    """Where a point sits around the head: azimuth degrees clockwise from straight ahead
    (positive right, +-180 behind), elevation degrees up, straight-line distance metres, and
    the distance along the floor, which the rules use (how far to walk, as on the belt)."""

    az_deg: float
    el_deg: float
    dist_m: float
    ground_m: float = math.nan


@dataclass(frozen=True)
class Hazard:
    """An obstacle as a point (hi=None) or an axis-aligned box lo..hi in world metres. The tick
    comes from the box's nearest point, so a long desk ticks from the edge you are walking at."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float] | None = None

    def nearest(self, p: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.lo, float)
        return lo if self.hi is None else np.clip(p, lo, np.asarray(self.hi, float))


def relative(head: np.ndarray, fwd: np.ndarray, up: np.ndarray, point: np.ndarray) -> Direction:
    """The point's direction from a head at `head` looking along `fwd` with `up` overhead. Any
    right-handed world works: right is fwd x up. fwd and up need not be unit or orthogonal."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    f = np.asarray(fwd, float) - np.dot(fwd, up) * up
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    v = np.asarray(point, float) - np.asarray(head, float)
    x, y, z = float(v @ right), float(v @ f), float(v @ up)
    return Direction(
        math.degrees(math.atan2(x, y)),
        math.degrees(math.atan2(z, math.hypot(x, y))),
        float(np.linalg.norm(v)),
        math.hypot(x, y),
    )


def head_vectors(
    pose: np.ndarray, forward=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position, forward and up from a 4x4 head-to-world matrix. The defaults are the OpenGL /
    ARKit camera (looks down -z, y up); pass the camera's own axes for other conventions."""
    r = pose[:3, :3]
    return pose[:3, 3], r @ np.asarray(forward, float), r @ np.asarray(up, float)


# ---- the rules (BeaconCue.swift mirrors these) ----


def _lerp(d: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    t = min(1.0, max(0.0, (a[0] - d) / (a[0] - b[0])))
    return a[1] + (b[1] - a[1]) * t


def beacon_period(dist_m: float) -> float:
    """Seconds between plucks: 1.4 s at 4 m or more, down to 0.45 s at arrival."""
    return _lerp(dist_m, BEACON_FAR, BEACON_NEAR)


def beacon_gain(dist_m: float) -> float:
    """Inverse distance from 1 m, held between a quarter and full: far stays audible."""
    return BEACON_GAIN * min(1.0, max(0.25, 1.0 / max(dist_m, 1e-3)))


def aligned(az_deg: float) -> bool:
    return abs(az_deg) <= ALIGNED_DEG


# (band name, frequency Hz, duration s, gain): BeepKind in BeepCue.swift
TICK_KINDS = {
    "slow": (2000.0, 0.012, 0.15),
    "fast": (880.0, 0.07, 0.4),
    "stop": (1320.0, 0.09, 0.8),
}


def band(dist_m: float) -> str | None:
    """ProximityBand(distance:): silent at 2 m or more, then slow, fast, stop below 0.5 m."""
    if not math.isfinite(dist_m) or dist_m >= TICK_RANGE_M:
        return None
    return "slow" if dist_m >= 1.0 else "fast" if dist_m >= 0.5 else "stop"


def tick_period(dist_m: float) -> float:
    """BeepCue.interval: 2.0 s at 2 m to 1.2 s at 1 m, 0.7 s to 0.4 s at 0.5 m, then 0.2 s."""
    b = band(dist_m)
    if b is None:
        return math.inf
    if b == "slow":
        return _lerp(dist_m, (2.0, 2.0), (1.0, 1.2))
    if b == "fast":
        return _lerp(dist_m, (1.0, 0.7), (0.5, 0.4))
    return 0.2


def repeat_period(period: float, held_s: float) -> float:
    """BeepScheduler: the band's interval, eased to one a second once a warning has held for
    SETTLE_AFTER_S. A new sector or a stronger band starts over at full pace."""
    return max(period, SETTLED_PERIOD_S) if held_s >= SETTLE_AFTER_S else period


class Repeats:
    """BeepScheduler's memory of the warning it is beeping, from the first beep until all is
    clear. At a desk the server's band flickers between fast and stop, and on the move the
    nearest sector hops as the head turns; restarting on each of those beeped five times a
    second. So only the first beep and a band worse than any yet sound at once; everything
    else, a new direction included, waits out the band's interval (each beep still comes
    from where the warning is now), eased once the warning has held."""

    RANK = {"slow": 1, "fast": 2, "stop": 3}

    def __init__(self) -> None:
        self.held = False
        self.since = self.last = -math.inf
        self.top = 0

    def clear(self) -> None:
        self.held = False

    def due(self, t: float, kind: str, period: float) -> bool:
        r = self.RANK[kind]
        if not self.held:
            self.held, self.since, self.top = True, t, r
        elif r > self.top:
            self.top = r
        elif t - self.last < repeat_period(period, t - self.since):
            return False
        self.last = t
        return True


# ---- the sounds ----


def _envelope(n: int, ramp_frac: float = 0.25) -> np.ndarray:
    ramp = max(1, int(n * ramp_frac))
    edge = np.minimum(np.arange(n), n - 1 - np.arange(n))
    return np.where(edge >= ramp, 1.0, 0.5 * (1 - np.cos(np.pi * edge / ramp)))


# Each beep carries the octave below at half strength. A pure 1320 or 2000 Hz tone gives the
# ear a delay it cannot place (a peak every period, or only the envelope): the simulated
# listener put the stop beep 69 deg off and on the wrong side a third of the time; with
# the octave, 6 deg (scripts/eval_localisation.py). 1.368 is the pair's peak, kept at 1.
SUB_OCTAVE = 0.5
SUB_PEAK = 1.368
# And a bright strike on the attack: partials 3 to 8 of the beep (below 18 kHz) at 1/k, 0.3 of
# the tone, gone in about 8 ms. A tone alone sits in one or two bands, where the pinna barely
# tells front from back: with a stranger's ears the simulated listener heard the fast and stop
# beeps in the mirrored hemisphere 45 and 30% of the time from a single beep; with the strike,
# 2 and 5% (front_back in spatial_listener). Lateral placement is unchanged (the delay is read
# below 1.2 kHz).
AIR = 0.3
AIR_TAU_S = 0.008
AIR_PARTIALS = range(3, 9)


def tick_sound(kind: str) -> np.ndarray:
    """The phone's beep (SpatialBeeper.tone): a sine and its octave below with raised-cosine
    edges, plus the strike on the attack; peak at gain, mono."""
    hz, dur, gain = TICK_KINDS[kind]
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    env = _envelope(n)
    tone = (np.sin(2 * np.pi * hz * t) + SUB_OCTAVE * np.sin(np.pi * hz * t)) / SUB_PEAK * env
    release = np.where(np.arange(n) < n // 2, 1.0, env)
    x = tone + _strike(hz, t) * release
    return gain * x / np.abs(x).max()


def _strike(hz: float, t: np.ndarray) -> np.ndarray:
    """The bright attack (AIR): partials 3 to 8 at 1/k, 1 ms in, gone in about 8 ms."""
    s = sum(np.sin(2 * np.pi * k * hz * t) / k for k in AIR_PARTIALS if k * hz < 18_000)
    return AIR * s * np.exp(-t / AIR_TAU_S) * np.minimum(1.0, t / 0.001)


def _pluck(hz: float, dur: float = 0.32) -> np.ndarray:
    """A soft mallet note: the fundamental plus a quiet fourth partial, fast attack, quick decay,
    and the strike (with it the simulated listener's static front/back confusions for the
    beacon fell from 28 to 12%)."""
    t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * hz * t) + 0.25 * np.sin(2 * np.pi * 4 * hz * t) * np.exp(-t / 0.03)
    attack = np.minimum(1.0, t / 0.004)
    fade = np.minimum(1.0, (dur - t) / (0.1 * dur))  # no click at the cut
    return tone * attack * np.exp(-t / 0.09) * fade + _strike(hz, t)


def beacon_sound(on_target: bool) -> np.ndarray:
    """One pluck off target; the pluck and the fifth above when the wearer faces the item."""
    a = _pluck(BEACON_HZ)
    if not on_target:
        return a
    b = _pluck(BEACON_HZ * 1.5)
    gap = int(0.11 * SAMPLE_RATE)
    out = np.zeros(gap + len(b))
    out[: len(a)] += a
    out[gap:] += 0.8 * b
    return out


def arrival_sound() -> np.ndarray:
    """Three rising notes (C, E, G) once, when the item is reached."""
    gap = int(0.09 * SAMPLE_RATE)
    notes = [_pluck(BEACON_HZ * r, 0.45) for r in (1.0, 1.26, 1.5)]
    out = np.zeros(2 * gap + len(notes[-1]))
    for i, n in enumerate(notes):
        out[i * gap : i * gap + len(n)] += n
    return 0.8 * out


# ---- placement ----


@cache
def _kemar() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(HRTF)
    return z["elev"].astype(float), z["azim"].astype(float), z["hrir"].astype(np.float64) / 32768


def hrir(az_deg: float, el_deg: float) -> np.ndarray:
    """(taps, 2) left/right impulse responses for a direction: the nearest measured elevation,
    blended between its two nearest azimuths. The compact set holds the right half (0..180,
    left and right ears); the left half is the mirror with the ears swapped."""
    elev, azim, ir = _kemar()
    el = elev[np.argmin(np.abs(elev - np.clip(el_deg, -40, 90)))]
    row = elev == el
    az_r, ir_r = azim[row], ir[row]
    a = abs(((az_deg + 180) % 360) - 180)
    order = np.argsort(az_r)
    az_r, ir_r = az_r[order], ir_r[order]
    if len(az_r) == 1:
        h = ir_r[0]
    else:
        j = int(np.clip(np.searchsorted(az_r, a), 1, len(az_r) - 1))
        w = (a - az_r[j - 1]) / max(az_r[j] - az_r[j - 1], 1e-9)
        h = (1 - np.clip(w, 0, 1)) * ir_r[j - 1] + np.clip(w, 0, 1) * ir_r[j]
    return h if ((az_deg + 180) % 360) - 180 >= 0 else h[:, ::-1]


def _one_pole(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    a = math.exp(-2 * math.pi * cutoff_hz / SAMPLE_RATE)
    return lfilter([1 - a], [1, -a], x)


def place_pan(mono: np.ndarray, az_deg: float) -> np.ndarray:
    """No dataset: Woodworth interaural delay, the far ear shadowed (quieter, duller), and a
    source behind dulled on both ears, so back is not the same as front."""
    az = math.radians(((az_deg + 180) % 360) - 180)
    lat = math.asin(max(-1.0, min(1.0, math.sin(az))))  # lateral angle, front/back folded
    itd = PAN_ITD_SCALE * HEAD_RADIUS_M / SPEED_OF_SOUND * (abs(lat) + abs(math.sin(lat)))
    shift = int(round(itd * SAMPLE_RATE))
    s = abs(math.sin(lat))
    near = mono * (1 + 0.3 * s)
    far = _one_pole(mono, 20_000 - 18_000 * s) * (1 - 0.5 * s)
    far = np.concatenate([np.zeros(shift), far])
    near = np.concatenate([near, np.zeros(shift)])
    if math.cos(az) < 0:  # behind: the pinna cuts the highs
        cut = 9_000 - 6_000 * -math.cos(az)
        near, far = _one_pole(near, cut), _one_pole(far, cut)
    left, right = (far, near) if az > 0 else (near, far)
    return np.stack([left, right], axis=1) * 0.75


def place(mono: np.ndarray, d: Direction, method: str = "hrtf") -> np.ndarray:
    """(n, 2) stereo for a mono sound at direction d (distance is applied by the caller)."""
    if method == "pan":
        return place_pan(mono, d.az_deg)
    h = hrir(d.az_deg, d.el_deg)
    return np.stack([fftconvolve(mono, h[:, 0]), fftconvolve(mono, h[:, 1])], axis=1) * HRTF_GAIN


# ---- the renderer ----


@dataclass(frozen=True)
class Event:
    """One sound in a render: when it starts, what it is, and where it came from."""

    t: float
    kind: str  # "beacon", "beacon_aligned", "arrive", "slow", "fast", "stop"
    where: Direction


def schedule(
    times: np.ndarray,
    positions: np.ndarray,
    forwards: np.ndarray,
    ups: np.ndarray,
    target: np.ndarray | None = None,
    hazards: list[Hazard] = (),
    beacon: bool = True,
    ticks: bool = True,
) -> list[Event]:
    """What plays when, decided every 40 ms like the phone. The beacon runs until the head is
    within 0.8 m of the target, then the arrival chime plays once. Only the nearest hazard
    ticks; a new or closer band sounds at once, a repeat waits out its interval (one a second
    once the warning has held 3 s)."""
    times = np.asarray(times, float)
    events: list[Event] = []
    next_beacon = -math.inf
    arrived = target is None or not beacon
    repeats = Repeats()
    for t in np.arange(times[0], times[-1], STEP_S):
        k = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1))
        head, fwd, up = positions[k], forwards[k], ups[k]
        if not arrived:
            d = relative(head, fwd, up, target)
            if d.ground_m < ARRIVE_M:
                arrived = True
                events.append(Event(float(t), "arrive", d))
            elif t >= next_beacon:
                kind = "beacon_aligned" if aligned(d.az_deg) else "beacon"
                events.append(Event(float(t), kind, d))
                next_beacon = t + beacon_period(d.ground_m)
        if not ticks or not hazards:
            continue
        near = [(relative(head, fwd, up, hz.nearest(head)), i) for i, hz in enumerate(hazards)]
        d, i = min(near, key=lambda x: x[0].ground_m)
        b = band(d.ground_m)
        if b is None:
            repeats.clear()
            continue
        if repeats.due(float(t), b, tick_period(d.ground_m)):
            events.append(Event(float(t), b, d))
    return events


STALE_S = 1.5  # BeaconScheduler / the beat loop: an old safety message plays nothing
SECTOR_NAMES = (
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
)
MODE_BAND = {"slow": "slow", "fast": "fast", "solid": "stop"}
BAND_PACE_M = {"slow": 1.5, "fast": 0.75, "stop": 0.3}  # mid-band distance for a bare cue


def take_warning(msg: dict) -> tuple[int, float | None, str] | None:
    """ProximityReading.warning on one safety message: (sector, metres, band) for the most
    urgent sector (strongest band, then ahead before the sides, then nearest). The front
    sector also counts the walking path, as ServerLink.proximity does."""
    rank = {"slow": 1, "fast": 2, "stop": 3}
    dist = list(msg.get("sectors") or [])
    cues = msg.get("cues") or []
    modes = [c.get("mode", "unknown") for c in cues]
    if dist and msg.get("path") is not None:
        dist[0] = min(dist[0] if dist[0] is not None else math.inf, msg["path"])
    best = None
    for i, mode in enumerate(modes[:8]):
        if mode == "unknown":
            continue
        d = dist[i] if i < len(dist) else None
        bands = [b for b in (band(d) if d is not None else None, MODE_BAND.get(mode)) if b]
        if not bands:
            continue
        b = max(bands, key=rank.get)
        key = (rank[b], i == 0, -(d if d is not None else math.inf))
        if best is None or key > best[0]:
            best = (key, (i, d, b))
    return best[1] if best else None


# How long a band must hold before it beeps (BeepScheduler.rise). Tuned on take1 and its
# judge replays by scripts/score_ticks.py: every beep inside a warning under 0.5 s goes (2.1
# a minute on the live config), rear beeps on the real take drop from 37% to 27%, and a real
# warning beeps 0.1 to 0.3 s later. The phone's display waits longer (WarningFilter.rise).
RISE_S = {"stop": 0.1, "fast": 0.2, "slow": 0.3}


class Onset:
    """A warning beeps only once its band has held for rise[band]: a one-frame phantom never
    sounds. The band played is the strongest one that has held long enough."""

    RANK = {"slow": 1, "fast": 2, "stop": 3}

    def __init__(self, rise: dict[str, float]) -> None:
        self.rise = rise
        self.since: dict[str, float] = {}

    def clear(self) -> None:
        self.since = {}

    def gate(
        self, t: float, w: tuple[int, float | None, str] | None
    ) -> tuple[int, float | None, str] | None:
        r = self.RANK[w[2]] if w else 0
        for b, rank in self.RANK.items():
            if rank <= r:
                self.since.setdefault(b, t)
            else:
                self.since.pop(b, None)
        ok = [b for b in self.since if t - self.since[b] >= self.rise.get(b, 0.0) - 1e-9]
        if not w or not ok:
            return None
        return (w[0], w[1], max(ok, key=self.RANK.get))


def schedule_take(
    msgs: list[dict],
    beacon: bool = True,
    ticks: bool = True,
    start: float | None = None,
    end: float | None = None,
    rise: dict[str, float] | None = RISE_S,
) -> list[Event]:
    """What the phone plays over a recorded take (judge replay or live jsonl), from its
    safety messages alone: the guide beacon at the guide's bearing and distance, the arrival
    notes when "found" turns on, and the hazard beep from the most urgent sector. Times are
    seconds from start (wall clock; default the first message), up to end. rise=None beeps
    on the first frame of a warning."""
    safety = [m for m in msgs if m.get("type") == "safety"]
    if not safety:
        return []
    t0 = min(m["wall"] for m in msgs) if start is None else start
    walls = np.array([m["wall"] - t0 for m in safety])
    events: list[Event] = []
    next_beacon = -math.inf
    was_found = False
    repeats = Repeats()
    onset = Onset(rise) if rise else None
    for t in np.arange(walls[0], walls[-1] + STEP_S, STEP_S):
        k = int(np.searchsorted(walls, t, side="right") - 1)
        m = safety[k]
        if t - walls[k] > STALE_S:
            next_beacon = -math.inf
            repeats.clear()
            if onset:
                onset.clear()
            continue
        g = m.get("guide") or {}
        fix = g.get("bearing") is not None and g.get("dist") is not None
        found = bool(m.get("found"))
        if beacon and found and not was_found:
            here = Direction(g.get("bearing") or 0.0, 0.0, 1.0, 1.0)
            events.append(Event(float(t), "arrive", here))
        was_found = found
        if not (beacon and fix and not found):
            next_beacon = -math.inf
        elif t >= next_beacon:
            d = Direction(float(g["bearing"]), 0.0, float(g["dist"]), float(g["dist"]))
            kind = "beacon_aligned" if aligned(d.az_deg) else "beacon"
            events.append(Event(float(t), kind, d))
            next_beacon = t + beacon_period(d.ground_m)
        w = take_warning(m) if ticks else None
        if onset:
            w = onset.gate(float(t), w)
        if w is None:
            repeats.clear()
            continue
        i, dist, b = w
        # A cue raised by the server without a distance beats at its band's own pace.
        dist_m = dist if dist is not None and band(dist) == b else BAND_PACE_M[b]
        if not repeats.due(float(t), b, tick_period(dist_m)):
            continue
        az = ((i * 45 + 180) % 360) - 180
        events.append(Event(float(t), b, Direction(float(az), 0.0, dist_m, dist_m)))
    stop = math.inf if end is None else end - t0
    return [e for e in events if 0 <= e.t <= stop]


def render_take(
    msgs: list[dict],
    method: str = "hrtf",
    beacon: bool = True,
    ticks: bool = True,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """(n, 2) float32 stereo of what the phone plays over a recorded take, t = 0 at start
    (default its first message), so it lines up with a clip cut from the same take. With an
    end it is exactly end - start long."""
    events = schedule_take(msgs, beacon, ticks, start, end)
    t0 = min(m["wall"] for m in msgs) if start is None else start
    total = (max(m["wall"] for m in msgs) + 1.0 if end is None else end) - t0
    return mix(events, total, method)


def _sound(kind: str) -> tuple[np.ndarray, bool]:
    """The mono sound for an event and whether distance scales it (ticks carry distance by
    their rate, as on the phone, so they are not scaled)."""
    if kind == "beacon":
        return beacon_sound(False), True
    if kind == "beacon_aligned":
        return beacon_sound(True), True
    if kind == "arrive":
        return arrival_sound(), False
    return tick_sound(kind), False


def mix(events: list[Event], total_s: float, method: str = "hrtf") -> np.ndarray:
    """Stereo float32 samples (n, 2) for a list of events, soft-limited."""
    n = int(total_s * SAMPLE_RATE)
    out = np.zeros((n, 2))
    for e in events:
        mono, scaled = _sound(e.kind)
        gain = beacon_gain(e.where.ground_m) if scaled else 1.0
        st = place(mono * gain, e.where, method)
        i = int(e.t * SAMPLE_RATE)
        if i >= n:
            continue
        m = min(len(st), n - i)
        out[i : i + m] += st[:m]
    return np.tanh(out).astype(np.float32)


def render(
    times,
    positions,
    forwards,
    ups,
    target=None,
    hazards: list[Hazard] = (),
    method: str = "hrtf",
    beacon: bool = True,
    ticks: bool = True,
) -> np.ndarray:
    """The simulation's sound track: a head pose track (times in s from 0, positions, forward
    and up vectors, all (N, 3) in world metres) plus the target item's position and the
    hazards, to stereo float32 samples (N_samples, 2) at SAMPLE_RATE covering times[0]..[-1]."""
    times = np.asarray(times, float)
    pos, fwd, up = (np.asarray(a, float) for a in (positions, forwards, ups))
    tgt = None if target is None else np.asarray(target, float)
    events = schedule(times, pos, fwd, up, tgt, list(hazards), beacon, ticks)
    shifted = [Event(e.t - times[0], e.kind, e.where) for e in events]
    return mix(shifted, float(times[-1] - times[0]), method)


def write_wav(path: Path | str, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """16-bit stereo; a rate other than 44.1 kHz resamples (48 kHz for video)."""
    if rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        samples = resample_poly(samples, rate // g, SAMPLE_RATE // g, axis=0)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
