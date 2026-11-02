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
from scipy.signal import fftconvolve, lfilter

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

HRTF = Path(__file__).parent / "hrtf" / "mit_kemar_compact.npz"
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


# ---- the sounds ----


def _envelope(n: int, ramp_frac: float = 0.25) -> np.ndarray:
    ramp = max(1, int(n * ramp_frac))
    edge = np.minimum(np.arange(n), n - 1 - np.arange(n))
    return np.where(edge >= ramp, 1.0, 0.5 * (1 - np.cos(np.pi * edge / ramp)))


def tick_sound(kind: str) -> np.ndarray:
    """The phone's beep (SpatialBeeper.tone): a sine with raised-cosine edges, mono."""
    hz, dur, gain = TICK_KINDS[kind]
    t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
    return gain * np.sin(2 * np.pi * hz * t) * _envelope(len(t))


def _pluck(hz: float, dur: float = 0.32) -> np.ndarray:
    """A soft mallet note: the fundamental plus a quiet fourth partial, fast attack, quick decay."""
    t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * hz * t) + 0.25 * np.sin(2 * np.pi * 4 * hz * t) * np.exp(-t / 0.03)
    attack = np.minimum(1.0, t / 0.004)
    fade = np.minimum(1.0, (dur - t) / (0.1 * dur))  # no click at the cut
    return tone * attack * np.exp(-t / 0.09) * fade


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
    itd = HEAD_RADIUS_M / SPEED_OF_SOUND * (abs(lat) + abs(math.sin(lat)))
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
    ticks; a new or closer band sounds at once, a repeat waits out its interval."""
    times = np.asarray(times, float)
    events: list[Event] = []
    next_beacon = -math.inf
    arrived = target is None or not beacon
    last_tick: tuple[float, int, int] | None = None  # (time, hazard index, band rank)
    rank = {"slow": 1, "fast": 2, "stop": 3}
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
            last_tick = None
            continue
        if (
            last_tick is not None
            and last_tick[1] == i
            and last_tick[2] >= rank[b]
            and t - last_tick[0] < tick_period(d.ground_m)
        ):
            continue
        events.append(Event(float(t), b, d))
        last_tick = (float(t), i, rank[b])
    return events


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


def write_wav(path: Path | str, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
