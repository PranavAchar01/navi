"""Directional panning for the belt: one object is felt on the two motors nearest its bearing,
never on all of them.

Eight motors sit 45 degrees apart, clockwise from straight ahead. An object at bearing b lies
between motor i = floor(b / 45) and the next one; its power is split between the two by angle
(constant-power pan: gains cos and sin of a quarter turn, so total power stays the same as it
moves round the belt). An object at 30 degrees right is motor 1 strong (0.87) and motor 0 light
(0.50). Intensity follows distance through `Params`, whose defaults are the settings picked by the
simulated-user sweep in docs/haptics-sim.md.

Several objects: `select` keeps the nearest one and, if there is one, a second that is at least
`sep_deg` away from it. Rule for who wins, in order: nearest first; objects within `tie_m` of each
other in distance rank by how close they are to straight ahead (the one you would walk into);
a second object closer than `sep_deg` to the winner is read as part of it and dropped.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

from .belt import MODES

N = 8
STEP = 360.0 / N


@dataclass(frozen=True)
class Target:
    bearing_deg: float  # clockwise from straight ahead, any range
    dist_m: float


@dataclass(frozen=True)
class Params:
    off_m: float = 1.5  # silent at or beyond this distance
    fast_m: float = 1.6  # slow pulse between off_m and here, fast pulse inside
    solid_m: float = 0.7  # solid inside this
    floor: float = 0.2  # intensity at off_m
    ceil: float = 1.0  # intensity at solid_m
    gamma: float = 1.5  # >1 keeps the far end quiet and ramps late
    pan: str = "linear"  # constant_power | linear | nearest
    k_targets: int = 1
    sep_deg: float = 45.0
    tie_m: float = 0.15


DEFAULT = Params()


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def pan_gains(bearing_deg: float, pan: str = "constant_power") -> tuple[int, float, int, float]:
    """(motor a, gain a, motor b, gain b) for a bearing; b is the next motor clockwise."""
    b = bearing_deg % 360.0
    a = int(b // STEP) % N
    f = (b - (b // STEP) * STEP) / STEP  # 0 at motor a, 1 at the next
    if pan == "nearest":
        return (a, 1.0, (a + 1) % N, 0.0) if f < 0.5 else (a, 0.0, (a + 1) % N, 1.0)
    if pan == "linear":
        return a, 1.0 - f, (a + 1) % N, f
    theta = f * math.pi / 2
    return a, math.cos(theta), (a + 1) % N, math.sin(theta)


def level(dist_m: float, p: Params = DEFAULT) -> float:
    """Intensity 0..1 for a distance: floor at off_m rising to ceil at solid_m."""
    if not math.isfinite(dist_m) or dist_m >= p.off_m:
        return 0.0
    if dist_m <= p.solid_m:
        return p.ceil
    x = (p.off_m - dist_m) / (p.off_m - p.solid_m)
    return p.floor + (p.ceil - p.floor) * x**p.gamma


def mode_for(dist_m: float, p: Params = DEFAULT) -> int:
    if not math.isfinite(dist_m) or dist_m >= p.off_m:
        return MODES["off"]
    if dist_m >= p.fast_m:
        return MODES["slow"]
    if dist_m >= p.solid_m:
        return MODES["fast"]
    return MODES["solid"]


def select(targets: list[Target], p: Params = DEFAULT) -> list[Target]:
    """The one or two objects that get motors. See the module docstring for the rule."""
    live = [t for t in targets if math.isfinite(t.dist_m) and t.dist_m < p.off_m]
    live.sort(key=lambda t: (round(t.dist_m / p.tie_m), abs(wrap(t.bearing_deg))))
    chosen: list[Target] = []
    for t in live:
        if all(abs(wrap(t.bearing_deg - c.bearing_deg)) >= p.sep_deg for c in chosen):
            chosen.append(t)
        if len(chosen) >= p.k_targets:
            break
    return chosen


def plan_targets(targets: list[Target], p: Params = DEFAULT) -> list[tuple[int, int]]:
    """Per-motor (mode, intensity 0..255) for the selected objects. A motor shared by two objects
    takes the louder one, with that object's pulse mode."""
    best = [(0.0, MODES["off"])] * N
    for t in select(targets, p):
        lv, mode = level(t.dist_m, p), mode_for(t.dist_m, p)
        a, ga, b, gb = pan_gains(t.bearing_deg, p.pan)
        for m, g in ((a, ga), (b, gb)):
            if g > 0 and lv * g > best[m][0]:
                best[m] = (lv * g, mode)
    return [(mode, int(round(255 * i))) if i > 0 else (MODES["off"], 0) for i, mode in best]


def plan_pan(
    targets: list[Target],
    known: list[bool],
    path_m: float,
    ttc_s: float,
    drop_m: float,
    p: Params = DEFAULT,
    hold: Hold | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """Same output as `belt.plan` (8 x (mode, intensity), flags), but panned. With `hold`, the
    objects first go through that bearing hysteresis (one cue that stays put).

    Closing fast (time to contact under 1.2 s) on something within 2.5 m ahead raises the pulse
    one step on the front motors, as `belt.plan` does; a sector with no trusted data and no cue is
    "unknown" (no buzz, the UI says so).
    """
    ahead = hold(targets, p) if hold else list(targets)
    if math.isfinite(path_m):
        ahead.append(Target(0.0, path_m))
    cmd = plan_targets(ahead, p)
    if ttc_s < 1.2:
        for m in (0, 1, N - 1):
            mode, inten = cmd[m]
            if inten and min(t.dist_m for t in ahead) < 2.5:
                cmd[m] = (min(MODES["solid"], max(mode, MODES["slow"]) + 1), max(inten, 204))
    for i, ok in enumerate(known):
        if not ok and cmd[i][0] == MODES["off"]:
            cmd[i] = (MODES["unknown"], 0)
    flags = 1 if drop_m < 1.5 else 0
    return cmd, flags


def cluster_strips(
    dists: list[float], bearings: list[float], gap_m: float = 0.4, p: Params = DEFAULT
) -> list[Target]:
    """Objects from depth strips ordered left to right: neighbouring strips within `gap_m` of the
    run's nearest are one object, placed at the distance-weighted bearing of its nearest strips."""
    out: list[Target] = []
    run: list[int] = []

    def close() -> None:
        if not run:
            return
        near = min(dists[i] for i in run)
        core = [i for i in run if dists[i] <= near + 0.15]
        w = [1.0 / max(dists[i], 0.1) for i in core]
        bearing = sum(wi * bearings[i] for wi, i in zip(w, core, strict=True)) / sum(w)
        out.append(Target(bearing, near))

    for i, d in enumerate(dists):
        if not math.isfinite(d) or d >= p.off_m:
            close()
            run = []
            continue
        if run and abs(d - min(dists[j] for j in run)) > gap_m:
            close()
            run = []
        run.append(i)
    close()
    return out


def strip_bearings(n: int, hfov_deg: float) -> list[float]:
    """Pinhole bearing of the centre of each of n equal-width image strips."""
    t = math.tan(math.radians(hfov_deg) / 2)
    return [math.degrees(math.atan((2 * (k + 0.5) / n - 1) * t)) for k in range(n)]


def scene_targets(
    strips: list[float],
    hfov_deg: float,
    map_sectors: list[float],
    map_known: list[bool],
    p: Params = DEFAULT,
) -> list[Target]:
    """Targets for one frame: fine objects from the live depth strips across the camera's field
    of view, and one coarse target at the sector centre for each side/back sector the map covers.
    """
    out = cluster_strips(strips, strip_bearings(len(strips), hfov_deg), p=p)
    half = hfov_deg / 2
    for i, (d, ok) in enumerate(zip(map_sectors, map_known, strict=True)):
        centre = wrap(i * STEP)
        if ok and math.isfinite(d) and abs(centre) > half:
            out.append(Target(centre, d))
    return out


class Hold:
    """Bearing hysteresis for the single panned cue (k_targets = 1), so it does not hop between
    motors frame to frame. State: the object currently felt.

    - It keeps its object until another is closer by `switch_m` (nearest-wins flips on ties).
    - Its bearing moves only when the object has moved more than `band_deg` from where the cue
      sits, then follows at the band's edge.
    - It switches off only `release_m` beyond the off distance, so a wall at the edge does not
      blink.
    - If it vanishes it stays for `keep` more frames, then goes quiet or hands over.
    The distance is always the live one, so a closing obstacle still ramps up and a solid cue is
    never held back."""

    def __init__(
        self,
        switch_m: float = 0.3,
        band_deg: float = 10.0,
        keep: int = 2,
        match_deg: float = 25.0,
        release_m: float = 0.3,
    ):
        self.switch_m, self.band_deg, self.keep, self.match_deg = (
            switch_m,
            band_deg,
            keep,
            match_deg,
        )
        self.release_m = release_m
        self.cur: Target | None = None
        self.miss = 0

    def __call__(self, targets: list[Target], p: Params = DEFAULT) -> list[Target]:
        picks = select(targets, p)
        cand = picks[0] if picks else None
        if self.cur is not None:
            near = [
                t
                for t in targets
                if math.isfinite(t.dist_m)
                and t.dist_m < p.off_m + self.release_m
                and abs(wrap(t.bearing_deg - self.cur.bearing_deg)) <= self.match_deg
            ]
            same = min(
                near, key=lambda t: abs(wrap(t.bearing_deg - self.cur.bearing_deg)), default=None
            )
            if same is not None:
                self.miss = 0
                if cand is not None and cand.dist_m < same.dist_m - self.switch_m:
                    self.cur = cand
                else:
                    off = wrap(same.bearing_deg - self.cur.bearing_deg)
                    move = (
                        off - math.copysign(self.band_deg, off) if abs(off) > self.band_deg else 0.0
                    )
                    self.cur = Target(
                        wrap(self.cur.bearing_deg + move), min(same.dist_m, p.off_m - 0.01)
                    )
            else:
                self.miss += 1
                closer = cand is not None and cand.dist_m < self.cur.dist_m - self.switch_m
                if closer or self.miss > self.keep:
                    self.cur, self.miss = cand, 0
        else:
            self.cur = cand
        return [self.cur] if self.cur else []


class Persist:
    """Depth strips must stay near for `n` frames in a row before they count: each strip reports the
    farthest of its last n readings, so a one-frame phantom never reaches a motor. A real obstacle
    is delayed by (n - 1) sensor frames."""

    def __init__(self, n: int = 1):
        self.n = max(1, n)
        self.hist: deque[list[float]] = deque(maxlen=self.n)

    def __call__(self, strips: list[float]) -> list[float]:
        self.hist.append(list(strips))
        return [max(col) for col in zip(*self.hist, strict=True)]


GUIDE_ALIGNED_DEG = 10.0
GUIDE_SOFT, GUIDE_FIRM = 90, 170  # drive 0..255: always below the obstacle cues (>= 204 close in)


def guide_cmd(bearing_deg: float) -> list[tuple[int, int]]:
    """The Guide cue for a waypoint at `bearing_deg` (positive right, as `nav.Guidance`): the
    "guide" pattern (two soft taps a second, unlike every obstacle pattern) on the two motors
    nearest the bearing, linear panning. Aligned within 10 degrees: the front motor alone, faint
    ("on track"). The further round the turn, the firmer, 90 up to 170 of 255, so it never
    reaches the strength of a close obstacle."""
    cmd = [(MODES["off"], 0)] * N
    b = bearing_deg % 360.0
    if abs(wrap(bearing_deg)) <= GUIDE_ALIGNED_DEG:
        cmd[0] = (MODES["guide"], GUIDE_SOFT)
        return cmd
    amp = GUIDE_SOFT + (GUIDE_FIRM - GUIDE_SOFT) * min(1.0, abs(wrap(bearing_deg)) / 90.0)
    a, ga, c, gc = pan_gains(b, "linear")
    for m, g in ((a, ga), (c, gc)):
        if g > 0.05:
            cmd[m] = (MODES["guide"], int(round(amp * g)))
    return cmd


def with_guide(
    cmd: list[tuple[int, int]], bearing_deg: float, override: bool = True
) -> tuple[list[tuple[int, int]], bool]:
    """Overlay the Guide cue on an obstacle command; returns (new command, guide shown).

    Stop always overrides Guide: while any motor is solid (something about to be touched) Guide is
    silent everywhere, so the wearer feels one message. Otherwise Guide only takes motors that are
    off, unknown or on the slow far-field pulse; a fast cue is never replaced.
    `override=False` exists for the simulator's control arm only."""
    if override and any(m == MODES["solid"] for m, _ in cmd):
        return list(cmd), False
    out = list(cmd)
    g = guide_cmd(bearing_deg)
    shown = False
    for i, (gm, gi) in enumerate(g):
        if gm == MODES["off"]:
            continue
        if not override or out[i][0] in (MODES["off"], MODES["unknown"], MODES["slow"]):
            out[i] = (gm, gi)
            shown = True
    return out, shown


def with_params(p: Params, **kw) -> Params:
    return replace(p, **kw)


class Tracker:
    """Prototype (not wired live): panning from tracked objects instead of per-frame clusters.

    Each frame's depth objects are matched to tracks by bearing (gate `gate_deg`); a track's
    bearing is smoothed (`alpha`), its distance is the live one. A new object must be seen
    `confirm` frames before it gets a motor, unless it is inside `p.solid_m` + 0.3 m (a close
    object is never delayed), and a track survives `coast` missed frames. Used as `hold=` in
    `plan_pan`: it returns the confirmed tracks as Targets and `select` picks the nearest."""

    def __init__(
        self, gate_deg: float = 15.0, alpha: float = 0.4, confirm: int = 3, coast: int = 2
    ):
        self.gate_deg, self.alpha, self.confirm, self.coast = gate_deg, alpha, confirm, coast
        self.tracks: list[dict] = []

    def __call__(self, targets: list[Target], p: Params = DEFAULT) -> list[Target]:
        live = [t for t in targets if math.isfinite(t.dist_m) and t.dist_m < p.off_m + 0.3]
        free = list(live)
        for tr in self.tracks:
            best = min(free, key=lambda t: abs(wrap(t.bearing_deg - tr["b"])), default=None)
            if best is not None and abs(wrap(best.bearing_deg - tr["b"])) <= self.gate_deg:
                free.remove(best)
                tr["b"] = wrap(tr["b"] + self.alpha * wrap(best.bearing_deg - tr["b"]))
                tr["d"], tr["hits"], tr["miss"] = best.dist_m, tr["hits"] + 1, 0
            else:
                tr["miss"] += 1
        self.tracks = [tr for tr in self.tracks if tr["miss"] <= self.coast]
        self.tracks += [{"b": t.bearing_deg, "d": t.dist_m, "hits": 1, "miss": 0} for t in free]
        return [
            Target(tr["b"], tr["d"] if tr["miss"] == 0 else min(tr["d"], p.off_m - 0.01))
            for tr in self.tracks
            if tr["hits"] >= self.confirm or tr["d"] < p.solid_m + 0.3
        ]
