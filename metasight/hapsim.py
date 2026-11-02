"""A simulated blind walker, for tuning the haptic map. SIMULATED USER MODEL, NOT A HUMAN STUDY.

The walker sees nothing. It knows the walls of the house (a person living there does) and follows
a route through the doors, but the furniture has been moved: obstacles reach it only as belt
cues, through a skin model. The belt side is the real code: depth strips become objects
(`pan.scene_targets`), objects become motor commands (`pan.plan_pan` / `belt.plan`), commands run
through the belt's pulse patterns and motor spin-up, then a perception model turns motor
vibration back into "something at this bearing and distance" with limited resolution. Steering
rules and body parameters are fixed across every setting that is compared, so differences come
from the haptic map alone.

Perception model (assumed from the psychophysics literature, not measured on this belt):
  - intensity JND about 20% (Weber), as lognormal noise on each motor's felt amplitude
  - detection floor: vibration under 0.10 of full drive is not felt
  - torso localisation noise: 6 degrees on the inferred bearing (motors are 45 degrees / 12 cm
    apart; adjacent motors fuse into one sensation between them, which is what panning uses)
  - motor spin-up 75 ms first-order lag, both ways (ERM: 50-100 ms)
  - tactile adaptation: a motor held on loses up to 30% of felt strength over 20 s
  - cue reading takes an 800 ms window (a slow pulse is only known after it has pulsed)
  - end-to-end latency: sensor frame 40 ms + processing 25 ms + link 15 ms = 80 ms by default;
    the bench has never measured camera-frame-to-motor (docs/ENGINEERING-REVIEW.md item 9), so
    these come from the loop rates in PROGRESS.md and the sweep reports sensitivity to it
  - human reaction 250 ms between feeling a change and acting on it
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from . import pan
from .belt import MODES, plan
from .geometry import sector_index

DT = 0.02  # physics and belt step
SENSE_DT = 0.04  # a sensor frame (25 fps)
HFOV = 105.0
STRIPS = 15
MAX_RANGE = 6.0
BODY_R = 0.25
V_MAX = 0.9  # m/s
TURN_MAX = math.radians(120)  # rad/s
REACT_S = 0.25
WINDOW_S = 0.8


@dataclass(frozen=True)
class Belt:
    """Everything about the haptic map that the sweep changes."""

    mapping: pan.Params = pan.DEFAULT
    scheme: str = "panned"  # panned (pan.plan_pan) | sector (belt.plan, the unpanned original)
    slow_hz: float = 1.5
    fast_hz: float = 4.0
    latency_s: float = 0.080
    spinup_s: float = 0.075
    guide: str = "none"  # none | on | no_override: a waypoint cue (pan.with_guide)
    hold: bool = False  # bearing hysteresis on the panned cue (pan.Hold)
    planner: bool = False  # steer by nav.Navigator's waypoint bearing instead of the sim's BFS
    persist: int = 1  # frames a strip must stay near (pan.Persist); 1 = off
    layout: str = "eight"  # eight motors, one per cell | diag4: firmware/belt's four diagonals
    # diag4: cp = constant power (firmware) | max3 = max of own cell and neighbours | lin = linear
    fold: str = "cp"


@dataclass(frozen=True)
class Skin:
    jnd: float = 0.20
    floor: float = 0.10
    loc_sd_deg: float = 6.0
    adapt_max: float = 0.30
    adapt_s: float = 20.0
    nav_drift_deg: float = 0.0  # walker's waypoint sense drift per sqrt(s); 0 = knows the way


DEFAULT_SKIN = Skin()


@dataclass
class Room:
    rects: np.ndarray  # (n, 4) x0, y0, x1, y1; walls first, then furniture
    n_walls: int
    start: tuple[float, float, float]  # x, y, heading (rad, 0 = +x, counter-clockwise)
    goal: tuple[float, float]
    size: tuple[float, float]
    seed: int = 0


# ------------------------------------------------------------------ rooms
def make_room(seed: int) -> Room:
    """A two-room flat, 9 x 5.5 m: a wall with a 0.9 m door, then 4 to 7 moved furniture pieces."""
    rng = np.random.default_rng(seed)
    w, h, t = 9.0, 5.5, 0.1
    walls = [(-t, -t, w + t, 0), (-t, h, w + t, h + t), (-t, 0, 0, h), (w, 0, w + t, h)]
    wx = rng.uniform(3.8, 5.2)
    door_y = rng.uniform(1.2, h - 2.1)
    walls += [(wx - t / 2, 0, wx + t / 2, door_y), (wx - t / 2, door_y + 0.9, wx + t / 2, h)]
    start = (0.8, rng.uniform(1.0, h - 1.0), 0.0)
    goal = (w - 0.9, rng.uniform(1.0, h - 1.0))
    furniture: list[tuple[float, float, float, float]] = []
    for _ in range(int(rng.integers(4, 8))):
        for _try in range(30):
            fw, fh = rng.uniform(0.4, 1.3, 2)
            x, y = rng.uniform(0.3, w - fw - 0.3), rng.uniform(0.3, h - fh - 0.3)
            box = (x, y, x + fw, y + fh)
            if (
                _clear(box, start[:2], 0.9)
                and _clear(box, goal, 0.7)
                and not _near_door(box, wx, door_y)
            ):
                furniture.append(box)
                break
    rects = np.array(walls + furniture, dtype=float)
    room = Room(rects, len(walls), start, goal, (w, h), seed)
    # every room needs a route about 1 m wide, so a reactive walker has room to go round
    return room if route(room, True, inflate=0.5) else make_room(seed + 10_000)


def _clear(box, pt, r) -> bool:
    x0, y0, x1, y1 = box
    return not (x0 - r < pt[0] < x1 + r and y0 - r < pt[1] < y1 + r)


def _near_door(box, wx, door_y) -> bool:
    x0, y0, x1, y1 = box
    return x0 - 0.6 < wx < x1 + 0.6 and y1 > door_y - 0.5 and y0 < door_y + 1.4


def _grid(room: Room, with_furniture: bool, res: float = 0.1, inflate: float = 0.32):
    nx, ny = int(room.size[0] / res), int(room.size[1] / res)
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    blocked = np.zeros((nx, ny), bool)
    rects = room.rects if with_furniture else room.rects[: room.n_walls]
    for i, (x0, y0, x1, y1) in enumerate(rects):
        r = 0.32 if i < room.n_walls else inflate  # the door stays open; furniture gaps widen
        blocked |= (gx > x0 - r) & (gx < x1 + r) & (gy > y0 - r) & (gy < y1 + r)
    return blocked, res


def route(room: Room, with_furniture: bool, inflate: float = 0.32) -> list[tuple[float, float]]:
    """Shortest path start to goal on a grid (BFS, 8-connected), thinned to waypoints."""
    blocked, res = _grid(room, with_furniture, inflate=inflate)
    s = (int(room.start[0] / res), int(room.start[1] / res))
    g = (int(room.goal[0] / res), int(room.goal[1] / res))
    blocked[s] = blocked[g] = False
    prev = {s: None}
    q = deque([s])
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while q:
        c = q.popleft()
        if c == g:
            break
        for dx, dy in moves:
            n = (c[0] + dx, c[1] + dy)
            if (
                0 <= n[0] < blocked.shape[0]
                and 0 <= n[1] < blocked.shape[1]
                and not blocked[n]
                and n not in prev
            ):
                prev[n] = c
                q.append(n)
    if g not in prev:
        return []
    path, c = [], g
    while c is not None:
        path.append(((c[0] + 0.5) * res, (c[1] + 0.5) * res))
        c = prev[c]
    return path[::-1]


def path_length(pts) -> float:
    return float(sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False)))


def waypoints(room: Room) -> list[tuple[float, float]]:
    """What the walker remembers: a route through the doors that ignores furniture, thinned to
    points roughly 1.2 m apart."""
    pts = route(room, with_furniture=False)
    out, last = [], pts[0]
    for p in pts[1:-1]:
        if math.dist(p, last) >= 1.2:
            out.append(p)
            last = p
    return [*out, room.goal]


# ------------------------------------------------------------------ sensing
def cast(rects: np.ndarray, ox: float, oy: float, angles: np.ndarray) -> np.ndarray:
    """Range along each ray (radians, world frame) to the nearest rectangle; slab method."""
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx0, tx1 = (rects[:, 0] - ox) / dx, (rects[:, 2] - ox) / dx
        ty0, ty1 = (rects[:, 1] - oy) / dy, (rects[:, 3] - oy) / dy
    tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
    tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
    tmin = np.nan_to_num(tmin, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    tmax = np.nan_to_num(tmax, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    hit = (tmax >= np.maximum(tmin, 0)) & (tmax > 0)
    t = np.where(hit, np.maximum(tmin, 0), np.inf)
    return np.minimum(t.min(axis=1), MAX_RANGE)


def sense(room: Room, x, y, heading, rng) -> tuple[list[float], list[float]]:
    """(depth strips across the camera's field of view, 8 map sectors), metres, with 4% noise.
    Bearings are clockwise from the heading; world angles run counter-clockwise."""
    b = np.radians(pan.strip_bearings(STRIPS, HFOV))
    sub = np.array([-0.5, 0.0, 0.5]) * math.radians(HFOV / STRIPS)
    r = cast(room.rects, x, y, (heading - (b[:, None] + sub[None, :])).ravel())
    strips = r.reshape(STRIPS, 3).min(axis=1) * rng.lognormal(0, 0.04, STRIPS)
    sec = heading - np.radians(np.arange(8)[:, None] * 45 + np.linspace(-22, 22, 5)[None, :])
    sectors = cast(room.rects, x, y, sec.ravel()).reshape(8, 5).min(axis=1)
    far = MAX_RANGE - 1e-6
    return (
        [float(d) if d < far else math.inf for d in strips],
        [float(d) if d < far else math.inf for d in sectors],
    )


def command(
    belt: Belt, strips, sectors, ttc, goal_b: float = 0.0, hold=None
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], bool]:
    """(final command, the obstacle command under it, whether Guide was shown)."""
    obstacle = _obstacle_cmd(belt, strips, sectors, ttc, hold)
    cmd, shown = obstacle, False
    if belt.guide != "none":
        cmd, shown = pan.with_guide(obstacle, goal_b, override=belt.guide == "on")
    return cmd, obstacle, shown


def _planner(room: Room):
    """nav.Navigator on the room, routed to the goal. The sim has x right, y up, heading
    counter-clockwise from +x; nav has forward (sin yaw, cos yaw), so z = -y and yaw = pi/2 + h."""
    from metasight import nav

    rects = [
        (x0, -y1, x1, -y0, 2.5 if i < room.n_walls else 0.9)
        for i, (x0, y0, x1, y1) in enumerate(room.rects)
    ]
    w, hgt = room.size
    plan = nav.plan_from_rects(rects, -0.3, -hgt - 0.3, w + 0.3, 0.3)
    n = nav.Navigator(plan)
    n.set_goal([room.goal[0], 0.0, -room.goal[1]], "keys")
    return n


def _obstacle_cmd(belt: Belt, strips, sectors, ttc, hold=None) -> list[tuple[int, int]]:
    inf = math.inf
    if belt.scheme == "sector":
        # The original: eight sector distances, front three from the camera, one motor each.
        n = len(strips)
        thirds = [min(strips[i * n // 3 : (i + 1) * n // 3]) for i in range(3)]
        merged = list(sectors)
        merged[7], merged[0], merged[1] = thirds
        path = min(strips[n // 2 - 1 : n // 2 + 2])
        cmd, _ = plan(merged, [True] * 8, path, ttc, inf)
        return cmd
    targets = pan.scene_targets(strips, HFOV, sectors, [True] * 8, belt.mapping)
    path = min(strips[STRIPS // 2 - 1 : STRIPS // 2 + 2])
    cmd, _ = pan.plan_pan(targets, [True] * 8, path, ttc, inf, belt.mapping, hold)
    return cmd


def duty(mode: int, inten: int, t: float, belt: Belt) -> float:
    """Motor drive 0..1 at time t: the belt_core.h patterns, with the rates as parameters."""
    if mode == MODES["slow"]:
        per = 1 / belt.slow_hz
        on = (t % per) < min(0.2, 0.3 * per)
    elif mode == MODES["fast"]:
        per = 1 / belt.fast_hz
        on = (t % per) < per / 2
    elif mode == MODES["solid"]:
        on = True
    elif mode == MODES["find"]:
        on = (t % 0.4) < 0.04
    elif mode == MODES["talk"]:
        on = (t % 1.5) < 0.05 or 0.13 <= (t % 1.5) < 0.18 or 0.26 <= (t % 1.5) < 0.31
    elif mode == MODES["guide"]:
        on = (t % 1.0) < 0.07 or 0.15 <= (t % 1.0) < 0.22
    else:
        on = False
    return inten / 255 if on else 0.0


# Physical motor bearings (deg, clockwise from front) per layout. diag4 is the built belt: four
# modules on the diagonals; each plays its own cell and 1/sqrt(2) of the two beside it (cp, as in
# firmware/belt/belt.ino), so "front" buzzes both front motors and "front-right" only one.
MOTOR_DEG = {"eight": [k * 45.0 for k in range(8)], "diag4": [45.0, 135.0, 225.0, 315.0]}


def to_motors(drive: np.ndarray, layout: str, fold: str = "cp") -> np.ndarray:
    """Eight cell drives to what the physical motors play. lin is linear panning: a cell between
    two motors gives each half, so the pair sums back to the cell's drive."""
    if layout == "eight":
        return drive
    if fold == "cp":  # constant power: a cell between two motors gives each 1/sqrt(2)
        return np.array(
            [
                min(
                    1.0,
                    math.sqrt(drive[c] ** 2 + 0.5 * (drive[c - 1] ** 2 + drive[(c + 1) % 8] ** 2)),
                )
                for c in (1, 3, 5, 7)
            ]
        )
    if fold == "lin":
        return np.array(
            [min(1.0, drive[c] + 0.5 * (drive[c - 1] + drive[(c + 1) % 8])) for c in (1, 3, 5, 7)]
        )
    return np.array([max(drive[c - 1], drive[c], drive[(c + 1) % 8]) for c in (1, 3, 5, 7)])


# ------------------------------------------------------------------ walker
@dataclass
class Result:
    success: bool
    time_s: float
    path_m: float
    shortest_m: float
    collisions: int
    contact_s: float
    min_clear_m: float
    false_buzzes: int = 0  # cue onsets with nothing real within range of that motor
    guide_frames: int = 0  # frames with a guide cue wanted (a goal and the guide on)
    stop_frames: int = 0  # of those, frames where an obstacle cue was solid
    stop_overridden: int = 0  # of those, frames where Guide yielded to Stop
    stop_violations: int = 0  # frames where a solid motor was changed by the overlay (must be 0)

    @property
    def efficiency(self) -> float:
        return self.shortest_m / self.path_m if self.success and self.path_m > 0 else 0.0


@dataclass
class _Felt:
    """Skin state: motor amplitude (after spin-up), adaptation, and a short history."""

    amp: np.ndarray = field(default_factory=lambda: np.zeros(8))
    adapt: np.ndarray = field(default_factory=lambda: np.zeros(8))
    hist: deque = field(default_factory=lambda: deque(maxlen=int((REACT_S + WINDOW_S) / DT) + 2))


def wrap_deg(a: float) -> float:
    return (a + 180) % 360 - 180


def perceive(window: np.ndarray, belt: Belt, skin: Skin, rng) -> list[tuple[float, float]]:
    """From the last 800 ms of felt amplitude per motor (rows = time) to the (bearing deg,
    distance m) pairs the walker believes. Runs of adjacent active motors are one object, its
    bearing the amplitude-weighted position (inverting the pan law for two motors). On diag4 the
    walker knows the code: two neighbours together mean the direction between them, at the
    strength of the stronger one (the firmware takes the max, it does not pan)."""
    p = belt.mapping
    deg = MOTOR_DEG[belt.layout]
    n = len(deg)
    gap = 360.0 / n
    peak = window.max(axis=0) * rng.lognormal(0, skin.jnd / 0.95 / math.sqrt(2), n)
    on_frac = (window > skin.floor).mean(axis=0)
    active = peak > skin.floor
    if not active.any():
        return []
    first = next((k for k in range(n) if not active[k]), None)
    order = list(range(n)) if first is None else [(first + 1 + k) % n for k in range(n)]
    runs: list[list[int]] = []
    cur: list[int] = []
    for m in order:
        if active[m]:
            cur.append(m)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    seen = []
    for run in runs:
        amps = peak[run]
        if len(run) == 1:
            ang, lv = deg[run[0]], float(amps[0])
        elif len(run) == 2:
            a, b = float(amps[0]), float(amps[1])
            if belt.layout == "diag4":
                if belt.fold == "cp":
                    f, lv = math.atan2(b, a) / (math.pi / 2), math.hypot(a, b)
                else:
                    f = b / (a + b)
                    lv = a + b if belt.fold == "lin" else max(a, b)
            elif p.pan == "linear":
                f, lv = b / (a + b), a + b
            elif p.pan == "nearest":
                f, lv = (0.0 if a >= b else 1.0), max(a, b)
            else:
                f, lv = math.atan2(b, a) / (math.pi / 2), math.hypot(a, b)
            ang, lv = deg[run[0]] + f * gap, min(lv, 1.0)
        else:  # a wide object, or two merged: centre of mass about the strongest motor
            k = int(np.argmax(amps))
            lo, hi = max(0, k - 1), min(len(run), k + 2)
            w = amps[lo:hi]
            offs = (np.arange(lo, hi) - k) * gap
            ang, lv = deg[run[k]] + float((w * offs).sum() / w.sum()), float(w.max())
        ang = wrap_deg(ang + rng.normal(0, skin.loc_sd_deg))
        seen.append((ang, _invert(lv, float(on_frac[run].max()), p)))
    return seen


def _invert(lv: float, frac_on: float, p: pan.Params) -> float:
    """Distance the walker infers: the pulse pattern gives the band, the strength the place in
    it (the walker knows the cue coding)."""
    if frac_on > 0.85:
        lo, hi = p.solid_m * 0.4, p.solid_m
    elif frac_on > 0.4:
        lo, hi = p.solid_m, p.fast_m
    else:
        lo, hi = p.fast_m, p.off_m
    x = min(1.0, max(0.0, (lv - p.floor) / max(1e-6, p.ceil - p.floor))) ** (1 / p.gamma)
    guess = p.off_m - x * (p.off_m - p.solid_m)  # stronger means nearer
    return float(min(hi, max(lo, guess)))


def steer(seen, goal_b: float, belt: Belt, side: int) -> tuple[float, float, int]:
    """(desired relative heading rad, clockwise positive; speed m/s; committed side) from what
    the belt says and where the next waypoint is. The same rule for every setting."""
    off = belt.mapping.off_m
    vec = np.array([math.cos(math.radians(goal_b)), math.sin(math.radians(goal_b))])
    ahead = math.inf
    for b, d in seen:
        if abs(b) < 100:
            k = max(0.0, 1 - d / off)
            if abs(b) <= 8:
                side = side or (1 if goal_b < 0 else -1)  # dead ahead: keep one side
                away = -side
            else:
                away = -1 if b > 0 else 1
            c = max(0.0, math.cos(math.radians(b)))
            vec = vec + np.array([-0.6 * k * c, 1.6 * k * away * c])
        if abs(b) < 35:
            ahead = min(ahead, d)
    want = math.atan2(vec[1], vec[0])
    speed = V_MAX * min(1.0, max(0.0, (ahead - 0.4) / 0.8))
    if ahead < 0.55:
        want = math.radians(70) * (side or 1)
    return want, speed, side


def run(
    room: Room,
    belt: Belt,
    skin: Skin = DEFAULT_SKIN,
    seed: int = 0,
    max_s: float = 90.0,
    phantom_rate: float = 0.0,
    sensor=None,
) -> Result:
    """One walk. `sensor(room, x, y, heading, rng)` replaces the 2D ray cast (hapsim3d renders
    depth instead); it returns (strips, sectors) like `sense`."""
    rng = np.random.default_rng(seed * 7919 + room.seed)
    x, y, heading = room.start
    wps = waypoints(room)
    shortest = path_length(route(room, True))
    nm = len(MOTOR_DEG[belt.layout])
    felt = _Felt(np.zeros(nm), np.zeros(nm))
    pending: deque[tuple[float, list]] = deque()
    live: list[tuple[int, int]] = [(0, 0)] * 8
    t, wi, dist, coll, contact_t = 0.0, 0, 0.0, 0, 0.0
    in_contact, clear, side, side_until = False, 9.0, 0, 0.0
    want, speed, last_speed, last_sense = 0.0, 0.0, 0.0, -1.0
    k_rise = 1 - math.exp(-DT / belt.spinup_s)
    lag = int(REACT_S / DT)
    persist = pan.Persist(belt.persist)
    hold = pan.Hold() if belt.hold else None
    drift = 0.0
    n = 0
    nav = _planner(room) if belt.planner else None
    gframes = sframes = sover = sviol = 0
    gtruth = 0.0
    fb, prev_on = 0, [False] * 8
    while t < max_s:
        if t - last_sense >= SENSE_DT - 1e-9:
            last_sense = t
            strips, sectors = (sensor or sense)(room, x, y, heading, rng)
            mid = min(strips[STRIPS // 2 - 1 : STRIPS // 2 + 2])
            ttc = mid / last_speed if last_speed > 0.05 else math.inf
            truth = list(strips), list(sectors)
            if phantom_rate > 0:  # depth error: a strip reads near for one frame (flying pixels)
                strips = [
                    float(rng.uniform(0.4, 1.8)) if rng.random() < phantom_rate else d
                    for d in strips
                ]
            gb = wrap_deg(-math.degrees(math.atan2(wps[wi][1] - y, wps[wi][0] - x) - heading))
            if nav is not None:
                g = nav.step([x, 0.0, -y], math.pi / 2 + heading)
                if g is not None:
                    gb = g.bearing_deg
                gtruth = gb
            cmd, obstacle, shown = command(belt, persist(strips), sectors, ttc, gb, hold)
            if belt.guide != "none":
                gframes += 1
                if any(m == MODES["solid"] for m, _ in obstacle):
                    sframes += 1
                    sover += not shown
                    sviol += any(
                        m == MODES["solid"] and cmd[k] != (m, i)
                        for k, (m, i) in enumerate(obstacle)
                    )
            # False buzz = a motor turns on while nothing real is within range of its 90 degree
            # span (its own sector and the strips that face it).
            on = [i > 0 for _, i in cmd]
            for m in range(8):
                if (
                    on[m]
                    and not prev_on[m]
                    and not _real_near(
                        truth, m, belt.mapping.off_m if belt.scheme == "panned" else 2.0
                    )
                ):
                    fb += 1
            prev_on = on
            pending.append((t + belt.latency_s, cmd))
        while pending and pending[0][0] <= t:
            live = pending.popleft()[1]
        drive = np.array([0.0 if m == MODES["guide"] else duty(m, i, t, belt) for m, i in live])
        drive = to_motors(drive, belt.layout, belt.fold)
        felt.amp += (drive - felt.amp) * k_rise
        felt.adapt += np.where(felt.amp > skin.floor, DT / skin.adapt_s, -2 * DT / skin.adapt_s)
        felt.adapt = np.clip(felt.adapt, 0, 1)
        felt.hist.append(felt.amp * (1 - skin.adapt_max * felt.adapt))
        if n % 5 == 0:  # decide at 10 Hz from what was felt REACT_S ago
            h = list(felt.hist)
            win = np.array(h[: max(1, len(h) - lag)][-int(WINDOW_S / DT) :])
            gx, gy = wps[wi]
            goal_true = wrap_deg(-math.degrees(math.atan2(gy - y, gx - x) - heading))
            if nav is not None:
                goal_true = wrap_deg(gtruth)
            # The walker's own sense of where the next waypoint is drifts without a cue (dead
            # reckoning, skin.nav_drift_deg per sqrt(s)); a felt Guide cue resets it to the
            # cue's bearing.
            if skin.nav_drift_deg > 0:
                drift += rng.normal(0, skin.nav_drift_deg * math.sqrt(0.1))
            gm = [(k, i) for k, (m, i) in enumerate(live) if m == MODES["guide"] and i > 0]
            if gm:
                w = sum(i for _, i in gm)
                ang = sum(((k if k <= 4 else k - 8) * 45.0) * i for k, i in gm) / w
                if skin.nav_drift_deg > 0:
                    drift = wrap_deg(ang + rng.normal(0, skin.loc_sd_deg) - goal_true)
            goal_b = wrap_deg(goal_true + drift)
            was = side
            want, speed, side = steer(perceive(win, belt, skin, rng), goal_b, belt, side)
            if side and not was:
                side_until = t + 2.5
            elif side and t > side_until:
                side = 0
        n += 1
        dpsi = max(-TURN_MAX * DT, min(TURN_MAX * DT, want))
        heading -= dpsi
        want -= dpsi
        v = speed * (0.5 if abs(want) > math.radians(60) else 1.0)
        nx, ny = x + v * DT * math.cos(heading), y + v * DT * math.sin(heading)
        hit = _push_out(room.rects, nx, ny)
        if hit is not None:
            nx, ny = hit
            coll += 0 if in_contact else 1
            in_contact, last_speed = True, 0.0
            contact_t += DT
        else:
            in_contact, last_speed = False, v
        dist += math.dist((x, y), (nx, ny))
        x, y = nx, ny
        t += DT
        if n % 10 == 0:
            clear = min(clear, _clearance(room.rects, x, y))
        extra = (gframes, sframes, sover, sviol)
        if nav is not None:
            if math.dist((x, y), room.goal) < 0.45:
                return Result(True, t, dist, shortest, coll, contact_t, clear, fb, *extra)
        elif math.dist((x, y), wps[wi]) < 0.45:
            wi += 1
            if wi >= len(wps):
                return Result(True, t, dist, shortest, coll, contact_t, clear, fb, *extra)
    return Result(
        False, t, dist, shortest, coll, contact_t, clear, fb, gframes, sframes, sover, sviol
    )


def _real_near(truth, m: int, off_m: float) -> bool:
    strips, sectors = truth
    bear = pan.strip_bearings(STRIPS, HFOV)
    centre = pan.wrap(m * 45.0)
    if sectors[m] < off_m:
        return True
    adj = [sectors[(m - 1) % 8], sectors[(m + 1) % 8]]
    if min(adj) < off_m:  # a panned cue for something between two motors is real
        return True
    return any(
        d < off_m for d, b in zip(strips, bear, strict=True) if abs(pan.wrap(b - centre)) < 45
    )


def _clearance(rects, x, y) -> float:
    dx = np.maximum(np.maximum(rects[:, 0] - x, x - rects[:, 2]), 0)
    dy = np.maximum(np.maximum(rects[:, 1] - y, y - rects[:, 3]), 0)
    return float(np.hypot(dx, dy).min()) - BODY_R


def _push_out(rects, x, y) -> tuple[float, float] | None:
    """If the body circle overlaps a rectangle, the nearest legal position; else None."""
    moved = False
    for _ in range(3):
        for x0, y0, x1, y1 in rects:
            cx, cy = min(max(x, x0), x1), min(max(y, y0), y1)
            d = math.hypot(x - cx, y - cy)
            if d < BODY_R:
                if d < 1e-9:
                    cx, cy, d = x, y, 1e-9
                x = cx + (x - cx) / d * BODY_R
                y = cy + (y - cy) / d * BODY_R
                moved = True
    return (x, y) if moved else None


def evaluate(
    belt: Belt, seeds: range | list[int], skin: Skin = DEFAULT_SKIN, per_room: int = 1
) -> list[Result]:
    out = []
    for s in seeds:
        room = make_room(s)
        for k in range(per_room):
            out.append(run(room, belt, skin, seed=k))
    return out


def cost(rs: list[Result]) -> float:
    """Lower is better: a failure costs 100, a collision 4, and every second 0.05 (so about 5
    for a 100 s crawl), averaged over runs."""
    return float(np.mean([100 * (not r.success) + 4 * r.collisions + 0.05 * r.time_s for r in rs]))


def summarize(rs: list[Result]) -> dict:
    ok = [r for r in rs if r.success]
    return {
        "runs": len(rs),
        "success": float(np.mean([r.success for r in rs])),
        "collisions": float(np.mean([r.collisions for r in rs])),
        "time_s": float(np.mean([r.time_s for r in ok])) if ok else math.nan,
        "efficiency": float(np.mean([r.efficiency for r in ok])) if ok else 0.0,
        "contact_s": float(np.mean([r.contact_s for r in rs])),
        "false_per_min": float(
            sum(r.false_buzzes for r in rs) / max(1e-9, sum(r.time_s for r in rs) / 60)
        ),
        "guide_frames": sum(r.guide_frames for r in rs),
        "stop_frames": sum(r.stop_frames for r in rs),
        "stop_overridden": sum(r.stop_overridden for r in rs),
        "stop_violations": sum(r.stop_violations for r in rs),
        "cost": cost(rs),
    }


__all__ = [
    "Belt",
    "Skin",
    "Room",
    "make_room",
    "run",
    "evaluate",
    "summarize",
    "cost",
    "replace",
    "sector_index",
]
