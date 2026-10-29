"""Getting somewhere with the map, and what the map tells the reflex loop.

Floor plan: the home map's occupancy (home.py: 10 cm cells, y up, floor near y = 0 but tilted by
drift) flattened to columns. A column is an obstacle when it holds surface between 0.2 and 1.9 m
above its local floor, and its top height is kept, so a coffee table (top under 0.6 m) is known
to be low. Columns the map never saw are passable at a price: the map is never complete.

Routes: one cost-to-go field per goal (Dijkstra from the goal over every free cell), so each
frame's direction is a lookup from wherever the wearer is now; drifting off the route needs no
replan. Cells closer than 0.25 m to an obstacle are closed (half a body); between 0.25 and 0.6 m
they cost more the closer they are. A fixed 0.32 m inflation would close a 0.6 m doorway; this
keeps it open and puts the route through its middle. Narrow passages on the route are flagged so
guidance can say "door ahead".

Priors for the reflex loop (`priors`, then `apply` on the belt command):
  (a) mapped obstacles the camera cannot see now: outside its field of view, or low and close
      enough to sit under the bottom edge of the frame (only the last raises a cue: beside and
      behind, memory's map sectors already cue);
  (b) new vs known: a live reading the map explains (a wall walked along) gets a calm cue, one
      nearer than anything mapped is new and gets a stronger one;
  (c) contact ahead: the body swept along the current heading and turn for 2.5 s, against the
      map, so a turn into a table warns before the table is in frame.
All of them need a fresh position: `pos_err_m` widens the tolerances, and past `MAX_POS_ERR_M`
the priors stand down (the belt then behaves as it does without them).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .belt import MODES
from .geometry import FAST_M, OFF_M, SLOW_M, cue_for_distance
from .home import OCC_M

CELL = OCC_M
MIN_H_M, MAX_H_M = 0.2, 1.9  # surface in this band above the floor is in the way
LOW_TOP_M = 0.6  # an obstacle whose top is under this is shin height
R_HARD_M = 0.25  # closer than this to an obstacle the body does not fit
R_SOFT_M = 0.6  # out to here, the nearer the dearer
SOFT_W = 3.0
UNKNOWN_W = 1.0  # extra cost per metre through columns the map never saw
NARROW_SIDE_M = 0.6  # blocked on both sides within this: a doorway or tight gap
LOOKAHEAD_M = 1.5
SIGHT_M = 0.4  # a shortcut toward the lookahead point keeps this clear where the route does
ARRIVE_M = 0.7
SNAP_M = 1.2  # a goal on a table, or a pose drifted into a wall, moves to free floor this near
REACH_M = 3.0  # map obstacles the priors look at
HORIZON_S = 2.5
BODY_R_M = 0.2
CONTACT_S = 1.0  # a predicted contact sooner than this raises its sector
MAX_POS_ERR_M = 0.6
CORRIDOR_M = 0.4  # the walking path: this far either side of the heading line...
PATH_AHEAD_M = 2.0  # ...and this far ahead. Off it, a mapped thing gets a slow cue at most
GHOST_FRAMES = 5  # the camera sees through a mapped cell this many frames running: it is gone

_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _shift(a: np.ndarray, di: int, dj: int, fill) -> np.ndarray:
    """out[i, j] = a[i + di, j + dj], `fill` outside."""
    out = np.full_like(a, fill)
    n, m = a.shape
    out[max(0, -di) : n - max(0, di), max(0, -dj) : m - max(0, dj)] = a[
        max(0, di) : n - max(0, -di), max(0, dj) : m - max(0, -dj)
    ]
    return out


@dataclass
class FloorPlan:
    i0: int  # cell key (x) of row 0
    j0: int  # cell key (z) of column 0
    blocked: np.ndarray  # bool (nx, nz)
    top: np.ndarray  # metres above the floor of each obstacle column's highest surface
    seen: np.ndarray  # bool, the map saw floor or surface here
    clear: np.ndarray = field(init=False)  # metres from the nearest obstacle
    cost: np.ndarray = field(init=False)  # weight per metre, inf where the body does not fit
    narrow: np.ndarray = field(init=False)
    nearest: np.ndarray = field(init=False)  # (2, nx, nz) index of the nearest free cell

    def __post_init__(self) -> None:
        self.clear = ndimage.distance_transform_edt(~self.blocked) * CELL - CELL / 2
        soft = np.clip((R_SOFT_M - self.clear) / (R_SOFT_M - R_HARD_M), 0, 1) ** 2
        self.cost = 1 + SOFT_W * soft + UNKNOWN_W * ~self.seen
        self.cost[self.clear < R_HARD_M] = np.inf
        free = np.isfinite(self.cost)
        k = int(round(NARROW_SIDE_M / CELL))
        self.narrow = np.zeros_like(free)
        for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            a = np.zeros_like(free)
            b = np.zeros_like(free)
            for s in range(1, k + 1):
                a |= _shift(self.blocked, di * s, dj * s, False)
                b |= _shift(self.blocked, -di * s, -dj * s, False)
            self.narrow |= a & b
        self.narrow &= free
        _, self.nearest = ndimage.distance_transform_edt(~free, return_indices=True)
        self.xs = (np.arange(self.blocked.shape[0]) + self.i0 + 0.5) * CELL
        self.zs = (np.arange(self.blocked.shape[1]) + self.j0 + 0.5) * CELL

    def ij(self, x: float, z: float) -> tuple[int, int]:
        return int(math.floor(x / CELL)) - self.i0, int(math.floor(z / CELL)) - self.j0

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.blocked.shape[0] and 0 <= j < self.blocked.shape[1]

    def free(self, i: int, j: int) -> bool:
        return self.inside(i, j) and math.isfinite(self.cost[i, j])

    def snap(self, i: int, j: int) -> tuple[int, int] | None:
        """The nearest free cell within SNAP_M, or None."""
        if not self.inside(i, j):
            return None
        a, b = int(self.nearest[0, i, j]), int(self.nearest[1, i, j])
        if not self.free(a, b) or math.hypot(a - i, b - j) * CELL > SNAP_M:
            return None
        return a, b


def floor_plan(occ: np.ndarray, margin: int = 10, floor_reach: int = 3) -> FloorPlan:
    """From the home map's occupied cell keys (n, 3: x, y, z)."""
    keys = np.asarray(occ, np.int64)
    i0, j0 = int(keys[:, 0].min()) - margin, int(keys[:, 2].min()) - margin
    nx = int(keys[:, 0].max()) - i0 + margin + 1
    nz = int(keys[:, 2].max()) - j0 + margin + 1
    ci, cj, cy = keys[:, 0] - i0, keys[:, 2] - j0, keys[:, 1]
    low = np.full((nx, nz), np.inf)
    np.minimum.at(low, (ci, cj), cy)
    seen = np.isfinite(low)
    # Local floor: median of the neighbours' lowest surfaces, so a table top whose own column
    # shows no floor is not taken for the floor.
    lowf = np.where(seen, low, np.nan)
    r = floor_reach
    pad = np.pad(lowf, r, constant_values=np.nan)
    stack = [pad[a : a + nx, b : b + nz] for a in range(2 * r + 1) for b in range(2 * r + 1)]
    with warnings.catch_warnings():  # columns with no seen neighbour; filled just below
        warnings.simplefilter("ignore", RuntimeWarning)
        floor = np.nanmedian(stack, axis=0)
    floor = np.where(np.isfinite(floor), floor, np.nanpercentile(lowf, 20))
    h = (cy - floor[ci, cj]) * CELL
    ob = (h >= MIN_H_M) & (h <= MAX_H_M)
    count = np.zeros((nx, nz), np.int32)
    np.add.at(count, (ci[ob], cj[ob]), 1)
    top = np.zeros((nx, nz))
    np.maximum.at(top, (ci[ob], cj[ob]), h[ob] + CELL)
    blocked = count > 0
    neigh = sum(_shift(blocked, di, dj, False).astype(int) for di, dj in _MOVES)
    blocked &= (count > 1) | (neigh > 0)  # a lone one-cell speck is noise
    top[~blocked] = 0
    seen = ndimage.binary_dilation(seen, iterations=2)
    return FloorPlan(i0, j0, blocked, top, seen)


def plan_from_rects(
    rects: list[tuple[float, float, float, float, float]],
    x0: float,
    z0: float,
    x1: float,
    z1: float,
) -> FloorPlan:
    """A floor plan drawn from boxes (x0, z0, x1, z1, top height), for the simulator and tests."""
    i0, j0 = int(math.floor(x0 / CELL)), int(math.floor(z0 / CELL))
    nx, nz = int(math.ceil(x1 / CELL)) - i0, int(math.ceil(z1 / CELL)) - j0
    xs = (np.arange(nx) + i0 + 0.5) * CELL
    zs = (np.arange(nz) + j0 + 0.5) * CELL
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    blocked = np.zeros((nx, nz), bool)
    top = np.zeros((nx, nz))
    for ax, az, bx, bz, t in rects:
        # every cell the box touches, so a 10 cm wall is never thinner than a cell
        m = np.abs(gx - np.clip(gx, ax, bx)) < CELL / 2 - 1e-6
        m &= np.abs(gz - np.clip(gz, az, bz)) < CELL / 2 - 1e-6
        blocked |= m
        top = np.where(m, np.maximum(top, t), top)
    return FloorPlan(i0, j0, blocked, top, np.ones_like(blocked))


# ------------------------------------------------------------------ routes
@dataclass
class Guidance:
    bearing_deg: float  # to the next waypoint, positive to the right
    waypoint_m: float
    route_m: float  # along the route to the goal
    narrow_m: float  # to the next doorway or tight gap on the route, inf if none
    arrived: bool


class Navigator:
    """Route to one goal at a time. `set_goal` costs one Dijkstra over the plan (tens of ms);
    `step` is per frame and costs well under a millisecond."""

    def __init__(self, plan: FloorPlan):
        self.plan = plan
        self.goal: np.ndarray | None = None
        self.label = ""
        self.togo: np.ndarray | None = None  # metres along the route from each cell
        self.next: np.ndarray | None = None  # flat index of each cell's next cell, -1 at the goal

    def set_goal(self, point, label: str = "") -> bool:
        """Route to a 3D map point (a pin). False when no free floor near it is reachable."""
        p = self.plan
        g = p.snap(*p.ij(float(point[0]), float(point[2])))
        if g is None:
            self.goal = None
            return False
        free = np.isfinite(p.cost)
        n = free.size
        flat = np.arange(n).reshape(free.shape)
        rows, cols, w = [], [], []
        for di, dj in _MOVES:
            to_free = _shift(free, di, dj, False)
            ok = free & to_free
            if di and dj:  # no cutting a corner past an obstacle
                ok &= _shift(free, di, 0, False) & _shift(free, 0, dj, False)
            c = _shift(p.cost, di, dj, np.inf)
            rows.append(flat[ok])
            cols.append(_shift(flat, di, dj, -1)[ok])
            w.append(math.hypot(di, dj) * CELL * 0.5 * (p.cost[ok] + c[ok]))
        graph = csr_matrix(
            (np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )
        gi = int(flat[g])
        cost, pred = dijkstra(graph, directed=True, indices=gi, return_predecessors=True)
        # The reverse graph is the same graph (every edge has its mirror), so the predecessor
        # toward the goal from each cell is its next step along the route.
        nxt = pred.copy()
        nxt[gi] = -1
        togo = np.full(n, np.inf)
        togo[gi] = 0.0
        order = np.argsort(cost)
        order = order[np.isfinite(cost[order])]
        r = free.shape[1]
        for c in order[1:]:
            m = nxt[c]
            togo[c] = togo[m] + CELL * math.hypot(c // r - m // r, c % r - m % r)
        self.goal = np.asarray(point, float)
        self.label = label
        self.togo, self.next = togo.reshape(free.shape), nxt
        return bool(np.isfinite(self.togo).sum() > 1)

    def clear_goal(self) -> None:
        self.goal = self.togo = self.next = None

    def _sight(self, a: tuple[int, int], b: tuple[int, int], need_m: float) -> bool:
        """A straight walk from a to b keeps at least need_m from every obstacle."""
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
        for k in range(1, n + 1):
            i = round(a[0] + (b[0] - a[0]) * k / n)
            j = round(a[1] + (b[1] - a[1]) * k / n)
            if not self.plan.free(i, j) or self.plan.clear[i, j] < need_m:
                return False
        return True

    def step(self, pos, yaw: float) -> Guidance | None:
        """Direction to walk now, from the head's map position and yaw (heading.Heading
        conventions: forward = (sin yaw, 0, cos yaw)). None when no route is known from here."""
        if self.goal is None or self.togo is None or self.next is None:
            return None
        p = self.plan
        x, z = float(pos[0]), float(pos[2])
        straight = math.hypot(self.goal[0] - x, self.goal[2] - z)
        c = p.ij(x, z)
        if not p.free(*c) or not math.isfinite(self.togo[c]):
            c = p.snap(*c)
            if c is None or not math.isfinite(self.togo[c]):
                return None
        route = float(self.togo[c])
        arrived = straight < ARRIVE_M or route < 0.3
        r = p.blocked.shape[1]
        chain, k, walked, narrow = [c], c[0] * r + c[1], 0.0, math.inf
        while self.next[k] >= 0 and walked < LOOKAHEAD_M + 2.0:
            m = int(self.next[k])
            walked += CELL * math.hypot(k // r - m // r, k % r - m % r)
            k = m
            cell = (k // r, k % r)
            if math.isinf(narrow) and p.narrow[cell]:
                narrow = walked
            if walked <= LOOKAHEAD_M:
                chain.append(cell)
        # The farthest point along the route still in a straight line. The shortcut must keep the
        # clearance the route itself keeps (up to SIGHT_M): a line that only just fits the body
        # cuts the inside of every corner, and a walker who drifts a little off it touches.
        need = np.minimum.accumulate([min(SIGHT_M, float(p.clear[cell])) for cell in chain])
        target = chain[-1]
        for n in range(len(chain) - 1, -1, -1):
            if self._sight(c, chain[n], float(need[n]) - CELL / 2):
                target = chain[n]
                break
        tx, tz = p.xs[target[0]], p.zs[target[1]]
        if target == c:
            tx, tz = self.goal[0], self.goal[2]
        bearing = _bearing(tx - x, tz - z, yaw)
        return Guidance(bearing, math.hypot(tx - x, tz - z), route, narrow, arrived)


def _bearing(dx: float, dz: float, yaw: float) -> float:
    """Degrees, positive to the right, of a map offset seen from a head at `yaw`."""
    f = (math.sin(yaw), math.cos(yaw))
    rgt = (-math.cos(yaw), math.sin(yaw))
    return math.degrees(math.atan2(dx * rgt[0] + dz * rgt[1], dx * f[0] + dz * f[1]))


# ------------------------------------------------------------------ priors
CAMERA_IDX = [7, 0, 1]  # front-left, front, front-right in the belt's 8 sectors


@dataclass
class Priors:
    hidden: list[float]  # 8: nearest mapped obstacle the camera cannot see now, metres
    hidden_low: list[bool]  # 8: that obstacle is shin height
    known: list[bool | None]  # 8: live reading explained by the map / new / nothing to judge
    contact_s: float  # predicted time until the body meets a mapped obstacle
    contact_m: float
    contact_bearing: float  # degrees from the current heading to what it meets
    active: bool  # the pose was fresh enough to use the map at all
    ms: float = 0.0
    # 8: nearest hidden obstacle inside the walking path (CORRIDOR_M by PATH_AHEAD_M)
    path_hidden: list[float] = field(default_factory=lambda: [math.inf] * 8)
    # 8: the live reading in that camera sector lies on the walking path
    on_path: list[bool] = field(default_factory=lambda: [True] * 8)
    ghosts: int = 0  # mapped cells cleared this frame because the camera saw through them


class Ghosts:
    """Mapped cells the camera keeps seeing through: the thing has moved or never was. After
    GHOST_FRAMES frames running, a cell is stale and the priors stop cueing it. Kept by world
    cell key, so it outlives floor plan rebuilds; a cell the camera sees again where the map has
    it is cleared. `stale` is what the map should forget. `seen` holds the cells the camera saw
    where the map has them this walk (and their neighbours); with `need_seen`, only those get a
    strong cue once they drop out of view, since a ghost under the frame is never seen first."""

    def __init__(self, need_seen: bool = False) -> None:
        self.count: dict[int, int] = {}
        self.stale: set[int] = set()
        self.seen: set[int] = set()
        self.need_seen = need_seen

    @staticmethod
    def keys(plan: FloorPlan, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
        return (ii.astype(np.int64) + plan.i0) * 1_000_003 + (jj.astype(np.int64) + plan.j0)

    def mask(self, keys: np.ndarray) -> np.ndarray:
        if not self.stale:
            return np.zeros(len(keys), bool)
        return np.isin(keys, np.fromiter(self.stale, np.int64, len(self.stale)))

    def seen_mask(self, keys: np.ndarray) -> np.ndarray:
        if not self.need_seen:
            return np.ones(len(keys), bool)
        if not self.seen:
            return np.zeros(len(keys), bool)
        return np.isin(keys, np.fromiter(self.seen, np.int64, len(self.seen)))

    def update(self, through: np.ndarray, confirmed: np.ndarray, matched: np.ndarray) -> int:
        """`through`: keys the camera saw past this frame; `confirmed`: keys it saw where mapped;
        `matched`: front faces the reading fits closely, what `seen` is built from. Returns how
        many turned stale."""
        new = 0
        for k in confirmed.tolist():
            self.count.pop(k, None)
            self.stale.discard(k)
        for k in matched.tolist():
            self.seen.update(k + di * 1_000_003 + dj for di in (-1, 0, 1) for dj in (-1, 0, 1))
        for k in through.tolist():
            c = self.count.get(k, 0) + 1
            self.count[k] = c
            if c >= GHOST_FRAMES and k not in self.stale:
                self.stale.add(k)
                new += 1
        return new


def priors(
    plan: FloorPlan,
    pos,
    yaw: float,
    camera: list[float],
    hfov_deg: float,
    speed_mps: float = 0.0,
    yaw_rate: float = 0.0,
    pos_err_m: float = 0.0,
    eye_h_m: float = 1.6,
    below_deg: float = 35.0,
    ghosts: Ghosts | None = None,
) -> Priors:
    """`camera`: the safety loop's front-left, front, front-right distances this frame (inf where
    nothing). `below_deg`: how far below level the bottom edge of the frame looks. `yaw_rate`
    in Heading units (radians a second, positive turning left). `ghosts`, when given, is
    updated from what the camera sees through, and its stale cells are left out."""
    t0 = time.perf_counter()
    inf = math.inf
    out = Priors([inf] * 8, [False] * 8, [None] * 8, inf, inf, 0.0, False)
    if pos_err_m > MAX_POS_ERR_M:
        out.ms = 1000 * (time.perf_counter() - t0)
        return out
    out.active = True
    x, z = float(pos[0]), float(pos[2])
    i, j = plan.ij(x, z)
    k = int(REACH_M / CELL) + 1
    a0, a1 = max(0, i - k), max(0, min(plan.blocked.shape[0], i + k + 1))
    b0, b1 = max(0, j - k), max(0, min(plan.blocked.shape[1], j + k + 1))
    win = plan.blocked[a0:a1, b0:b1]
    vis_d, vis_b = np.zeros(0), np.zeros(0)
    # Ghost checks: g_ are the mapped front faces the camera could see through; c_ are all
    # visible cells, stale ones too, so a cell seen again where the map has it comes back.
    g_d, g_b, g_k = np.zeros(0), np.zeros(0), np.zeros(0, np.int64)
    c_d, c_b, c_k = g_d, g_b, g_k
    stale_key = np.zeros(0, np.int64)
    if win.any():
        ii, jj = np.nonzero(win)
        kk = Ghosts.keys(plan, ii + a0, jj + b0) if ghosts is not None else None
        gone = ghosts.mask(kk) if ghosts is not None else np.zeros(len(ii), bool)
        sure_cell = ghosts.seen_mask(kk) if ghosts is not None else np.ones(len(ii), bool)
        if kk is not None:
            stale_key = kk[gone]
        dx, dz = plan.xs[ii + a0] - x, plan.zs[jj + b0] - z
        f, rgt = (math.sin(yaw), math.cos(yaw)), (-math.cos(yaw), math.sin(yaw))
        ahead, side = dx * f[0] + dz * f[1], dx * rgt[0] + dz * rgt[1]
        d = np.maximum(np.hypot(dx, dz) - CELL / 2, 0.05)
        bear = np.degrees(np.arctan2(side, ahead))
        top = plan.top[ii + a0, jj + b0]
        near = d < REACH_M
        sec = (((bear + 22.5) % 360) // 45).astype(int)
        in_fov = np.abs(bear) < hfov_deg / 2 - 2
        under = np.arctan2(eye_h_m - top, d) > math.radians(below_deg)  # below the frame
        visible = in_fov & ~under & near
        hidden = near & ~visible & ~gone
        path = (ahead > 0) & (ahead < PATH_AHEAD_M) & (np.abs(side) < CORRIDOR_M + pos_err_m / 2)
        for s in range(8):
            m = hidden & (sec == s)
            if m.any():
                q = int(np.argmin(np.where(m, d, inf)))
                out.hidden[s] = float(d[q])
                out.hidden_low[s] = bool(top[q] < LOW_TOP_M)
                if (m & path & sure_cell).any():
                    out.path_hidden[s] = float(d[m & path & sure_cell].min())
        vis_d, vis_b = d[visible & ~gone], bear[visible & ~gone]
        if kk is not None:
            # Only the front face of what the map has, per 2 degree bearing, and only clearly in
            # view (off the frame's edges and bottom): a box whose near edge is under the frame
            # says nothing about its far side, and cells behind a front face are not in sight.
            cand = near & ~gone & (np.abs(bear) < hfov_deg / 2 - 5)
            bins = np.floor(bear / 2).astype(int)
            front = np.zeros(len(d), bool)
            if cand.any():
                idx = np.flatnonzero(cand)
                order = idx[np.lexsort((d[idx], bins[idx]))]
                first = np.ones(len(order), bool)
                first[1:] = bins[order][1:] != bins[order][:-1]
                front[order[first]] = True
            low = np.arctan2(eye_h_m - top, d) > math.radians(below_deg - 5)
            sure = front & ~low
            seen = visible & (np.abs(bear) < hfov_deg / 2 - 5)
            g_d, g_b, g_k = d[sure], bear[sure], kk[sure]
            c_d, c_b, c_k = d[seen], bear[seen], kk[seen]
    through, seen_there, matched = [], [], []
    for n, s in enumerate(CAMERA_IDX):  # (b) does the map explain what the camera reads?
        live = camera[n]
        if not math.isfinite(live) or live >= REACH_M:
            if len(g_b):
                lo = -hfov_deg / 2 + n * hfov_deg / 3
                m = (g_b >= lo + 1) & (g_b < lo + hfov_deg / 3 - 1) & (g_d < REACH_M - 0.5)
                through.append(g_k[m])
            out.on_path[s] = False
            continue
        lo = -hfov_deg / 2 + n * hfov_deg / 3  # that third of the view, not the 45 degree sector
        m = (vis_b >= lo) & (vis_b < lo + hfov_deg / 3)
        if len(g_b):
            # The third's reading is its nearest strip, so a mapped cell nearer than it, by more
            # than the depth error, was seen through. Inf (nothing in range) sees through all.
            gm = (g_b >= lo + 1) & (g_b < lo + hfov_deg / 3 - 1)
            through.append(g_k[gm & (g_d < live - (0.3 + 0.15 * g_d + pos_err_m))])
            matched.append(g_k[gm & (np.abs(g_d - live) < 0.1 + 0.05 * g_d + pos_err_m)])
        if len(c_b):
            cm = (c_b >= lo) & (c_b < lo + hfov_deg / 3)
            seen_there.append(c_k[cm & (np.abs(c_d - live) < 0.3 + 0.15 * c_d + pos_err_m)])
        b = math.radians((n - 1) * hfov_deg / 3)  # that third's centre, from the heading
        out.on_path[s] = abs(live * math.sin(b)) < CORRIDOR_M + BODY_R_M and live < PATH_AHEAD_M
        if not m.any():  # nothing mapped there: new, if the map saw that floor at all
            b = yaw - math.radians((n - 1) * hfov_deg / 3)  # in the map's frame
            li, lj = plan.ij(x + live * math.sin(b), z + live * math.cos(b))
            if plan.inside(li, lj) and plan.seen[li, lj]:
                out.known[s] = False
            continue
        out.known[s] = live >= float(vis_d[m].min()) - (0.3 + 0.15 * live + pos_err_m)
    if ghosts is not None:
        cat = np.concatenate
        out.ghosts = ghosts.update(
            cat(through) if through else np.zeros(0, np.int64),
            cat(seen_there) if seen_there else np.zeros(0, np.int64),
            cat(matched) if matched else np.zeros(0, np.int64),
        )
    if speed_mps >= 0.2:
        psi, px, pz = yaw, x, z
        for step in range(1, int(HORIZON_S / 0.1) + 1):
            psi += yaw_rate * 0.1
            px += speed_mps * 0.1 * math.sin(psi)
            pz += speed_mps * 0.1 * math.cos(psi)
            ci, cj = plan.ij(px, pz)
            if plan.inside(ci, cj) and plan.clear[ci, cj] < BODY_R_M - pos_err_m / 2:
                # what the body meets: the obstacle cell nearest the contact point
                lo_i, lo_j = max(0, ci - 4), max(0, cj - 4)
                ob = np.argwhere(plan.blocked[lo_i : ci + 5, lo_j : cj + 5]) + [lo_i, lo_j]
                if ghosts is not None and len(ob):
                    ok = Ghosts.keys(plan, ob[:, 0], ob[:, 1])
                    ob = ob[~np.isin(ok, stale_key) & ghosts.seen_mask(ok)]
                    if not len(ob):
                        continue  # only a ghost, or nothing seen this walk: walk on through it
                if len(ob):
                    a, b = ob[np.argmin(np.hypot(ob[:, 0] - ci, ob[:, 1] - cj))]
                    px, pz = plan.xs[a], plan.zs[b]
                out.contact_s = step * 0.1
                out.contact_m = speed_mps * out.contact_s
                out.contact_bearing = _bearing(px - x, pz - z, yaw)
                break
    out.ms = 1000 * (time.perf_counter() - t0)
    return out


def _stronger(cue: tuple[int, int], d: float) -> tuple[int, int]:
    c = cue_for_distance(d)
    mode, inten = MODES[c.mode], int(round(255 * c.intensity))
    if mode == MODES["off"]:
        return cue
    if cue[0] in (MODES["off"], MODES["unknown"]) or mode > cue[0]:
        return mode, max(inten, cue[1] if cue[0] == mode else 0)
    return cue


def apply(
    cmd: list[tuple[int, int]],
    pri: Priors,
    sectors: list[float],
    off_path: str = "slow",
    raise_new: bool = False,
    hidden: bool = True,
) -> dict:
    """Change the belt command in place with the map's priors. `sectors` are the eight distances
    the command was planned from. Priors only add or raise cues on the camera sectors; the one
    downgrade is a map-only fast cue beside or behind you to slow. Strong cues from the map are
    for the walking path (CORRIDOR_M by PATH_AHEAD_M, or a predicted contact); a hidden thing off
    it gets a slow cue at most (`off_path="slow"`) or none (`"none"`). A new reading on the path is
    only raised with `raise_new`: in the simulator that raise made the walker swerve into things
    (collisions 0.53 to 0.67 per walk over seeds 3000-3059). `hidden=False` drops rule (a) and
    leaves the predicted contact as the only map cue ahead: with `Ghosts(need_seen=True)` that
    is the simulator's best trade (docs/navigation.md, "Path gating"). Returns what changed."""
    did = {"hidden": 0, "calm": 0, "new": 0, "contact": 0}
    if not pri.active:
        return did
    slow, fast, solid = MODES["slow"], MODES["fast"], MODES["solid"]
    hit = round(pri.contact_bearing / 45) % 8 if math.isfinite(pri.contact_s) else -1
    for s in range(8):
        # (a) what the map has that the camera cannot see, in front only (low things under the
        # frame's bottom edge) and only nearer than 1 m. Beside and behind, memory's own map
        # sectors already cue. On his recorded walk the floor plan's side cues more than doubled
        # the onsets (about one in ten was then seen by the camera), and its 1 to 2 m slow pulse
        # flickered at the 2 m edge with every wobble of the pose: 9 short buzzes a minute.
        if hidden and s in CAMERA_IDX and pri.hidden[s] < SLOW_M:
            if pri.path_hidden[s] < SLOW_M:
                new = _stronger(cmd[s], pri.path_hidden[s])
            elif off_path == "slow" and cmd[s][0] in (MODES["off"], MODES["unknown"]):
                new = (slow, 150)
            else:
                new = cmd[s]
            if new != cmd[s]:
                cmd[s] = new
                did["hidden"] += 1
        k = pri.known[s]
        on_course = s == hit and pri.contact_s < 2.0
        if s not in CAMERA_IDX and not on_course and cmd[s][0] == fast:
            # (b) beside or behind, where the cue came from the map: calm to a slow pulse. Live
            # depth (the camera sectors) is never weakened, and nothing is calmed below slow.
            cmd[s] = (slow, min(cmd[s][1], 150))
            did["calm"] += 1
        elif (
            raise_new
            and k is False
            and pri.on_path[s]
            and sectors[s] < OFF_M
            and cmd[s][0]
            in (
                slow,
                fast,
            )
        ):
            cmd[s] = (cmd[s][0] + 1, max(cmd[s][1], 204))  # (b) new: one step stronger
            did["new"] += 1
    if hit >= 0 and pri.contact_s < CONTACT_S:  # (c) the turn or heading runs into something mapped
        want = solid if pri.contact_s < 0.8 else fast
        if cmd[hit][0] in (MODES["off"], MODES["unknown"], slow) or cmd[hit][0] < want:
            cmd[hit] = (want, max(cmd[hit][1], 204))
            did["contact"] += 1
    return did


__all__ = [
    "FloorPlan",
    "floor_plan",
    "plan_from_rects",
    "Navigator",
    "Guidance",
    "Priors",
    "priors",
    "apply",
    "FAST_M",
]
