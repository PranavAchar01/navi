"""What the glasses saw a moment ago and can no longer see: field-of-view memory for hazards.

The glasses camera is narrow (about 45 by 72 degrees, portrait) and its pitch follows the head.
A desk edge seen while looking down drops under the frame when the wearer looks up at the
monitor, though it still sits at hip height a step ahead; a chair seen while facing it leaves
the side of the view on a turn. Cues built from the current frame alone forget both.

`HazardMemory` keeps every obstacle cell the camera has seen (5 cm floor cells, with the height
of what stands there) in a level frame turned by the head's yaw and moved by its position, so
a cell keeps its place in the room after it leaves the view. Heights come from the depth and the
head pitch: the floor's own pitch while it is in view (`level_floor`), the accelerometer with
the offset learned from the floor (`PitchFuse`) while it is not. While a cell is in view the
live frame decides: seen again, it is refreshed; seen through (the depth there lies beyond it),
it is dropped; hidden behind something nearer, it is kept. Out of view, it is kept for `keep_s`
and is what `update` reports, per belt sector, in the head's own frame. docs/fov-catalogue.md
has the failure modes and the numbers.

Frames: level head frame (right, forward, up from the floor) in metres. World frame: x, z on the
floor with the heading convention of `heading.Heading` (forward = (sin yaw, cos yaw), yaw grows
turning left).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# The glasses' forward axis in their motion frame. The DAT docs give no axes; gravity reads about
# -9.4 on x standing level, and the image agrees that the camera looks along +z (scripts/
# fov_walk_events.py: the picture moves with the pitch this gives, frames at 1 degree show the
# wall level ahead, at 73 degrees the wearer's lap).
FWD = np.array([0.0, 0.0, 1.0])

CELL_M = 0.05
DROP_CELL_M = 0.1  # floor-edge cells are coarser: an edge's range wobbles a row's worth
SECTORS = 8
CORRIDOR_HALF_M = 0.45  # the path a walking body sweeps, as safety.BODY_HALF_M
PATH_AHEAD_M = 2.0
LOW_SLOPE = 0.02  # metres of extra height a low cell needs per metre of range
MIN_SPAN = 0.8  # metres of floor a pitch fit needs (L16)


def pitch_from_gravity(acc) -> float:
    """Degrees the camera looks below level, from an accelerometer reading (gravity included)."""
    a = np.asarray(acc, float)
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        return 0.0
    return math.degrees(math.asin(max(-1.0, min(1.0, -float(a @ FWD) / n))))


def floor_pitch(normal) -> float:
    """Degrees below level the camera looks, from the floor's unit normal in camera axes
    (OpenCV: x right, y down, z forward; the normal points up), as `safety.analyse` fits it."""
    return -math.degrees(math.asin(max(-1.0, min(1.0, float(normal[2])))))


class PitchFuse:
    """Head pitch from the accelerometer, its offset learned from the floor while it is in view.

    The accelerometer has no idea where the camera points inside the frame (mounting, fit on the
    nose); the floor plane does, but only while the floor is in the frame. Each frame with a floor
    moves the offset `alpha` of the way to agree; without one, the accelerometer carries on.

    A floor that disagrees by more than `gate_deg` is not taken: after a head lift the floor fit
    can keep the last plane on the strength of a few points (fovsim desk_lookup, 26 degrees off),
    and a box entering the bottom of the view tilts the fit 6 to 12 degrees for a second (fovsim
    low_box). The gate starts wide (`first_gate_deg`) until `settle` floors agree, then narrows.
    If the floors keep disagreeing the same way for `reseed` frames, the offset itself was wrong
    (the glasses moved on the nose) and it is taken afresh. With `snap`, a frame whose floor is
    taken returns the floor's own pitch: steadier than the accelerometer (which shakes about 2
    degrees frame to frame on tonight's walk), and a 2 degree error lifts the floor 7 cm at 2 m."""

    def __init__(
        self,
        alpha: float = 0.1,
        gate_deg: float = 4.0,
        first_gate_deg: float = 10.0,
        settle: int = 5,
        reseed: int = 20,
        snap: bool = True,
    ):
        self.alpha = alpha
        self.snap = snap
        self.gate_deg = gate_deg
        self.first_gate_deg = first_gate_deg
        self.settle = settle
        self.reseed = reseed
        self.offset: float | None = None
        self.taken = 0
        self._out: list[float] = []

    def update(self, imu_pitch: float | None, normal=None) -> float | None:
        fl = floor_pitch(normal) if normal is not None else None
        if imu_pitch is None:
            return fl
        if fl is not None:
            d = fl - imu_pitch
            gate = self.first_gate_deg if self.taken < self.settle else self.gate_deg
            if self.offset is None or abs(d - self.offset) < gate:
                old = d if self.offset is None else self.offset
                self.offset = old + self.alpha * (d - old)
                self.taken += 1
                self._out.clear()
                if self.snap:
                    return fl
            else:
                self._out = [*self._out[1 - self.reseed :], d]  # the last `reseed` refusals
                if len(self._out) >= self.reseed and np.ptp(self._out) < self.gate_deg:
                    self.offset = float(np.median(self._out))
                    self.taken = 1
                    self._out.clear()
        return imu_pitch + (self.offset or 0.0)


def level_points(depth: np.ndarray, hfov_deg: float, pitch_deg: float, eye_h: float):
    """Each pixel of a z-depth map (metres) as (right, forward, height above the floor), for a
    camera `eye_h` above the floor looking `pitch_deg` below level with no roll."""
    h, w = depth.shape
    f = 0.5 * w / math.tan(math.radians(hfov_deg) / 2)
    u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    xc, yc, zc = u * depth / f, v * depth / f, depth
    p = math.radians(pitch_deg)
    fwd = math.cos(p) * zc - math.sin(p) * yc
    up = -math.sin(p) * zc - math.cos(p) * yc
    return xc, fwd, eye_h + up


def level_floor(
    depth: np.ndarray,
    hfov_deg: float,
    pitch_deg: float,
    eye_h: float,
    iters: int = 3,
    min_span: float | None = None,
) -> tuple[float, float] | None:
    """The floor's pitch and the camera's height over it, refined from a pitch guess.

    Pixels that sit within a band of the floor under the guess are fitted with a no-roll plane
    (sin p * z + cos p * y = H in camera coordinates, y down), the band narrows, and the fit runs
    again. A box or a desk edge at the bottom of the view stays out of the fit because it stands
    above the band, which safety's own fit (a plane through the lowest points) does not promise:
    in fovsim low_box it read 21 to 27 degrees for a second with the head at 15. Returns None when
    fewer than 40 pixels look like floor, or the fit wanders more than 12 degrees off the guess.
    With `min_span`, the floor pixels in the last band must run at least that far (5th to 95th
    percentile of forward distance): a strip of floor under a wall leaves the pitch free to tilt
    into the wall (fovsim shelf_head read 13 to 30 degrees with the head at 5 to 18)."""
    h, w = depth.shape
    f = 0.5 * w / math.tan(math.radians(hfov_deg) / 2)
    v = np.arange(h)[:, None] - h / 2 + 0.5
    yc = np.broadcast_to(v * depth / f, depth.shape)
    ok = np.isfinite(depth) & (depth > 0.05)
    p, H = pitch_deg, eye_h
    for tol in (0.25, 0.12, 0.06)[:iters]:
        _, fwd, hgt = level_points(depth, hfov_deg, p, H)
        m = ok & (np.abs(hgt) < tol) & (fwd < 4.0)
        if m.sum() < 40:
            return None
        A = np.stack([depth[m], yc[m]], 1)
        (a, b), *_ = np.linalg.lstsq(A, np.ones(len(A)), rcond=None)
        p = math.degrees(math.atan2(a, b))
        H = 1.0 / math.hypot(a, b)
    if min_span is not None and np.subtract(*np.percentile(fwd[m], [95, 5])) < min_span:
        return None
    if abs(p - pitch_deg) > 12.0:
        return None
    return p, H


@dataclass
class FovCues:
    hidden: list[float]  # 8 sectors clockwise from ahead: nearest remembered cell out of view
    hidden_h: list[float]  # its height above the floor, metres (nan where none)
    path_m: float  # nearest remembered out-of-view cell inside the walking path
    path_h: float
    cells: int  # cells held
    cleared: int  # cells dropped this frame because the camera saw through them
    in_view: int = 0
    near: list[float] = field(default_factory=list)  # 8: nearest held cell, in view or not


def band_name(h: float, eye_h: float) -> str:
    """Where on the body a hazard at height `h` meets the wearer."""
    if not math.isfinite(h):
        return ""
    if h < 0.35:
        return "low"
    if h < 1.1:
        return "hip"
    return "head" if h > eye_h - 0.4 else "chest"


class HazardMemory:
    """Remembered obstacle cells. `update` once per frame with the scaled depth.

    `keep_s`: how long a cell lives out of view. `min_hits`: frames a cell must be seen in before
    it is reported (single-frame depth noise must not linger). `band`: heights that are in the
    way, metres above the floor (low clutter to just over the head). `reach_m`: cells further
    than this from the wearer are forgotten. `gate`: None reports every held cell; "hip_pitch"
    only those under 1.1 m whose head pitch changed more than `gate_deg` since they were seen;
    "moved" those the head's pitch, yaw (same `gate_deg`) or a 0.3 m step since explains."""

    def __init__(
        self,
        hfov_deg: float,
        eye_h: float,
        keep_s: float = 6.0,
        min_hits: int = 2,
        band: tuple[float, float] | None = None,
        reach_m: float = 3.0,
        max_range_m: float = 2.5,
        stride: int = 2,
        low_slope: float = LOW_SLOPE,
        gate: str | None = None,
        gate_deg: float = 15.0,
    ):
        self.hfov = hfov_deg
        self.low_slope = low_slope
        self.gate = gate
        self.gate_deg = gate_deg
        self.eye_h = eye_h
        self.keep_s = keep_s
        self.min_hits = min_hits
        self.band = band or (0.08, eye_h + 0.15)
        self.reach_m = reach_m
        self.max_range_m = max_range_m
        self.stride = stride
        self.clear()

    def clear(self) -> None:
        self.key = np.zeros(0, np.int64)
        self.pts = np.zeros((0, 3))  # world x, z, height
        self.seen_t = np.zeros(0)
        self.hits = np.zeros(0, np.int32)
        self.seen_pose = np.zeros((0, 4))  # head pitch (deg), yaw (rad), x, z when last seen

    @staticmethod
    def _keys(x, z, hgt) -> np.ndarray:
        i = np.floor(x / CELL_M).astype(np.int64) + (1 << 20)
        j = np.floor(z / CELL_M).astype(np.int64) + (1 << 20)
        k = np.clip(np.floor(hgt / 0.1).astype(np.int64), 0, 31)
        return (i << 26) | (j << 5) | k

    def update(
        self,
        t: float,
        depth: np.ndarray,
        pitch_deg: float,
        yaw: float,
        pos=(0.0, 0.0),
        mask: np.ndarray | None = None,
    ) -> FovCues:
        """`depth`: z-depth in metres (0 or non-finite where unknown). `pos`: the head's floor
        position (x, z) in the same frame as `yaw`. `mask`, when given, limits new cells to
        pixels the safety geometry calls obstacles (same shape as `depth`)."""
        s = self.stride
        d = np.asarray(depth, np.float32)[::s, ::s]
        hh, ww = d.shape
        ok = np.isfinite(d) & (d > 0.05)
        r, f, hgt = level_points(np.where(ok, d, 0), self.hfov, pitch_deg, self.eye_h)
        fw = (math.sin(yaw), math.cos(yaw))
        rt = (-math.cos(yaw), math.sin(yaw))
        px, pz = float(pos[0]), float(pos[1])
        pose = np.array([pitch_deg, yaw, px, pz])

        # 1. What the frame says about the cells already held.
        cleared = 0
        in_view = np.zeros(len(self.key), bool)
        if len(self.key):
            dx, dz = self.pts[:, 0] - px, self.pts[:, 1] - pz
            cr = dx * rt[0] + dz * rt[1]
            cf = dx * fw[0] + dz * fw[1]
            cu = self.pts[:, 2] - self.eye_h
            p = math.radians(pitch_deg)
            zc = math.cos(p) * cf - math.sin(p) * cu
            yc = -math.sin(p) * cf - math.cos(p) * cu
            foc = 0.5 * ww / math.tan(math.radians(self.hfov) / 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                uu = np.floor(foc * cr / zc + ww / 2).astype(np.int64)
                vv = np.floor(foc * yc / zc + hh / 2).astype(np.int64)
            in_view = (zc > 0.1) & (uu >= 1) & (uu < ww - 1) & (vv >= 1) & (vv < hh - 1)
            idx = np.flatnonzero(in_view)
            live = d[vv[idx], uu[idx]]
            known = ok[vv[idx], uu[idx]]
            tol = 0.12 + 0.08 * zc[idx]
            through = known & (live > zc[idx] + tol)
            back = known & (np.abs(live - zc[idx]) <= tol)
            self.seen_t[idx[back]] = t
            self.seen_pose[idx[back]] = pose
            drop = np.zeros(len(self.key), bool)
            drop[idx[through]] = True
            cleared = int(drop.sum())
            old = (t - self.seen_t > self.keep_s) | (np.hypot(dx, dz) > self.reach_m)
            keep = ~(drop | old)
            self.key, self.pts = self.key[keep], self.pts[keep]
            self.seen_t, self.hits = self.seen_t[keep], self.hits[keep]
            self.seen_pose = self.seen_pose[keep]
            in_view = in_view[keep]

        # 2. New cells from this frame.
        lo, hi = self.band
        rng = np.hypot(r, f)
        # The floor's height error grows with range (depth scale noise along a slanted ray, a
        # degree of pitch is 3.5 cm at 2 m), so low cells need more height further out.
        new = ok & (hgt > lo + self.low_slope * rng) & (hgt < hi) & (rng < self.max_range_m)
        if mask is not None:
            new &= np.asarray(mask, bool)[::s, ::s]
        if new.any():
            nr, nf, nh = r[new], f[new], hgt[new]
            wx = px + nr * rt[0] + nf * fw[0]
            wz = pz + nr * rt[1] + nf * fw[1]
            k = self._keys(wx, wz, nh)
            k, first = np.unique(k, return_index=True)
            wp = np.column_stack([wx[first], wz[first], nh[first]])
            pos_in = np.searchsorted(self.key, k)
            pos_in = np.clip(pos_in, 0, max(len(self.key) - 1, 0))
            have = (self.key[pos_in] == k) if len(self.key) else np.zeros(len(k), bool)
            self.seen_t[pos_in[have]] = t
            self.seen_pose[pos_in[have]] = pose
            self.hits[pos_in[have]] += 1
            add = ~have
            self.key = np.concatenate([self.key, k[add]])
            self.pts = np.vstack([self.pts, wp[add]])
            self.seen_t = np.concatenate([self.seen_t, np.full(add.sum(), t)])
            self.hits = np.concatenate([self.hits, np.ones(add.sum(), np.int32)])
            self.seen_pose = np.vstack([self.seen_pose, np.tile(pose, (int(add.sum()), 1))])
            in_view = np.concatenate([in_view, np.ones(add.sum(), bool)])
            order = np.argsort(self.key)
            self.key, self.pts = self.key[order], self.pts[order]
            self.seen_t, self.hits, in_view = self.seen_t[order], self.hits[order], in_view[order]
            self.seen_pose = self.seen_pose[order]

        # 3. Report what is held and out of view, in the head's frame.
        out = FovCues(
            [math.inf] * SECTORS, [math.nan] * SECTORS, math.inf, math.nan, len(self.key), cleared
        )
        out.in_view = int(in_view.sum())
        out.near = [math.inf] * SECTORS
        sure = self.hits >= self.min_hits
        if self.gate is not None and len(self.key):
            # Report a held cell only when the head's own motion explains why it left the view.
            sp = self.seen_pose
            dp = np.abs(pitch_deg - sp[:, 0]) > self.gate_deg
            if self.gate == "hip_pitch":
                sure &= dp & (self.pts[:, 2] < 1.1)
            else:  # "moved": pitch, yaw or a step
                dy = np.abs((yaw - sp[:, 1] + math.pi) % (2 * math.pi) - math.pi)
                step = np.hypot(px - sp[:, 2], pz - sp[:, 3]) > 0.3
                sure &= dp | (dy > math.radians(self.gate_deg)) | step
        if sure.any():
            dx, dz = self.pts[:, 0] - px, self.pts[:, 1] - pz
            cr = dx * rt[0] + dz * rt[1]
            cf = dx * fw[0] + dz * fw[1]
            dist = np.maximum(np.hypot(cr, cf) - CELL_M / 2, 0.05)
            sec = ((np.degrees(np.arctan2(cr, cf)) + 22.5) % 360 // 45).astype(int)
            for si in range(SECTORS):
                m = sure & (sec == si)
                if m.any():
                    out.near[si] = float(dist[m].min())
                m &= ~in_view
                if m.any():
                    q = int(np.argmin(np.where(m, dist, np.inf)))
                    out.hidden[si] = float(dist[q])
                    out.hidden_h[si] = float(self.pts[q, 2])
            pm = sure & ~in_view & (np.abs(cr) < CORRIDOR_HALF_M) & (cf > 0) & (cf < PATH_AHEAD_M)
            if pm.any():
                q = int(np.argmin(np.where(pm, cf, np.inf)))
                out.path_m = float(max(cf[q] - CELL_M / 2, 0.05))
                out.path_h = float(self.pts[q, 2])
        return out


class DropMemory:
    """Remembered floor edges: where the floor ends in a step down or a stair.

    A drop of 18 cm is below `safety.analyse`'s 25 cm line, and a drop seen from 2 m leaves the
    bottom of the view at about 1.5 m, the same way a low box does. The floor's height cannot
    tell it from far floor (a degree of pitch is 9 cm at 5 m), but the depth can: going up a
    column, floor ranges grow by a steady step per row, and at a floor edge the next row
    already looks at the lower floor, a jump several steps long. A floor pixel with such a jump
    above it is a floor edge. Edges are held as 10 cm cells in the world frame, like
    `HazardMemory`, and a held edge is dropped when the frame sees the floor run smoothly
    through it. `update` reports the nearest held edge inside the walking path, in metres along
    it. Add edges only on frames whose pitch is backed by a floor (`add=False` otherwise)."""

    def __init__(
        self,
        hfov_deg: float,
        eye_h: float,
        keep_s: float = 8.0,
        min_hits: int = 2,
        fall_m: float = 0.14,
        max_range_m: float = 3.5,
        min_px: int = 6,
        below_m: float = 0.0,
        stride: int = 2,
    ):
        self.hfov, self.eye_h = hfov_deg, eye_h
        self.keep_s, self.min_hits = keep_s, min_hits
        self.fall_m = fall_m
        self.min_px = min_px
        self.below_m = below_m
        self.max_range_m, self.stride = max_range_m, stride
        self.clear()

    def clear(self) -> None:
        self.key = np.zeros(0, np.int64)
        self.pts = np.zeros((0, 2))  # world x, z
        self.seen_t = np.zeros(0)
        self.hits = np.zeros(0, np.int32)

    def update(
        self, t: float, depth: np.ndarray, pitch_deg: float, yaw: float, pos=(0.0, 0.0), add=True
    ) -> float:
        s = self.stride
        d = np.asarray(depth, np.float32)[::s, ::s]
        hh, ww = d.shape
        ok = np.isfinite(d) & (d > 0.05)
        r, f, hgt = level_points(np.where(ok, d, 0), self.hfov, pitch_deg, self.eye_h)
        fw = (math.sin(yaw), math.cos(yaw))
        rt = (-math.cos(yaw), math.sin(yaw))
        px, pz = float(pos[0]), float(pos[1])
        rng = np.hypot(r, f)
        # Floor-like: near floor height (loosely, the pitch is a degree or two off) and level
        # with the two rows below it, so the pitch error cancels in what follows.
        tol = 0.05 + 0.01 * rng
        cont = np.ones_like(ok)
        cont[:-2] = (np.abs(hgt[:-2] - hgt[1:-1]) < tol[:-2]) & (
            np.abs(hgt[:-2] - hgt[2:]) < tol[:-2]
        )
        flat = ok & cont & (np.abs(hgt) < 0.15 + 0.03 * rng) & (rng < self.max_range_m)
        # Going one row up the image from a floor pixel: on floor the height stays put (whatever
        # the pitch error), at a floor edge it falls to the lower floor beyond.
        fall = np.zeros_like(hgt)
        fall[1:] = hgt[1:] - hgt[:-1]
        both = np.zeros_like(ok)
        both[1:] = ok[1:] & ok[:-1]
        below = np.zeros_like(ok)  # the row above lies below the floor, not just lower than it
        below[1:] = hgt[:-1] < -(self.below_m + 0.015 * rng[:-1])
        edge = flat & both & below & (fall > self.fall_m + 0.01 * rng)
        edge[:2] = False
        smooth = flat & both & (np.abs(fall) < 0.6 * self.fall_m + 0.01 * rng)

        if len(self.key):
            dx, dz = self.pts[:, 0] - px, self.pts[:, 1] - pz
            cr = dx * rt[0] + dz * rt[1]
            cf = dx * fw[0] + dz * fw[1]
            cu = -self.eye_h
            p = math.radians(pitch_deg)
            zc = math.cos(p) * cf - math.sin(p) * cu
            yc = -math.sin(p) * cf - math.cos(p) * cu
            foc = 0.5 * ww / math.tan(math.radians(self.hfov) / 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                uu = np.floor(foc * cr / zc + ww / 2).astype(np.int64)
                vv = np.floor(foc * yc / zc + hh / 2).astype(np.int64)
            # The cell's pixel is only good to a few rows (scale and pitch error move an edge
            # 0.2 m, two rows): it is gone only when every row around it is plain floor.
            win = 4
            inv = (zc > 0.1) & (uu >= 1) & (uu < ww - 1) & (vv >= win) & (vv < hh - win)
            idx = np.flatnonzero(inv)
            u_, v_ = uu[idx], vv[idx]
            through = np.ones(len(idx), bool)
            near_edge = np.zeros(len(idx), bool)
            for dv in range(-win, win + 1):
                through &= smooth[v_ + dv, u_]
                near_edge |= edge[v_ + dv, u_]
            gone = np.zeros(len(self.key), bool)
            gone[idx[through]] = True
            self.seen_t[idx[near_edge]] = t
            keep = ~(gone | (t - self.seen_t > self.keep_s) | (np.hypot(dx, dz) > 4.0))
            self.key, self.pts = self.key[keep], self.pts[keep]
            self.seen_t, self.hits = self.seen_t[keep], self.hits[keep]

        if add and edge.sum() >= self.min_px:
            # The edge is on the floor: its distance ahead from the ray's length and the eye
            # height alone, so a degree of pitch error does not move it.
            fe = np.sqrt(
                np.maximum(f[edge] ** 2 + (hgt[edge] - self.eye_h) ** 2 - self.eye_h**2, 0)
            )
            wx = px + r[edge] * rt[0] + fe * fw[0]
            wz = pz + r[edge] * rt[1] + fe * fw[1]
            i = np.floor(wx / DROP_CELL_M).astype(np.int64) + (1 << 20)
            j = np.floor(wz / DROP_CELL_M).astype(np.int64) + (1 << 20)
            k, first = np.unique((i << 26) | j, return_index=True)
            wp = np.column_stack([wx[first], wz[first]])
            at = np.clip(np.searchsorted(self.key, k), 0, max(len(self.key) - 1, 0))
            have = (self.key[at] == k) if len(self.key) else np.zeros(len(k), bool)
            self.seen_t[at[have]] = t
            self.hits[at[have]] += 1
            new = ~have
            self.key = np.concatenate([self.key, k[new]])
            self.pts = np.vstack([self.pts, wp[new]])
            self.seen_t = np.concatenate([self.seen_t, np.full(int(new.sum()), t)])
            self.hits = np.concatenate([self.hits, np.ones(int(new.sum()), np.int32)])
            order = np.argsort(self.key)
            self.key, self.pts = self.key[order], self.pts[order]
            self.seen_t, self.hits = self.seen_t[order], self.hits[order]

        sure = self.hits >= self.min_hits
        if not sure.any():
            return math.inf
        dx, dz = self.pts[sure, 0] - px, self.pts[sure, 1] - pz
        cr = dx * rt[0] + dz * rt[1]
        cf = dx * fw[0] + dz * fw[1]
        pm = (np.abs(cr) < CORRIDOR_HALF_M) & (cf > 0) & (cf < PATH_AHEAD_M)
        return float(cf[pm].min()) if pm.any() else math.inf


class FovLoop:
    """The server's side of the memory, one call per frame (shadow mode first, see
    docs/fov-catalogue.md). Pitch from the accelerometer and the floor, yaw and position from
    `heading.Heading`, new cells only where the safety grid saw an obstacle and outside person
    boxes (a passer-by must not leave a trail)."""

    def __init__(self, hfov_deg: float, eye_h: float, **memory):
        self.hfov, self.eye_h = hfov_deg, eye_h
        self.fuse = PitchFuse()
        self.mem = HazardMemory(hfov_deg, eye_h, **memory)
        self.drops = DropMemory(hfov_deg, eye_h)

    def step(self, now: float, depth_m, grid, floor: bool, obstacles, accel, heading) -> dict:
        """`depth_m`: scaled depth in metres. `grid`: `Scene.grid` (255 = not an obstacle).
        `obstacles`: detections with `.label` and a normalised `.box`. `accel`: recent
        accelerometer rows (glasses axes). `heading`: the server's `Heading`."""
        import cv2

        z = np.nan_to_num(np.asarray(depth_m, np.float32), nan=0.0, posinf=0.0)
        h, w = z.shape
        imu = pitch_from_gravity(np.mean(accel, axis=0)) if len(accel) else None
        guess = imu + (self.fuse.offset or 0.0) if imu is not None else 30.0
        lf = level_floor(z, self.hfov, guess, self.eye_h, min_span=MIN_SPAN) if z.any() else None
        normal = None
        if lf is not None:
            p = math.radians(lf[0])
            normal = np.array([0.0, -math.cos(p), -math.sin(p)])
            z = z * float(np.clip(self.eye_h / lf[1], 0.8, 1.25))
        pitch = self.fuse.update(imu, normal)
        if pitch is None:
            return {"pitch": None}
        mask = None
        if floor:
            g = cv2.resize(np.asarray(grid, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = g < 255
        for o in obstacles:
            if o.label == "person":
                x0, y0, x1, y1 = o.box
                if mask is None:
                    mask = np.ones((h, w), bool)
                mask[int(y0 * h) : int(math.ceil(y1 * h)), int(x0 * w) : int(math.ceil(x1 * w))] = (
                    False
                )
        if lf is None:  # no floor to back the pitch: clear what the frame sees through, add nothing
            mask = np.zeros((h, w), bool)
        yaw = heading.yaw if heading.yaw is not None else heading.turn
        pos = (0.0, 0.0) if heading.pos is None else (float(heading.pos[0]), float(heading.pos[2]))
        fc = self.mem.update(now, z, pitch, yaw, pos, mask)
        drop = self.drops.update(now, z, pitch, yaw, pos, add=lf is not None)

        def fin(v):
            return round(v, 2) if math.isfinite(v) else None

        return {
            "pitch": round(pitch, 1),
            "floor": lf is not None,
            "hidden": [fin(v) for v in fc.hidden],
            "band": [
                band_name(hh, self.eye_h) if math.isfinite(hh) else None for hh in fc.hidden_h
            ],
            "path": fin(fc.path_m),
            "cells": fc.cells,
            "drop": fin(drop),
            "posed": heading.pos is not None,
        }
