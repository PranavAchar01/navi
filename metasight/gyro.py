"""Head rotation from the glasses' gyroscope, for placing map windows through fast turns.

The map measures rotation from pictures, and a fast head turn leaves too little overlap to
measure it; the gyro measures it directly. The gyro reads in its own axes and on its own clock,
and neither is documented for the glasses (the DAT docs give units only), so `calibrate`
recovers the clock offset, the camera-from-gyro rotation and the gyro bias by matching the gyro's
turns against the turns the map measures between keyframes. Rotations here are camera-to-world
in OpenCV camera axes (x right, y down, z forward), like the map's poses.
"""

import math

import cv2
import numpy as np

MAX_GAP_S = 0.25  # a hole in the samples longer than this leaves the turn across it unknown
CAL_MIN_PAIRS = 24  # keyframe pairs needed before a calibration is trusted
CAL_MIN_TURN_DEG = 4.0  # pairs that barely turn say nothing about the axes
CAL_OK_DEG = 4.0  # median disagreement between gyro and map turns for a usable calibration
CAL_GAIN = 0.35  # ... and at most this share of the map's median turn between the same frames
CAL_PEAKS = 5  # clock offsets tried in full, from the best turn-speed matches
CAL_PEAK_SEP_S = 0.5  # apart by at least this


def _exp(v: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))
    return rot


def _log(rot: np.ndarray) -> np.ndarray:
    v, _ = cv2.Rodrigues(np.asarray(rot, float))
    return v.ravel()


def angle_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1) / 2))))


def chordal_mean(rots: list[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.sum(rots, axis=0))
    if np.linalg.det(u @ vt) < 0:
        return u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return u @ vt


class GyroTrack:
    """Gyro samples (t in seconds on the gyro clock, rad/s in gyro axes), integrated into an
    orientation over time. Grows as samples arrive; `rel` answers in the map's clock and axes."""

    def __init__(self) -> None:
        self.t = np.zeros(0)
        self.w = np.zeros((0, 3))
        self._rot = np.zeros((0, 3, 3))
        self.cam_from_gyro = np.eye(3)
        self.bias = np.zeros(3)
        self.offset = 0.0  # gyro clock = map clock + offset
        self.clock_known = False  # the sender already moved samples onto the map clock, roughly
        self.calibrated = False
        self.cal_err_deg: float | None = None

    def add(self, t: np.ndarray, w: np.ndarray) -> None:
        t, w = np.asarray(t, float), np.asarray(w, float).reshape(-1, 3)
        if self.t.size:
            keep = t > self.t[-1]
            t, w = t[keep], w[keep]
        if not t.size:
            return
        self.t = np.concatenate([self.t, t])
        self.w = np.concatenate([self.w, w])
        self._integrate(len(self._rot))

    def _integrate(self, start: int) -> None:
        """Orientation at every sample from `start` on, holding each rate until the next."""
        rots = list(self._rot[:start]) or [np.eye(3)]
        for k in range(max(1, start), len(self.t)):
            dt = min(self.t[k] - self.t[k - 1], MAX_GAP_S)
            rots.append(rots[-1] @ _exp((self.w[k - 1] - self.bias) * dt))
        self._rot = np.array(rots[: len(self.t)])

    def set_bias(self, bias: np.ndarray) -> None:
        self.bias = np.asarray(bias, float)
        self._rot = np.zeros((0, 3, 3))
        self._integrate(0)

    def covers(self, ts: float, te: float) -> bool:
        a, b = ts + self.offset, te + self.offset
        if not self.t.size or a < self.t[0] or b > self.t[-1]:
            return False
        i, j = np.searchsorted(self.t, [a, b])
        seg = self.t[max(0, i - 1) : j + 1]
        return seg.size < 2 or float(np.diff(seg).max()) <= MAX_GAP_S

    def _orient(self, t: float) -> np.ndarray:
        tg = t + self.offset
        k = int(np.clip(np.searchsorted(self.t, tg) - 1, 0, len(self.t) - 1))
        return self._rot[k] @ _exp((self.w[k] - self.bias) * (tg - self.t[k]))

    def rel(self, ta: float, tb: float) -> np.ndarray:
        """The camera's rotation from time tb into the camera at time ta (cam_a <- cam_b)."""
        g = self._orient(ta).T @ self._orient(tb)
        return self.cam_from_gyro @ g @ self.cam_from_gyro.T

    def speed(self, t: np.ndarray) -> np.ndarray:
        """Turn rate (rad/s) at map-clock times, from the raw samples: needs no axes."""
        return np.interp(np.asarray(t) + self.offset, self.t, np.linalg.norm(self.w, axis=1))


def calibrate(track: GyroTrack, pairs: list[tuple[float, float, np.ndarray]]) -> bool:
    """Fit clock offset, camera-from-gyro rotation and bias from keyframe pairs (ta, tb, the
    map's cam_a <- cam_b rotation). Sets `track.calibrated` when the gyro and the map agree."""
    pairs = [p for p in pairs if p[1] > p[0]]
    if len(pairs) < CAL_MIN_PAIRS or not track.t.size:
        return False
    ta = np.array([p[0] for p in pairs])
    tb = np.array([p[1] for p in pairs])
    vis = np.array([_log(p[2]) for p in pairs])
    # 1. Clock offset from turn speed alone, which does not depend on the unknown axes. Only
    # short pairs: over a long one the head turns back and forth and the net turn says little.
    dt = tb - ta
    short = dt <= 1.5 * dt.min() + 0.2
    sa, sb = ta[short], tb[short]
    vis_rate = np.linalg.norm(vis[short], axis=1) / dt[short]

    def speed_corr(off: float) -> float:
        track.offset = off
        mid = np.linspace(0, 1, 7)
        g = np.mean([track.speed(sa + f * (sb - sa)) for f in mid], axis=0)
        if g.std() < 1e-6:
            return -1.0
        return float(np.corrcoef(g, vis_rate)[0, 1])

    if track.calibrated:  # a refit only follows a small drift of the clocks
        base, span = track.offset, 1.0
    elif track.clock_known:  # only network delay left between the clocks
        base, span = track.offset, 2.0
    else:  # the clocks may share nothing, so start from the two starts
        base, span = track.t[0] - ta.min(), max(30.0, float(tb.max() - ta.min()))
    grid = base + np.arange(-span, span, 0.1)
    score = np.array([speed_corr(o) for o in grid])
    # Steady turning (circling a table) correlates almost as well at a wrong offset as at the
    # right one, so the best few peaks each get the full fit and the rotations decide.
    peaks = []
    for k in np.argsort(-score):
        if all(abs(grid[k] - q) >= CAL_PEAK_SEP_S for q in peaks):
            peaks.append(grid[k])
        if len(peaks) == CAL_PEAKS:
            break
    turn = np.linalg.norm(vis, axis=1) > math.radians(CAL_MIN_TURN_DEG)
    if turn.sum() < CAL_MIN_PAIRS // 2:
        return False
    bias0 = track.bias.copy()

    def err(off: float) -> float:
        track.offset = off
        return float(
            np.median(
                [angle_deg(track.rel(a, b).T @ p[2]) for a, b, p in zip(ta, tb, pairs, strict=True)]
            )
        )

    def fit_at(peak: float) -> tuple[float, float, np.ndarray, np.ndarray]:
        fine = peak + np.arange(-0.12, 0.12, 0.01)
        track.offset = float(max(fine, key=speed_corr))
        track.set_bias(bias0)
        # 2. Axes, then bias, a few rounds: rotation vectors map across as v_cam = R v_gyro.
        for _ in range(3):
            track.cam_from_gyro = np.eye(3)
            g = np.array([_log(track.rel(a, b)) for a, b in zip(ta, tb, strict=True)])
            u, _, vt = np.linalg.svd(vis[turn].T @ g[turn])
            d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
            track.cam_from_gyro = u @ d @ vt
            # What is left after the axes is bias: e = log(gyro^-1 map) = -R b dt for small
            # turns. Longer pairs pin bias best (map noise is per pair, bias grows with time).
            res = np.array(
                [_log(track.rel(a, b).T @ p[2]) for a, b, p in zip(ta, tb, pairs, strict=True)]
            )
            db = -(res @ track.cam_from_gyro).sum(0) / (tb - ta).sum()
            track.set_bias(track.bias + db)
        # 3. Now the axes are known, the offset can be tuned on the full rotation.
        off = float(min(track.offset + np.arange(-0.1, 0.1, 0.005), key=err))
        return err(off), off, track.cam_from_gyro.copy(), track.bias.copy()

    best = min((fit_at(q) for q in peaks), key=lambda f: f[0])
    track.cal_err_deg, track.offset, track.cam_from_gyro = best[0], best[1], best[2]
    track.set_bias(best[3])
    # Small turns leave small errors even at a wrong offset: the gyro has to explain most of
    # the turning, not just stay under an absolute bar.
    still = float(np.median(np.degrees(np.linalg.norm(vis, axis=1))))
    track.calibrated = track.cal_err_deg < min(CAL_OK_DEG, CAL_GAIN * still)
    return track.calibrated


def synth(
    times: np.ndarray,
    rots: np.ndarray,
    rate_hz: float = 30.0,
    bias_dps: float = 0.5,
    noise: float = 0.01,
    cam_from_gyro: np.ndarray | None = None,
    offset: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gyro samples a real sensor would give for a camera that turned through `rots` (camera-to-
    world at `times`): rates in gyro axes, a constant bias, white noise, clock shifted by offset.
    For testing the map against walks whose true rotation is known."""
    rng = np.random.default_rng(seed)
    rc = np.eye(3) if cam_from_gyro is None else cam_from_gyro
    ts = np.arange(times[0], times[-1], 1.0 / rate_hz)
    k = np.clip(np.searchsorted(times, ts) - 1, 0, len(times) - 2)
    f = (ts - times[k]) / (times[k + 1] - times[k])
    ori = [rots[i] @ _exp(f_ * _log(rots[i].T @ rots[i + 1])) for i, f_ in zip(k, f, strict=True)]
    w = np.array([_log(a.T @ b) for a, b in zip(ori[:-1], ori[1:], strict=True)]) * rate_hz
    w = w @ rc  # camera axes into gyro axes: v_gyro = R^T v_cam
    bias = rng.normal(0, math.radians(bias_dps), 3)
    w = w + bias + rng.normal(0, noise, w.shape)
    return ts[:-1] + offset, w
