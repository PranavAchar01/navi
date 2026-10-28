"""Ray-Ban Meta glasses, emulated end to end, talking to the server the way the iOS app does.

What is emulated, and the source for each number:

- Camera stream: the Wearables Device Access Toolkit's `StreamingResolution` presets, portrait
  9:16 (high 720x1280, medium 504x896, low 360x640) at 2, 7, 15, 24 or 30 fps. The app asks for
  medium at 24 fps (ios/CameraAccess/ViewModels/CameraViewModel.swift). Landscape footage is
  cropped to 9:16 at full height, which narrows its field of view; the crop's horizontal field
  of view is printed and must match `hfov_deg` in the server profile (metasight.glasses.toml).
- Bluetooth Classic link: the toolkit docs say a tight link first lowers the resolution one
  step, then the frame rate, never below 15 fps. Bandwidth, latency and jitter here are
  assumptions until measured on real glasses.
- Motion: `MotionSample` (timestampNs, accelerometer m/s^2 with gravity, about +9.81 on the axis
  opposing gravity, gyroscope rad/s) at a `MotionSamplingRate` of 5 to 60 Hz; the app asks for
  30. The head's rotation is measured from the footage itself (bearing vectors of tracked
  corners, aligned frame to frame), so the gyro agrees with what the camera saw. Axes as the
  app sends them: x right, y up, z toward the face.
- Speech: the toolkit's transcription results, partial then final, for scheduled questions.
  Final transcripts that read as questions go to the server as "ask", as in GlassesSensors.swift.
- Phone bridge: ServerLink.swift's wire format. Binary frames are Float64 LE capture time (ms
  since epoch), UInt32 LE seq, then JPEG. Motion is batched every 100 ms. Device and link state
  go out as "device" messages so the viewer can show them.

Usage:
    python -m metasight.glasses_emu VIDEO [--res medium] [--fps 24] [--imu-hz 30]
        [--ask "43:where is my water bottle?"] [--loops 2] [--bandwidth 2.0] [--dip 20:8:0.6]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import struct
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .query import is_question

RESOLUTIONS = {"high": (720, 1280), "medium": (504, 896), "low": (360, 640)}
STEPS = ["high", "medium", "low"]
FRAME_RATES = [2, 7, 15, 24, 30]
MIN_ADAPT_FPS = 15
IMU_RATES = [5, 10, 15, 24, 30, 60]
G = 9.81

# Link assumptions (not in the toolkit docs): HEVC at about 0.12 bits per pixel per frame,
# 90 ms glasses-to-phone base latency, 15 ms jitter.
BITS_PER_PIXEL = 0.12
BASE_LATENCY_S = 0.09
JITTER_S = 0.015
GYRO_NOISE = 0.004  # rad/s, white
GYRO_BIAS = 0.002  # rad/s, fixed per axis
ACCEL_NOISE = 0.06  # m/s^2


# ---------------------------------------------------------------- optics
def crop_hfov(src_w: int, src_h: int, src_hfov_deg: float) -> tuple[slice, float]:
    """Columns of a centred 9:16 crop at full height, and that crop's horizontal field of view.
    A portrait source is used whole."""
    cw = min(src_w, round(src_h * 9 / 16))
    x0 = (src_w - cw) // 2
    t = math.tan(math.radians(src_hfov_deg) / 2) * cw / src_w
    return slice(x0, x0 + cw), math.degrees(2 * math.atan(t))


# ---------------------------------------------------------------- head motion from footage
def _bearings(pts: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    b = np.column_stack([(pts[:, 0] - cx) / f, (pts[:, 1] - cy) / f, np.ones(len(pts))])
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R with b ~ R a (rows are vectors)."""
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def frame_rotation(prev: np.ndarray, cur: np.ndarray, hfov_deg: float) -> np.ndarray | None:
    """Rotation of the camera between two grey frames, as R with (direction in the new camera
    frame) = R @ (direction in the old one), OpenCV axes. Translation is ignored: corners are
    mostly metres away and a head turns far more than it moves between frames."""
    p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
    if p0 is None or len(p0) < 12:
        return None
    p1, ok, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
    ok = ok.ravel() == 1
    if ok.sum() < 12:
        return None
    h, w = prev.shape
    f = 0.5 * w / math.tan(math.radians(hfov_deg) / 2)
    a = _bearings(p0.reshape(-1, 2)[ok], f, w / 2, h / 2)
    b = _bearings(p1.reshape(-1, 2)[ok], f, w / 2, h / 2)
    r = _kabsch(a, b)
    for _ in range(2):  # drop movers and bad tracks, refit
        err = np.linalg.norm(b - a @ r.T, axis=1)
        keep = err <= max(2.5 * np.median(err), 1e-4)
        if keep.sum() < 12:
            break
        r = _kabsch(a[keep], b[keep])
    return r


def _rotvec(r: np.ndarray) -> np.ndarray:
    v, _ = cv2.Rodrigues(r)
    return v.ravel()


LEVEL_RATE = 0.03  # share of the tilt error removed per frame (about a 1 s time constant)


def _relevel(q: np.ndarray, up_world: np.ndarray) -> np.ndarray:
    """Pull the camera's tilt part of the way back to level. Frame-to-frame rotations add up
    small pitch and roll errors that would otherwise tip gravity over within seconds; a worn
    head stays near level when walking, and turning (yaw) is left alone."""
    u = q.T @ up_world
    v = np.array([0.0, -1.0, 0.0])
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return q
    angle = math.atan2(s, float(u @ v)) * LEVEL_RATE
    rc, _ = cv2.Rodrigues(axis / s * angle)
    return q @ rc.T


CV_TO_DEVICE = np.diag([1.0, -1.0, -1.0])  # OpenCV (x right, y down, z ahead) -> glasses axes


@dataclass
class HeadTrack:
    t: np.ndarray  # frame times, s
    omega: np.ndarray  # (N, 3) angular velocity over [t_i, t_i+1), rad/s, glasses axes
    up: np.ndarray  # (N, 3) unit up vector at t_i, glasses axes


def head_track(frames: list[np.ndarray], fps: float, hfov_deg: float) -> HeadTrack:
    """Angular velocity and up direction per frame, starting level."""
    n = len(frames)
    omega = np.zeros((n, 3))
    up = np.zeros((n, 3))
    q = np.eye(3)  # world <- camera (OpenCV axes); the world's up is -y
    up_world = np.array([0.0, -1.0, 0.0])
    for i in range(n):
        up[i] = CV_TO_DEVICE @ (q.T @ up_world)
        if i + 1 < n:
            r = frame_rotation(frames[i], frames[i + 1], hfov_deg)
            if r is None:
                r = np.eye(3)
            cam = r.T  # the new camera's axes in the old camera's frame
            omega[i] = CV_TO_DEVICE @ _rotvec(cam) * fps
            q = _relevel(q @ cam, up_world)
    return HeadTrack(np.arange(n) / fps, omega, up)


def imu_samples(track: HeadTrack, hz: int, rng: random.Random) -> list[tuple[float, list[float]]]:
    """(t, [ax, ay, az, gx, gy, gz]) at `hz`, held from the frame interval each sample falls in,
    with noise and a fixed gyro bias."""
    bias = [rng.uniform(-GYRO_BIAS, GYRO_BIAS) for _ in range(3)]
    out = []
    dur = track.t[-1]
    k = 0
    for j in range(int(dur * hz)):
        t = j / hz
        while k + 1 < len(track.t) and track.t[k + 1] <= t:
            k += 1
        bob = 0.6 * math.sin(2 * math.pi * 1.8 * t)  # walking bounce along up, m/s^2
        a = (G + bob) * track.up[k] + [rng.gauss(0, ACCEL_NOISE) for _ in range(3)]
        g = track.omega[k] + bias + [rng.gauss(0, GYRO_NOISE) for _ in range(3)]
        out.append((t, [*map(float, a), *map(float, g)]))
    return out


# ---------------------------------------------------------------- Bluetooth link
@dataclass
class Link:
    """The toolkit's adaptation: resolution steps down first, then frame rate (not below 15)."""

    want_res: str
    want_fps: int
    bandwidth_bps: float
    dips: list[tuple[float, float, float]] = field(default_factory=list)  # (start, length, x)
    res: str = ""
    fps: int = 0
    calm_since: float = 0.0

    def __post_init__(self) -> None:
        self.res, self.fps = self.want_res, self.want_fps

    def budget(self, t: float) -> float:
        x = 1.0
        for start, length, factor in self.dips:
            if start <= t < start + length:
                x = min(x, factor)
        return self.bandwidth_bps * x

    @staticmethod
    def need(res: str, fps: int) -> float:
        w, h = RESOLUTIONS[res]
        return w * h * BITS_PER_PIXEL * fps

    def update(self, t: float) -> None:
        b = self.budget(t)
        if self.need(self.res, self.fps) > b:
            self.calm_since = t
            i = STEPS.index(self.res)
            if self.res == self.want_res and i + 1 < len(STEPS):  # one resolution step
                self.res = STEPS[i + 1]
            else:
                lower = [f for f in FRAME_RATES if MIN_ADAPT_FPS <= f < self.fps]
                if lower:
                    self.fps = lower[-1]
            return
        if t - self.calm_since < 2.0:
            return
        # Step back up once the higher setting would fit with room to spare.
        if self.fps < self.want_fps:
            up = min(f for f in FRAME_RATES if f > self.fps)
            if self.need(self.res, up) < 0.8 * b:
                self.fps, self.calm_since = up, t
        elif self.res != self.want_res:
            up = STEPS[STEPS.index(self.res) - 1]
            if self.need(up, self.fps) < 0.8 * b:
                self.res, self.calm_since = up, t

    def latency(self, rng: random.Random, t: float) -> float:
        w, h = RESOLUTIONS[self.res]
        airtime = w * h * BITS_PER_PIXEL / self.budget(t)
        return BASE_LATENCY_S + airtime + abs(rng.gauss(0, JITTER_S))


# ---------------------------------------------------------------- speech
def parse_asks(items: list[str]) -> list[tuple[float, str]]:
    out = []
    for it in items:
        t, _, text = it.partition(":")
        out.append((float(t), text.strip()))
    return sorted(out)


def transcript_steps(text: str, start: float) -> list[tuple[float, str, bool]]:
    """Partial transcripts word by word (about 0.3 s a word), then the final one."""
    words = text.split()
    steps = [(start + 0.3 * (i + 1), " ".join(words[: i + 1]), False) for i in range(len(words))]
    steps.append((start + 0.3 * len(words) + 0.4, text, True))
    return steps


# ---------------------------------------------------------------- the run
def load_footage(path: str, src_hfov: float, res: str) -> tuple[list, list, float, float]:
    """Stream-sized BGR frames, small grey frames for head tracking, source fps, stream hfov."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    color, grey = [], []
    cols, hfov = None, src_hfov
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if cols is None:
            cols, hfov = crop_hfov(f.shape[1], f.shape[0], src_hfov)
        color.append(cv2.resize(f[:, cols], RESOLUTIONS[res], interpolation=cv2.INTER_AREA))
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        grey.append(cv2.resize(g, (480, round(480 * f.shape[0] / f.shape[1]))))
    return color, grey, fps, hfov


async def run(args: argparse.Namespace) -> None:
    import websockets

    rng = random.Random(args.seed)
    print(f"loading {args.video} ...")
    color, grey, src_fps, hfov = load_footage(args.video, args.src_hfov, args.res)
    track = head_track(grey, src_fps, args.src_hfov)
    imu = imu_samples(track, args.imu_hz, rng)
    dur = len(color) / src_fps
    print(
        f"{len(color)} frames, {dur:.1f} s; stream field of view {hfov:.1f} deg horizontal "
        f"(server profile hfov_deg must match); head turned "
        f"{math.degrees(np.abs(track.omega[:, 1]).sum() / src_fps):.0f} deg in total"
    )
    link = Link(
        args.res,
        args.fps,
        args.bandwidth * 1e6,
        [
            tuple(map(float, d.split(":")))
            for d in args.dip  # type: ignore[misc]
        ],
    )
    asks = parse_asks(args.ask)
    speech = [s for t, text in asks for s in transcript_steps(text, t)]

    async with websockets.connect(args.url, max_size=None) as ws:

        async def device(state: str, **extra) -> None:
            w, h = RESOLUTIONS[link.res]
            await ws.send(
                json.dumps(
                    {
                        "type": "device",
                        "name": "Ray-Ban Meta (emulated)",
                        "state": state,
                        "res": [w, h],
                        "fps": link.fps,
                        "imu_hz": args.imu_hz,
                        "kbps": round(link.need(link.res, link.fps) / 1000),
                        "budget_kbps": round(link.budget(clock()) / 1000),
                        **extra,
                    }
                )
            )

        shown: dict = {}

        async def reader() -> None:
            async for m in ws:
                if isinstance(m, bytes):
                    continue
                d = json.loads(m)
                if d.get("type") == "answer":
                    print(f"{clock():6.1f} answer: {d['text']}")
                elif d.get("type") == "belt" and d.get("found") and not shown.get("found"):
                    print(f"{clock():6.1f} belt: found it")
                    shown["found"] = True
                elif d.get("type") == "safety" and args.verbose and d.get("guide"):
                    if int(clock()) != shown.get("guide"):
                        shown["guide"] = int(clock())
                        print(f"{clock():6.1f} guide {d['guide']}")

        t0 = time.monotonic()
        epoch0 = time.time()

        def clock() -> float:
            return time.monotonic() - t0

        rt = asyncio.create_task(reader())
        for state in ("powered on", "donned", "session started"):
            await device(state)
        await asyncio.sleep(0.3)
        t0 = time.monotonic()
        epoch0 = time.time()

        total = dur * args.loops
        seq, next_frame, imu_i, sp_i = 0, 0.0, 0, 0
        pending: list[tuple[float, bytes]] = []  # (deliver at, message)
        motion_buf: list[list[float]] = []
        last_flush = last_status = 0.0
        latencies: list[float] = []
        while clock() < total + 0.5:
            now = clock()
            # Camera: capture on the stream's clock, deliver after the link's latency.
            if now >= next_frame and next_frame < total:
                link.update(next_frame)
                ft = next_frame % dur
                i = min(int(ft * src_fps), len(color) - 1)
                img = color[i]
                if link.res != args.res:
                    img = cv2.resize(img, RESOLUTIONS[link.res], interpolation=cv2.INTER_AREA)
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                lat = link.latency(rng, next_frame)
                latencies.append(lat)
                seq += 1
                head = struct.pack("<dI", (epoch0 + next_frame) * 1000, seq)
                pending.append((next_frame + lat, head + jpg.tobytes()))
                next_frame += 1.0 / link.fps
            while pending and pending[0][0] <= now:
                await ws.send(pending.pop(0)[1])
            # Motion: samples as they happen, batched every 100 ms like ServerLink.
            while imu_i < len(imu) * args.loops:
                t_s, row = imu[imu_i % len(imu)]
                t_abs = (imu_i // len(imu)) * dur + t_s
                if t_abs > now:
                    break
                motion_buf.append([round(t_abs, 4), *[round(v, 4) for v in row]])
                imu_i += 1
            if now - last_flush >= 0.1 and motion_buf:
                await ws.send(json.dumps({"type": "motion", "s": motion_buf}))
                motion_buf, last_flush = [], now
            # Speech.
            while sp_i < len(speech) and speech[sp_i][0] <= now:
                _, text, final = speech[sp_i]
                await ws.send(json.dumps({"type": "transcript", "text": text, "final": final}))
                if final:
                    print(f"{now:6.1f} heard: {text}")
                    if is_question(text):
                        await ws.send(json.dumps({"type": "ask", "text": text}))
                sp_i += 1
            if now - last_status >= 1.0:
                recent = latencies[-30:] or [0.0]
                await device("streaming", latency_ms=round(1000 * float(np.mean(recent))))
                last_status = now
            await asyncio.sleep(0.004)
        await device("session ended")
        await asyncio.sleep(1.0)
        rt.cancel()
    print(f"sent {seq} frames, {imu_i} motion samples")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("video")
    p.add_argument("--url", default="ws://127.0.0.1:8445/ws")
    p.add_argument("--res", choices=STEPS, default="medium")
    p.add_argument("--fps", type=int, choices=FRAME_RATES, default=24)
    p.add_argument("--imu-hz", type=int, choices=IMU_RATES, default=30)
    p.add_argument("--src-hfov", type=float, default=105.0, help="source footage, degrees")
    p.add_argument("--bandwidth", type=float, default=2.0, help="link budget, Mbit/s")
    p.add_argument("--dip", action="append", default=[], help="start:length:factor, seconds")
    p.add_argument("--ask", action="append", default=[], help='"seconds:spoken question"')
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-v", "--verbose", action="store_true")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
