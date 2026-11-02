"""Body motion for the rendered walker, planned from an exported walk (scripts/export_sim_scene.py).

The sim gives a path (x, y, heading per video frame). This turns it into what a body does on
that path, so the render in blender_sim_room.py has no foot sliding and no pops:

- the root stays on the path; the body yaw follows the heading at a capped turn rate;
- footsteps are planted in world space, one every STEP_M of travel (short, blindfolded steps)
  or of turning on the spot, and a planted foot never moves until it lifts;
- a step that is in the air when the walker stops is finished, so it never freezes mid-swing;
- at a seat the walker turns its back to the chair while stepping back onto the spot where
  the sit-down lands the pelvis on the seat, and the swivel chair turns to meet it;
- a contact is a stumble (a short jolt away from what was hit, plus a lean the renderer adds);
- clip weights (walk, idle, sit) are smoothed so the renderer can blend without pops.

Pure Python (no numpy, no bpy) so it runs inside Blender and under pytest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

FPS = 30
STEP_M = 0.45  # blindfolded: a short step
TURN_STEP_M = 0.22  # gait travel per radian turned on the spot (a step about every 115 degrees)
STANCE_HALF_M = 0.1  # foot centre to body centre, sideways
SWING = (0.0, 0.42)  # left foot in the air for this part of the gait cycle; right is +0.5
LIFT_M = 0.06  # shuffle height
YAW_RATE = math.radians(150)  # cap on how fast the body turns
YAW_TAU_S = 0.12  # body yaw lag behind the heading
POS_HALF = 2  # path smoothing half-width in frames (the sim steps at 50 Hz)
BLEND_TAU_S = 0.1
MOVING_MPS = 0.06
SETTLE_S = 0.25  # time to finish a step left in the air at a stop
SIT_TURN_S = 1.0  # turn round and step back to the seat
SIT_DOWN_S = 0.8  # lower onto it
HIPS_BEHIND_M = 0.28  # the sit clip moves the pelvis this far back from the feet
STUMBLE_S = 0.55
STUMBLE_M = 0.06
FOOT_Z = 0.02  # ankle height of a planted foot


@dataclass
class Pose:
    x: float
    y: float
    yaw: float  # body yaw, counter-clockwise from +x
    feet: tuple  # ((x, y, z, yaw), (x, y, z, yaw)), left then right
    phase: float  # gait cycle 0..1 (left foot lifts at 0, lands at SWING[1])
    w: dict  # clip weights: walk, idle, sit (sum 1)
    sit_t: float  # 0..1 through the sit-down
    stumble: float  # 0..1 envelope of a stumble
    stumble_dir: float  # world angle the body is pushed toward
    head_yaw: float  # head turn, radians, positive = left (counter-clockwise)
    chair_turn: float  # swivel of the chair, radians, from its built yaw
    extra: dict = field(default_factory=dict)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _smooth_path(frames):
    n = len(frames)
    out = []
    for i in range(n):
        lo, hi = max(0, i - POS_HALF), min(n - 1, i + POS_HALF)
        k = hi - lo + 1
        out.append(
            (
                sum(frames[j]["x"] for j in range(lo, hi + 1)) / k,
                sum(frames[j]["y"] for j in range(lo, hi + 1)) / k,
            )
        )
    return out


def _body_yaw(frames):
    """Heading unwrapped, then followed with a lag and a rate cap: no snaps, no spins."""
    dt = 1.0 / FPS
    raw = [frames[0]["heading"]]
    for f in frames[1:]:
        raw.append(raw[-1] + wrap(f["heading"] - raw[-1]))
    # the sim heading dithers frame to frame while it steers; average it first
    avg = []
    for i in range(len(raw)):
        lo, hi = max(0, i - 3), min(len(raw) - 1, i + 3)
        avg.append(sum(raw[lo : hi + 1]) / (hi - lo + 1))
    yaw = [avg[0]]
    a = 1 - math.exp(-dt / YAW_TAU_S)
    for t in avg[1:]:
        d = (t - yaw[-1]) * a
        d = max(-YAW_RATE * dt, min(YAW_RATE * dt, d))
        yaw.append(yaw[-1] + d)
    return yaw


def _rect_dir(px, py, rects):
    """World angle from the nearest point of the nearest box to (px, py)."""
    best, ang = 1e9, 0.0
    for cx, cy, hx, hy, yaw in rects:
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = c * (px - cx) - s * (py - cy), s * (px - cx) + c * (py - cy)
        qx, qy = max(-hx, min(hx, lx)), max(-hy, min(hy, ly))
        d = math.hypot(lx - qx, ly - qy)
        if d < best:
            c2, s2 = math.cos(yaw), math.sin(yaw)
            wx, wy = cx + c2 * qx - s2 * qy, cy + s2 * qx + c2 * qy
            best, ang = d, math.atan2(py - wy, px - wx)
    return ang


def plan(frames, motor_deg, seat=None, rects=()):
    """Poses for every video frame. `seat` is (cx, cy, face) of the chair to sit on, where
    face is the world angle its seat faces; `rects` are (cx, cy, hx, hy, yaw) obstacle boxes."""
    n = len(frames)
    dt = 1.0 / FPS
    sit0 = next((i for i, f in enumerate(frames) if f["sit"] > 0), None)
    walk_n = n if sit0 is None else sit0
    pos = _smooth_path(frames[:walk_n]) if walk_n else []
    yaw = _body_yaw(frames[:walk_n]) if walk_n else []
    # the sit: from where the walk stopped, turn round and step back to the stand point
    sit_turn_n = int(SIT_TURN_S * FPS)
    sit_down_n = int(SIT_DOWN_S * FPS)
    chair = [0.0] * n
    if sit0 is not None and seat is not None:
        cx, cy, face = seat
        px, py = pos[-1] if pos else (frames[sit0]["x"], frames[sit0]["y"])
        y0 = yaw[-1] if yaw else frames[sit0]["heading"]
        away = math.atan2(py - cy, px - cx)
        tx, ty = cx + math.cos(away) * HIPS_BEHIND_M, cy + math.sin(away) * HIPS_BEHIND_M
        y1 = y0 + wrap(away - y0)
        swivel = wrap(away - face)
        for k in range(sit0, n):
            u = smoothstep((k - sit0 + 1) / sit_turn_n)
            pos.append((px + (tx - px) * u, py + (ty - py) * u))
            yaw.append(y0 + (y1 - y0) * u)
            chair[k] = swivel * smoothstep((k - sit0 + 1) / (sit_turn_n * 0.8))
    elif sit0 is not None:  # no seat known: stand still
        for _ in range(sit0, n):
            pos.append(pos[-1])
            yaw.append(yaw[-1])
    # gait travel: distance walked plus turning on the spot
    inc = [0.0]
    for i in range(1, n):
        ds = math.hypot(pos[i][0] - pos[i - 1][0], pos[i][1] - pos[i - 1][1])
        inc.append(ds + TURN_STEP_M * abs(yaw[i] - yaw[i - 1]))
    cyc = 2 * STEP_M
    # finish a step left in the air: where the body stops mid-swing, keep the feet going
    settle = cyc * (SWING[1] - SWING[0]) / (SETTLE_S * FPS)
    g = [0.0]
    for i in range(1, n):
        gi = g[-1] + inc[i]
        ph = (gi / cyc) % 1.0
        swinging = SWING[0] < ph < SWING[1] or SWING[0] + 0.5 < ph < SWING[1] + 0.5
        if inc[i] < MOVING_MPS * dt and swinging:
            end = SWING[1] if ph < 0.5 else SWING[1] + 0.5
            gi = min(gi + settle, (gi // cyc) * cyc + end * cyc)
        g.append(gi)

    def at_g(target):
        """Body (x, y, yaw) when the gait travel first reaches `target` (the end if never)."""
        lo, hi = 0, n - 1
        if g[-1] < target:
            return pos[-1][0], pos[-1][1], yaw[-1]
        while lo < hi:
            mid = (lo + hi) // 2
            if g[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return pos[lo][0], pos[lo][1], yaw[lo]

    def plant(side, g_land):
        # the pelvis passes over the foot at mid-stance
        stance = 1.0 - (SWING[1] - SWING[0])
        bx, by, byaw = at_g(g_land + 0.5 * stance * cyc)
        s = 1 if side == 0 else -1  # left is +90 degrees from the heading
        return (
            bx + math.cos(byaw + s * math.pi / 2) * STANCE_HALF_M,
            by + math.sin(byaw + s * math.pi / 2) * STANCE_HALF_M,
            byaw,
        )

    def foot(side, gi):
        off = 0.0 if side == 0 else 0.5
        u = gi / cyc - off
        k = math.floor(u)
        ph = u - k
        land = (k + off + SWING[1]) * cyc  # this cycle's landing; before g=0 means the start
        if ph >= SWING[1]:  # planted since that landing
            x, y, fy = plant(side, land)
            return (x, y, FOOT_Z, fy)
        start = plant(side, land - cyc)
        if ph < SWING[0]:
            return (*start[:2], FOOT_Z, start[2])
        end = plant(side, land)
        v = smoothstep((ph - SWING[0]) / (SWING[1] - SWING[0]))
        dyaw = wrap(end[2] - start[2])
        return (
            start[0] + (end[0] - start[0]) * v,
            start[1] + (end[1] - start[1]) * v,
            FOOT_Z + LIFT_M * math.sin(math.pi * v),
            start[2] + dyaw * v,
        )

    # stumbles at each new contact
    hits = [i for i in range(1, walk_n) if frames[i]["collisions"] > frames[i - 1]["collisions"]]
    st_n = int(STUMBLE_S * FPS)
    # clip weights, smoothed
    a = 1 - math.exp(-dt / BLEND_TAU_S)
    w = {"walk": 0.0, "idle": 1.0, "sit": 0.0}
    w1 = dict(w)  # two passes of the filter: weights ease in and out, no kink at the start
    head = 0.0
    out = []
    for i in range(n):
        moving = (
            i < walk_n
            and (inc[i] > MOVING_MPS * dt or (i + 1 < n and inc[i + 1] > MOVING_MPS * dt))
        ) or (i > 0 and g[i] - g[i - 1] > 1e-4)
        sit_t = 0.0
        if sit0 is not None and seat is not None and i >= sit0 + sit_turn_n:
            sit_t = smoothstep((i - sit0 - sit_turn_n) / sit_down_n)
        tgt = {"walk": 1.0 if moving else 0.0, "idle": 0.0 if moving else 1.0, "sit": 0.0}
        if sit_t > 0:
            tgt = {"walk": 0.0, "idle": 0.0, "sit": 1.0}
        for kk in w:
            w1[kk] += (tgt[kk] - w1[kk]) * a
            w[kk] += (w1[kk] - w[kk]) * a
        tot = sum(w.values())
        wn = {kk: v / tot for kk, v in w.items()}
        # stumble envelope
        e, sdir = 0.0, 0.0
        for h in hits:
            if 0 <= i - h < st_n:
                e = math.sin(math.pi * (i - h) / st_n)
                sdir = _rect_dir(pos[h][0], pos[h][1], rects) if rects else yaw[h] + math.pi
        # head turns toward the strongest felt motor (motor angles clockwise from ahead)
        felt = frames[i]["felt"] if i < walk_n else [0.0] * len(motor_deg)
        k_max = max(range(len(motor_deg)), key=lambda k: felt[k] if k < len(felt) else 0.0)
        want = 0.0
        if k_max < len(felt) and felt[k_max] > 0.2:
            want = -math.radians(max(-40.0, min(40.0, wrap_deg(motor_deg[k_max]))))
        head += (want - head) * (1 - math.exp(-dt / 0.25))
        jx = math.cos(sdir) * STUMBLE_M * e
        jy = math.sin(sdir) * STUMBLE_M * e
        out.append(
            Pose(
                x=pos[i][0] + jx,
                y=pos[i][1] + jy,
                yaw=yaw[i],
                feet=(foot(0, g[i]), foot(1, g[i])),
                phase=(g[i] / cyc) % 1.0,
                w=wn,
                sit_t=sit_t,
                stumble=e,
                stumble_dir=sdir,
                head_yaw=head * (1 - sit_t),
                chair_turn=chair[i],
            )
        )
    return out


def wrap_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0
