import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sim_motion as M  # noqa: E402

EIGHT = [k * 45.0 for k in range(8)]


def _frame(x, y, hd, sit=0.0, coll=0):
    return {
        "x": x,
        "y": y,
        "heading": hd,
        "sit": sit,
        "felt": [0.0] * 8,
        "collisions": coll,
        "walking": sit == 0,
    }


def _walk():
    """0.8 m/s along +x for 3 s, a 90 degree turn on the spot, 1.5 s more along +y, a stop."""
    fr, x, y = [], 0.0, 0.0
    for _ in range(90):
        x += 0.8 / M.FPS
        fr.append(_frame(x, y, 0.0))
    for k in range(30):
        fr.append(_frame(x, y, math.pi / 2 * (k + 1) / 30))
    for _ in range(45):
        y += 0.8 / M.FPS
        fr.append(_frame(x, y, math.pi / 2))
    for _ in range(30):
        fr.append(_frame(x, y, math.pi / 2))
    return fr


def test_planted_feet_do_not_slide():
    poses = M.plan(_walk(), EIGHT)
    for side in (0, 1):
        for a, b in zip(poses, poses[1:], strict=False):
            fa, fb = a.feet[side], b.feet[side]
            if fa[2] == M.FOOT_Z and fb[2] == M.FOOT_Z:
                assert math.hypot(fb[0] - fa[0], fb[1] - fa[1]) < 1e-6


def test_feet_keep_up_with_the_body():
    poses = M.plan(_walk(), EIGHT)
    for p in poses:
        for f in p.feet:
            assert math.hypot(f[0] - p.x, f[1] - p.y) < 0.6


def test_turn_rate_is_capped_and_the_spot_turn_takes_steps():
    poses = M.plan(_walk(), EIGHT)
    rates = [abs(b.yaw - a.yaw) * M.FPS for a, b in zip(poses, poses[1:], strict=False)]
    assert max(rates) <= M.YAW_RATE + 1e-9
    lifts = sum(1 for p in poses[90:125] for f in p.feet if f[2] > M.FOOT_Z + 0.01)
    assert lifts > 0  # it steps round, it does not spin on planted feet


def test_both_feet_down_after_a_stop():
    poses = M.plan(_walk(), EIGHT)
    for f in poses[-1].feet:
        assert f[2] == M.FOOT_Z
    assert poses[-1].w["idle"] > 0.9


def test_blend_weights_have_no_pops():
    poses = M.plan(_walk(), EIGHT)
    for a, b in zip(poses, poses[1:], strict=False):
        for k in ("walk", "idle"):
            assert abs(b.w[k] - a.w[k]) < 0.25


def test_sit_lands_the_pelvis_on_the_seat():
    fr = _walk()[:120]
    cx, cy = fr[-1]["x"] + 0.5, fr[-1]["y"]
    for k in range(48):
        fr.append(_frame(fr[-1]["x"], fr[-1]["y"], fr[-1]["heading"], sit=(k + 1) / 48))
    for _ in range(36):
        fr.append(dict(fr[-1]))
    poses = M.plan(fr, EIGHT, seat=(cx, cy, 0.0))
    end = poses[-1]
    # the sit clip moves the pelvis HIPS_BEHIND_M back, onto the chair centre
    hx = end.x - math.cos(end.yaw) * M.HIPS_BEHIND_M
    hy = end.y - math.sin(end.yaw) * M.HIPS_BEHIND_M
    assert math.hypot(hx - cx, hy - cy) < 0.02
    assert end.sit_t == 1.0 and end.w["sit"] > 0.99
    # the chair swivelled to face the walker
    assert abs(M.wrap(end.chair_turn - math.pi)) < 0.05


def test_contact_is_a_stumble_away_from_the_box():
    fr = _walk()[:60]
    for k in range(30, 60):
        fr[k]["collisions"] = 1
    poses = M.plan(fr, EIGHT, rects=[(fr[30]["x"], fr[30]["y"] + 0.4, 0.2, 0.2, 0.0)])
    assert max(p.stumble for p in poses[30:50]) > 0.9
    assert abs(M.wrap(poses[35].stumble_dir + math.pi / 2)) < 0.3  # pushed to -y
