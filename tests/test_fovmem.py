import math

import numpy as np

from metasight.fovmem import (
    HazardMemory,
    PitchFuse,
    band_name,
    floor_pitch,
    level_floor,
    level_points,
    pitch_from_gravity,
)

HFOV = 44.8
EYE = 1.6
W, H = 108, 192  # portrait, as the glasses stream


def render(pitch, wall=None, box=None, eye=EYE):
    """Z-depth of a floor seen `pitch` degrees down, plus a wall `wall` m ahead and a box
    (distance, half width, top height) standing on the floor straight ahead."""
    f = 0.5 * W / math.tan(math.radians(HFOV) / 2)
    u, v = np.meshgrid(np.arange(W) - W / 2 + 0.5, np.arange(H) - H / 2 + 0.5)
    p = math.radians(pitch)
    down = math.sin(p) + math.cos(p) * v / f  # per unit z
    ahead = math.cos(p) - math.sin(p) * v / f
    z = np.where(down > 1e-6, eye / np.maximum(down, 1e-6), np.inf)
    if wall is not None:
        z = np.minimum(z, np.where(ahead > 1e-6, wall / np.maximum(ahead, 1e-6), np.inf))
    if box is not None:
        dist, half, top = box
        zb = np.where(ahead > 1e-6, dist / np.maximum(ahead, 1e-6), np.inf)
        hb = eye - zb * down
        right = zb * u / f
        hit = (hb >= 0) & (hb <= top) & (np.abs(right) <= half)
        z = np.where(hit & (zb < z), zb, z)
    return np.where(np.isfinite(z) & (z < 15), z, 0.0).astype(np.float32)


def test_pitch_from_gravity_level_and_down():
    assert abs(pitch_from_gravity([-9.8, 0, 0])) < 1e-6
    g = [-9.8 * math.cos(math.radians(30)), 0, -9.8 * math.sin(math.radians(30))]
    assert abs(pitch_from_gravity(g) - 30) < 1e-6


def test_floor_pitch_matches_level_points():
    p = math.radians(25)
    assert abs(floor_pitch(np.array([0, -math.cos(p), -math.sin(p)])) - 25) < 1e-6
    _, fwd, hgt = level_points(render(25), HFOV, 25, EYE)
    floor = render(25) > 0
    assert np.abs(hgt[floor]).max() < 1e-3
    assert fwd[floor].min() > 0


def test_level_floor_ignores_a_box_and_recovers_from_a_bad_guess():
    z = render(15, box=(1.4, 0.15, 0.12))
    for guess in (9, 15, 22):
        p, h = level_floor(z, HFOV, guess, EYE)
        assert abs(p - 15) < 0.2 and abs(h - EYE) < 0.02
    # Scaled depth: the camera height comes back scaled, which is the depth scale's ruler.
    assert abs(level_floor(z * 1.1, HFOV, 15, EYE)[1] - 1.1 * EYE) < 0.03


def test_level_floor_none_without_floor():
    assert level_floor(render(-20, wall=1.0), HFOV, -20, EYE) is None


def test_pitch_fuse_learns_offset_and_gates_outliers():
    fuse = PitchFuse(snap=False)
    n = lambda deg: np.array([0, -math.cos(math.radians(deg)), -math.sin(math.radians(deg))])  # noqa: E731
    for _ in range(10):
        fuse.update(12.0, n(15.0))  # the accelerometer reads 3 degrees low
    assert abs(fuse.offset - 3.0) < 1e-6
    fuse.update(12.0, n(24.0))  # a tilted fit (box in view): refused
    assert abs(fuse.offset - 3.0) < 1e-6
    assert abs(fuse.update(40.0) - 43.0) < 1e-6  # floor out of view: accelerometer plus offset
    for _ in range(25):  # glasses moved on the nose: a steady new offset is taken afresh
        fuse.update(12.0, n(20.0))
    assert abs(fuse.offset - 8.0) < 0.5


def test_pitch_fuse_snap_returns_the_floor():
    fuse = PitchFuse()
    p = math.radians(15)
    assert abs(fuse.update(13.0, np.array([0, -math.cos(p), -math.sin(p)])) - 15) < 1e-6


def test_band_names():
    assert band_name(0.1, EYE) == "low"
    assert band_name(0.8, EYE) == "hip"
    assert band_name(1.5, EYE) == "head"


def test_memory_holds_a_box_after_it_drops_under_the_view():
    mem = HazardMemory(HFOV, EYE, low_slope=0.0)
    box_at = 1.7
    out = None
    for i, x in enumerate(np.arange(0.0, 1.3, 0.1)):
        out = mem.update(i * 0.1, render(15, box=(box_at - x, 0.15, 0.3)), 15.0, 0.0, (0.0, x))
    # At 0.5 m the box is under the bottom edge of the view; memory still has it ahead.
    assert out.hidden[0] < 0.6 and abs(out.hidden[0] - 0.5) < 0.12
    assert out.path_m < 0.6
    assert band_name(out.hidden_h[0], EYE) == "low"


def test_memory_drops_a_cell_seen_through():
    mem = HazardMemory(HFOV, EYE, low_slope=0.0)
    for i in range(3):
        mem.update(i * 0.1, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0)
    held = mem.update(0.3, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0).cells
    assert held > 0
    out = mem.update(0.4, render(15), 15.0, 0.0)  # the box was carried away: floor behind it
    assert out.cleared > 0 and out.cells < held


def test_memory_forgets_after_keep_s_and_needs_min_hits():
    mem = HazardMemory(HFOV, EYE, keep_s=1.0, min_hits=2, low_slope=0.0)
    out = mem.update(0.0, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0)
    assert math.isinf(out.path_m)  # one sighting is not enough
    mem.update(0.1, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0)
    # Turn 90 degrees to the right: the box is held out of view on the left.
    out = mem.update(0.2, render(15), 15.0, -math.pi / 2)
    assert out.hidden[6] < 1.6
    out = mem.update(1.5, render(15), 15.0, -math.pi / 2)
    assert out.cells == 0


def test_memory_keys_dedupe():
    mem = HazardMemory(HFOV, EYE, low_slope=0.0)
    z = render(15, box=(1.5, 0.15, 0.3))
    mem.update(0.0, z, 15.0, 0.0)
    n = len(mem.key)
    mem.update(0.1, z, 15.0, 0.0)
    assert len(mem.key) == n and np.all(np.diff(mem.key) > 0) and mem.hits.max() == 2


def test_memory_gate_moved_needs_head_motion():
    mem = HazardMemory(HFOV, EYE, low_slope=0.0, gate="moved")
    for i in range(2):
        mem.update(i * 0.1, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0)
    # Same pose, frame lost (blank depth): nothing out of view the head's motion explains.
    out = mem.update(0.2, np.zeros((H, W), np.float32), 15.0, 0.0)
    assert math.isinf(out.path_m)
    out = mem.update(0.3, render(15), 15.0, -math.pi / 2)  # a 90 degree turn explains it
    assert out.hidden[6] < 1.6
