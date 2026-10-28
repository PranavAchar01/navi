import math

import numpy as np

from metasight.geometry import (
    SECTORS,
    cue_for_distance,
    gravity_alignment,
    map_sector_distances,
    merge_sectors,
    relative_to,
    sector_distances_from_depth,
    sector_index,
)


def level_pose(x=0.0, y=1.4, z=0.0, yaw_deg=0.0):
    """Camera looking along -z (map frame), optionally turned right by yaw_deg."""
    a = math.radians(yaw_deg)
    fwd = np.array([math.sin(a), 0.0, -math.cos(a)])
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = right, -up, fwd
    pose[:3, 3] = [x, y, z]
    return pose


def test_sector_distances_use_a_low_percentile_not_the_median():
    depth = np.full((100, 90), 4.0)
    depth[60:, 0:30] = 0.8  # a chair leg-sized thing, but a big enough patch to pass the 5th pct
    left, centre, right = sector_distances_from_depth(depth)
    assert left < 1.0 and centre == 4.0 and right == 4.0


def test_sector_distances_ignore_the_ceiling():
    depth = np.full((100, 90), 4.0)
    depth[:20] = 0.3  # ceiling lamp in the top 30%
    assert min(sector_distances_from_depth(depth)) == 4.0


def test_sector_distances_handle_empty_strips():
    depth = np.zeros((10, 9))
    assert all(math.isinf(d) for d in sector_distances_from_depth(depth))


def test_sector_index_wraps_around_the_back():
    assert sector_index(0) == 0
    assert sector_index(44) == 1
    assert sector_index(180) == 4
    assert sector_index(-170) == 4
    assert sector_index(-90) == SECTORS.index("left")


def test_cue_levels_follow_the_plan():
    assert cue_for_distance(2.5).mode == "off"
    assert cue_for_distance(math.inf).mode == "off"
    assert cue_for_distance(1.5).mode == "slow"
    assert cue_for_distance(0.8).mode == "fast"
    assert cue_for_distance(0.3).mode == "solid"
    assert cue_for_distance(0.8).intensity > cue_for_distance(1.5).intensity


def test_merge_puts_camera_sectors_in_front_and_map_everywhere_else():
    from_map = [9.0] * 8
    merged = merge_sectors([1.0, 2.0, 3.0], from_map)  # front-left, front, front-right
    assert merged[SECTORS.index("front-left")] == 1.0
    assert merged[SECTORS.index("front")] == 2.0
    assert merged[SECTORS.index("front-right")] == 3.0
    assert merged[SECTORS.index("back")] == 9.0


def test_map_sectors_see_a_chair_behind_you():
    pose = level_pose()
    rng = np.random.default_rng(1)
    chair = rng.normal([0.0, 0.7, 0.9], 0.03, (200, 3))  # 0.9 m behind (+z is back)
    far_wall = rng.normal([0.0, 1.0, -2.8], 0.03, (200, 3))
    d = map_sector_distances(np.vstack([chair, far_wall]), pose, floor_y=0.0)
    assert d[SECTORS.index("back")] < 1.0
    assert d[SECTORS.index("front")] > 2.5
    assert math.isinf(d[SECTORS.index("left")])


def test_map_sectors_ignore_floor_and_ceiling():
    pose = level_pose()
    floor = np.column_stack(
        [
            np.random.default_rng(0).uniform(-1, 1, 300),
            np.zeros(300),
            np.random.default_rng(1).uniform(-1, 1, 300),
        ]
    )
    ceiling = floor + [0, 2.6, 0]
    d = map_sector_distances(np.vstack([floor, ceiling]), pose, floor_y=0.0)
    assert all(math.isinf(x) for x in d)


def test_map_sectors_turn_with_the_camera():
    behind_origin = np.random.default_rng(2).normal([0.0, 0.8, 1.0], 0.02, (100, 3))
    right = map_sector_distances(behind_origin, level_pose(yaw_deg=90), floor_y=0.0)
    # Facing +x, a point at +z is on the right-hand side (clockwise from forward).
    assert right[SECTORS.index("right")] < 1.2


def test_relative_bearing_signs():
    pose = level_pose()  # facing -z, right is +x
    assert relative_to(np.array([0, 1.4, -3.0]), pose).bearing_deg == 0
    assert round(relative_to(np.array([2.0, 1.4, 0.0]), pose).bearing_deg) == 90
    assert abs(relative_to(np.array([0, 1.4, 3.0]), pose).bearing_deg) == 180
    rel = relative_to(np.array([3.0, 0.9, 4.0]), pose)
    assert round(rel.horizontal_m, 2) == 5.0 and round(rel.dy_m, 2) == -0.5


def test_gravity_alignment_level_phone_and_either_accelerometer_sign():
    for sign in (1, -1):
        rot = gravity_alignment(sign * np.array([0.0, -9.8, 0.1]))
        assert np.allclose(rot, np.diag([1, -1, -1]), atol=0.02)
        assert np.isclose(np.linalg.det(rot), 1.0)


def test_gravity_alignment_tilted_down_phone_levels_the_map():
    t = math.radians(30)  # camera pitched 30 deg down: up in camera axes tilts toward +z
    up_cam = np.array([0.0, -math.cos(t), -math.sin(t)])
    rot = gravity_alignment(up_cam)
    assert np.allclose(rot @ up_cam, [0, 1, 0], atol=1e-6)


def test_tilted_floor_is_not_an_obstacle_but_a_table_is():
    rng = np.random.default_rng(1)
    xz = rng.uniform(-3, 3, (40000, 2))
    floor = np.c_[xz[:, 0], 0.15 * xz[:, 0], xz[:, 1]]  # floor tilting 15 cm per metre
    noise = np.c_[
        rng.uniform(-3, 3, (400, 1)),
        rng.uniform(-1.5, -0.6, (400, 1)),
        rng.uniform(-3, 3, (400, 1)),
    ]
    tx, tz = np.meshgrid(np.linspace(-0.5, 0.5, 30), np.linspace(-1.8, -1.2, 20))
    table = np.c_[tx.ravel(), 0.75 + 0.15 * tx.ravel(), tz.ravel()]
    pose = level_pose()
    d = map_sector_distances(np.vstack([floor, noise]), pose)
    assert all(math.isinf(v) for v in d)
    d = map_sector_distances(np.vstack([floor, noise, table]), pose)
    assert 1.1 < d[0] < 1.3 and all(math.isinf(v) for v in d[2:7])
