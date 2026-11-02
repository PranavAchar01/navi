import math

import numpy as np

from metasight import recon


def _desk_points(yaw: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    p = rng.uniform([-0.6, 0.72, -0.3], [0.6, 0.75, 0.3], (4000, 3))
    c, s = math.cos(yaw), math.sin(yaw)
    x, z = p[:, 0] * c - p[:, 2] * s, p[:, 0] * s + p[:, 2] * c
    return np.column_stack([x + 1.0, p[:, 1], z - 2.0])


def test_upright_box_recovers_rotated_desk():
    box = recon.upright_box(_desk_points(math.radians(30)), lo=0, hi=100)
    sx, _, sz = box["size"]
    assert abs(max(sx, sz) - 1.2) < 0.03 and abs(min(sx, sz) - 0.6) < 0.03
    assert abs(box["center"][0] - 1.0) < 0.02 and abs(box["center"][2] + 2.0) < 0.02
    # yaw is defined modulo 90 degrees for a rectangle
    assert min(abs(math.degrees(box["yaw"]) - d) for d in (30, 120, -60)) < 3


def test_snap_keeps_a_slightly_turned_desk_square_to_the_walls():
    box = recon.upright_box(_desk_points(math.radians(8)), lo=0, hi=100, snap=1.35)
    assert box["yaw"] == 0.0
    box = recon.upright_box(_desk_points(math.radians(40)), lo=0, hi=100, snap=1.35)
    assert box["yaw"] != 0.0


def test_corners_and_footprint_stay_inside_box():
    box = {"center": [0.0, 0.4, 0.0], "size": [1.2, 0.8, 0.6], "yaw": 0.0}
    c = recon.box_corners(box, 0.8)
    assert np.allclose(np.abs(c[:, [0, 2]]), [0.6, 0.3]) and np.allclose(c[:, 1], 0.8)
    g = recon.footprint_grid(box)
    assert (np.abs(g[:, 0]) <= 0.6).all() and (np.abs(g[:, 1]) <= 0.3).all() and len(g) > 10


def test_clearance_ignores_legs_and_solid_is_zero():
    under = np.full(100, 0.70)
    under[:15] = 0.02  # legs and a bin
    assert abs(recon.clearance(under, 0.0, 0.74) - 0.70) < 1e-6
    assert recon.clearance(np.full(50, 0.01), 0.0, 0.9) == 0.0
    assert recon.clearance(np.full(50, np.inf), 0.0, 0.9) == 0.9  # nothing above: up to the top


def test_edges_include_underside_only_with_room_beneath():
    box = {"center": [0.0, 0.37, 0.0], "size": [1.2, 0.74, 0.6], "yaw": 0.0}
    e = recon.edges(box, 0.0, 0.70)
    assert [x["kind"] for x in e] == ["top", "underside"]
    assert e[0]["points"][0] == e[0]["points"][-1] and len(e[0]["points"]) == 5
    assert [x["kind"] for x in recon.edges(box, 0.0, 0.0)] == ["top"]


def test_mass_is_clipped_to_prior_range():
    kg, material, movable = recon.mass_material("keyboard", [3.0, 3.0, 3.0])
    assert kg == 1.6 and material == "plastic" and movable
    assert recon.mass_material("unknown thing", [1, 1, 1]) == (None, None, True)


def test_continuity_and_trusted_flag_jumps():
    pos = np.array([[0, 0, 0], [0.02, 0, 0], [2.0, 0, 0], [0.04, 0, 0], [0.06, 0, 0]], float)
    assert abs(recon.continuity(pos) - 0.5) < 1e-9
    assert recon.trusted(pos).tolist() == [True, False, False, False, True]
