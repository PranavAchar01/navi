import math

import numpy as np

from metasight import nav
from metasight.belt import MODES, plan

# Two rooms along z, a wall at z = 4 with a 0.6 m door at x 1.7 to 2.3. Boxes: x0, z0, x1, z1, top.
WALLS = [
    (0.0, 0.0, 4.0, 0.1, 2.5),
    (0.0, 7.9, 4.0, 8.0, 2.5),
    (0.0, 0.0, 0.1, 8.0, 2.5),
    (3.9, 0.0, 4.0, 8.0, 2.5),
    (0.0, 3.95, 1.7, 4.05, 2.5),
    (2.3, 3.95, 4.0, 4.05, 2.5),
]
TABLE = (1.5, 1.6, 2.5, 2.2, 0.45)  # a coffee table in the first room


def _plan(*extra):
    return nav.plan_from_rects([*WALLS, *extra], 0, 0, 4, 8)


def test_route_goes_through_a_door_a_fixed_inflation_would_close():
    p = _plan()
    n = nav.Navigator(p)
    assert n.set_goal([2.0, 0.5, 7.0])
    g = n.step([2.0, 1.6, 1.0], yaw=0.0)
    assert g is not None and abs(g.bearing_deg) < 15
    assert 5.5 < g.route_m < 7.5
    assert g.narrow_m < 3.5  # the door is on the route
    # Walk the route by following the bearing: it must pass the door near its middle.
    pos, yaw, door_x = np.array([0.6, 1.6, 0.6]), 0.0, []
    for _ in range(400):
        g = n.step(pos, yaw)
        assert g is not None
        if g.arrived:
            break
        yaw -= math.radians(g.bearing_deg)
        pos += 0.05 * np.array([math.sin(yaw), 0, math.cos(yaw)])
        if 3.9 < pos[2] < 4.1:
            door_x.append(pos[0])
    assert g.arrived
    assert door_x and all(1.85 < x < 2.15 for x in door_x)
    # The door's middle is under 0.32 m from its frame: hapsim's fixed inflation closes it.
    i, j = p.ij(2.0, 4.0)
    assert p.clear[i, j] < 0.32 and p.free(i, j)


def test_goal_on_a_table_routes_to_the_floor_beside_it():
    p = _plan(TABLE)
    n = nav.Navigator(p)
    assert n.set_goal([2.0, 0.5, 1.9])  # keys on the table
    g = n.step([2.0, 1.6, 6.0], yaw=math.pi)  # other room, facing back toward the door
    assert g is not None and math.isfinite(g.route_m)


def test_bearing_sign_matches_heading():
    n = nav.Navigator(_plan())
    n.set_goal([1.0, 0.5, 2.0])
    # Standing at x 3 facing +z, the goal is ahead and toward -x: with forward (sin, cos) and
    # right (-cos, sin), -x is to the right.
    g = n.step([3.0, 1.6, 1.0], yaw=0.0)
    assert g is not None and g.bearing_deg > 20


def test_low_table_under_the_frame_is_a_hidden_obstacle():
    p = _plan(TABLE)
    pri = nav.priors(p, [2.0, 1.6, 0.9], 0.0, [math.inf] * 3, hfov_deg=60, below_deg=35)
    assert pri.active
    assert pri.hidden[0] < 1.0 and pri.hidden_low[0]


def test_new_vs_known_and_contact():
    p = _plan()
    # Walking along the x = 0.1 wall, 0.5 m from it, facing +z. The wall is to the right
    # (-x is right), and the camera with a wide view reads it where the map has it.
    pos = [0.6, 1.6, 2.0]
    pri = nav.priors(p, pos, 0.0, [math.inf, math.inf, 0.75], hfov_deg=120, speed_mps=0.8)
    assert pri.known[1] is True
    # Something at 0.6 m dead ahead that the map does not have (the next wall is 2 m away).
    pri = nav.priors(p, pos, 0.0, [math.inf, 0.6, math.inf], hfov_deg=60, speed_mps=0.8)
    assert pri.known[0] is False
    # Walking at 1 m/s toward the door wall 1.2 m ahead: contact about 0.9 s out.
    pri = nav.priors(p, [1.0, 1.6, 2.8], 0.0, [math.inf] * 3, hfov_deg=60, speed_mps=1.0)
    assert 0.6 < pri.contact_s < 1.2 and abs(pri.contact_bearing) < 30
    # A stale pose turns the priors off.
    assert not nav.priors(p, pos, 0.0, [math.inf] * 3, 60, pos_err_m=1.0).active


def test_apply_raises_a_new_obstacle_and_calms_only_map_cues_beside_you():
    sectors = [0.6, 0.7, 0.8, math.inf, math.inf, math.inf, math.inf, math.inf]
    cmd, _ = plan(sectors, [True] * 8, math.inf, math.inf, math.inf)
    assert cmd[1][0] == MODES["fast"] and cmd[2][0] == MODES["fast"]
    pri = nav.Priors([math.inf] * 8, [False] * 8, [None] * 8, math.inf, math.inf, 0.0, True)
    pri.known[1] = True  # the wall to the front-right is on the map: live depth, left alone
    pri.known[0] = False  # what is ahead is not
    before = list(cmd)
    did = nav.apply(cmd, pri, sectors)
    assert cmd[1] == before[1]
    assert cmd[2][0] == MODES["slow"] and did["calm"] == 1  # right: map-only, calmed
    assert cmd[0][0] == MODES["solid"] and did["new"] == 1


def test_priors_never_weaken_a_live_depth_cue():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        sectors = list(rng.choice([0.3, 0.6, 0.8, 1.2, 1.8, math.inf], 8))
        cmd, _ = plan(sectors, [True] * 8, math.inf, math.inf, math.inf)
        pri = nav.Priors(
            list(rng.choice([0.3, 0.7, 1.5, math.inf], 8)),
            [False] * 8,
            list(rng.choice([True, False, None], 8)),
            float(rng.choice([0.5, 1.5, math.inf])),
            1.0,
            float(rng.uniform(-180, 180)),
            True,
        )
        before = list(cmd)
        nav.apply(cmd, pri, sectors)
        for s in nav.CAMERA_IDX:
            if before[s][0] in (MODES["slow"], MODES["fast"], MODES["solid"]):
                assert cmd[s][0] >= before[s][0] and cmd[s][1] >= before[s][1]
        for s in range(8):
            assert before[s][0] != MODES["solid"] or cmd[s] == before[s]
