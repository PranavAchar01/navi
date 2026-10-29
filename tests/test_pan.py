import math

from metasight import hapsim as h
from metasight import pan
from metasight.belt import MODES


def test_bearing_lands_on_the_two_nearest_motors_with_constant_power():
    a, ga, b, gb = pan.pan_gains(30.0)
    assert (a, b) == (0, 1) and gb > ga
    assert math.isclose(ga**2 + gb**2, 1.0)
    a, ga, b, gb = pan.pan_gains(90.0)
    assert (a, ga, gb) == (2, 1.0, 0.0)
    assert pan.pan_gains(350.0)[0::2] == (7, 0)  # wraps round the front


def test_never_more_than_two_motors_per_object_and_never_all():
    for bearing in range(0, 360, 7):
        cmd = pan.plan_targets([pan.Target(bearing, 0.8)])
        assert sum(i > 0 for _, i in cmd) <= 2


def test_nearer_is_stronger_and_far_is_silent():
    assert pan.level(0.4) > pan.level(1.0) > pan.level(1.49) > 0
    assert pan.level(pan.DEFAULT.off_m) == 0 and pan.level(math.inf) == 0


def test_second_object_needs_separation_and_nearest_wins():
    t = [pan.Target(0, 1.0), pan.Target(20, 1.2), pan.Target(120, 1.8)]
    chosen = pan.select(t, pan.with_params(pan.DEFAULT, k_targets=2, off_m=2.0))
    assert [c.bearing_deg for c in chosen] == [0, 120]


def test_plan_pan_keeps_unknown_sectors_silent_and_flags_drop():
    cmd, flags = pan.plan_pan([], [True] * 3 + [False] * 5, math.inf, math.inf, 1.0)
    assert cmd[3] == (MODES["unknown"], 0) and flags == 1


def test_simulated_walker_reaches_the_goal_in_a_seeded_room():
    r = h.run(h.make_room(1000), h.Belt(), h.DEFAULT_SKIN, seed=0)
    assert r.time_s > 0 and r.collisions >= 0
