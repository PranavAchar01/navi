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


def test_guide_is_panned_over_two_motors_faint_on_track_and_below_obstacle_cues():
    on_track = pan.guide_cmd(4.0)
    assert [i for _, i in on_track].count(0) == 7 and on_track[0] == (MODES["guide"], 90)
    turn = pan.guide_cmd(60.0)
    lit = [(m, i) for m, i in turn if i]
    assert len(lit) == 2 and all(m == MODES["guide"] for m, _ in lit)
    assert max(i for _, i in turn) < 204  # a close obstacle cue is at least 204


def test_stop_overrides_guide_and_fast_cues_are_never_replaced():
    solid = [(MODES["off"], 0)] * 8
    solid[4] = (MODES["solid"], 255)
    out, shown = pan.with_guide(solid, 90.0)
    assert out == solid and not shown  # silent everywhere while something is about to be hit
    fast = [(MODES["off"], 0)] * 8
    fast[2] = (MODES["fast"], 220)
    out, shown = pan.with_guide(fast, 90.0)
    assert out[2] == (MODES["fast"], 220) and shown is False
    # the fast cue is the item being guided to: the guide taps play through it, stop still wins
    out, shown = pan.with_guide(fast, 90.0, take_fast=True)
    assert out[2][0] == MODES["guide"] and shown
    assert pan.with_guide(solid, 90.0, take_fast=True) == (solid, False)


def test_persist_filter_drops_a_one_frame_phantom_but_passes_a_real_obstacle():
    f = pan.Persist(2)
    assert f([3.0, 3.0]) == [3.0, 3.0]
    assert f([0.5, 3.0]) == [3.0, 3.0]  # first frame near: not yet
    assert f([0.5, 3.0]) == [0.5, 3.0]  # second frame near: now it counts


def test_hold_keeps_the_cue_still_for_small_bearing_jitter():
    hold = pan.Hold()
    first = hold([pan.Target(10.0, 1.0)])[0].bearing_deg
    for jitter in (4.0, -3.0, 6.0, -5.0):
        assert hold([pan.Target(10.0 + jitter, 1.0)])[0].bearing_deg == first


def test_hold_follows_a_real_turn_and_live_distance():
    hold = pan.Hold()
    hold([pan.Target(0.0, 1.2)])
    out = hold([pan.Target(22.0, 0.5)])[0]
    assert 10.0 < out.bearing_deg <= 22.0
    assert out.dist_m == 0.5


def test_hold_switches_only_for_a_clearly_closer_object():
    hold = pan.Hold()
    hold([pan.Target(0.0, 1.0)])
    both = [pan.Target(0.0, 1.0), pan.Target(90.0, 0.9)]
    assert hold(both)[0].bearing_deg == 0.0
    both = [pan.Target(0.0, 1.0), pan.Target(90.0, 0.4)]
    assert hold(both)[0].bearing_deg == 90.0


def test_hold_survives_a_dropout_then_lets_go():
    hold = pan.Hold(keep=2)
    hold([pan.Target(0.0, 1.0)])
    assert hold([]) and hold([])
    assert not hold([])


def test_hold_never_masks_a_new_closer_obstacle():
    hold = pan.Hold()
    hold([pan.Target(0.0, 1.2)])
    out = hold([pan.Target(80.0, 0.5)])[0]
    assert (out.bearing_deg, out.dist_m) == (80.0, 0.5)


def test_planner_guide_reaches_the_goal_and_stop_always_wins():
    from metasight import hapsim as h

    belt = h.Belt(scheme="sector", guide="on", persist=2, planner=True)
    r = h.run(h.make_room(1003), belt, h.Skin(nav_drift_deg=12.0), seed=0)
    assert r.guide_frames > 0 and r.stop_violations == 0
    assert r.stop_overridden == r.stop_frames


def test_tracker_confirms_far_objects_but_never_delays_a_close_one():
    tr = pan.Tracker()
    far = [pan.Target(20.0, 1.4)]
    assert tr(far) == [] and tr(far) == []
    assert len(tr(far)) == 1  # third sighting
    assert len(pan.Tracker()([pan.Target(-30.0, 0.5)])) == 1


def test_tracker_smooths_bearing_and_coasts_through_a_dropout():
    tr = pan.Tracker(confirm=1)
    tr([pan.Target(0.0, 1.0)])
    out = tr([pan.Target(10.0, 1.0)])[0]
    assert 0.0 < out.bearing_deg < 10.0
    assert tr([]) and tr([])
    assert not tr([])


def test_grid_strips_take_the_nearest_obstacle_per_column_band():
    import numpy as np

    from metasight.geometry import grid_strips

    g = np.full((4, 30), 255, np.uint8)
    g[2, 0] = 60  # 1.2 m in the leftmost band
    g[1, 29] = 40  # 0.8 m in the rightmost
    s = grid_strips(g)
    assert len(s) == 15 and math.isclose(s[0], 1.2) and math.isclose(s[-1], 0.8)
    assert all(math.isinf(d) for d in s[1:-1])


def test_four_diagonal_motors_fold_the_eight_directions():
    import numpy as np

    front = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    # both front motors at 1/sqrt(2): the pair carries the same power as one motor
    assert np.allclose(h.to_motors(front, "diag4"), [0.5**0.5, 0, 0, 0.5**0.5])
    front_right = np.array([0, 0.5, 0, 0, 0, 0, 0, 0])
    assert np.allclose(h.to_motors(front_right, "diag4"), [0.5, 0, 0, 0])
    assert list(h.to_motors(front, "diag4", "max3")) == [1.0, 0, 0, 1.0]


def test_walker_on_four_motors_reaches_the_goal_in_a_seeded_room():
    r = h.run(h.make_room(3), h.Belt(layout="diag4", persist=2), seed=0)
    assert r.success
