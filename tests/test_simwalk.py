import pytest

from metasight import hapsim as h
from metasight import pan

pytest.importorskip("mujoco")
from metasight import simwalk as S  # noqa: E402


def test_every_scenario_builds_with_the_chair_and_its_goal_inside_the_room():
    for name in S.SCENARIOS:
        sc = S.scenario(name)
        w, d = sc.world.size
        assert 0 < sc.goal[0] < w and 0 < sc.goal[1] < d
        assert sc.world.rects()  # walls and furniture reach the kinematic walker
    find = S.scenario("find_chair")
    assert find.target == "chair" and find.guide and find.sit
    assert S.scenario("avoid_chair").hazard == "chair"


def test_a_walk_runs_the_belt_pipeline_and_reports_every_metric():
    sc = S.scenario("low_box")
    belt = h.Belt(layout="diag4", hfov=S.HFOV, mapping=pan.DEFAULT)
    tr = S.walk(sc, belt, seed=1000)
    m = tr.metrics()
    assert set(m) >= {"success", "collisions", "min_clear_m", "time_s", "false_per_min", "cost"}
    assert tr.steps and len(tr.steps[0][4]) == 4  # felt levels for the four built motors
    assert m["min_clear_m"] >= 0


def test_cueing_the_second_object_keeps_the_walker_off_the_uncued_side():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_sim_walk as R

    sc = S.scenario("find_chair")
    one = S.walk(sc, R.belt_for("I5", "eight"), seed=2001, record=False, **R.ARMS["I5"][2])
    two = S.walk(sc, R.belt_for("I6", "eight"), seed=2001, record=False, **R.ARMS["I6"][2])
    assert one.metrics()["collisions"] > 0
    assert two.metrics()["collisions"] == 0 and two.metrics()["success"]
