import numpy as np

from metasight.memory import _rotation_to


def test_rotation_to_takes_down_to_minus_y():
    down = np.array([0.3, 0.9, 0.1])
    down /= np.linalg.norm(down)
    target = np.array([0.0, -1.0, 0.0])
    r = _rotation_to(down, target)
    assert np.allclose(r @ down, target, atol=1e-6)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)


def _at(seq, ts, x):
    from metasight.memory import Keyframe, _Placed

    pose = np.eye(4)
    pose[0, 3] = x
    kf = Keyframe(seq, ts, np.zeros((4, 4, 3), np.uint8))
    return _Placed(kf, np.zeros((4, 4, 3)), np.ones((4, 4), bool), pose)


def test_walkable_drops_frames_placed_metres_off_and_keeps_walking():
    from metasight.memory import walkable

    overlap = _at(0, 0.0, 0.0)
    # 004732: two frames of a window put 2.7 m away within a second, then back on the path.
    window = [_at(1, 1.0, 0.8), _at(2, 2.0, 3.5), _at(3, 3.0, 3.6), _at(4, 4.0, 1.9)]
    assert walkable(window, overlap) == [0, 3]
    assert walkable([_at(1, 1.0, 3.0), _at(2, 2.0, 3.4)], None) == [0, 1]
