import numpy as np

from metasight.memory import _rotation_to


def test_rotation_to_takes_down_to_minus_y():
    down = np.array([0.3, 0.9, 0.1])
    down /= np.linalg.norm(down)
    target = np.array([0.0, -1.0, 0.0])
    r = _rotation_to(down, target)
    assert np.allclose(r @ down, target, atol=1e-6)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
