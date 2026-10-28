import math

import cv2
import numpy as np

from metasight.loops import compose, inverse, optimize, shared_matches


def _sim(yaw_deg: float, t, s: float = 1.0):
    rot, _ = cv2.Rodrigues(np.array([0.0, math.radians(yaw_deg), 0.0]))
    return s, rot, np.array(t, float)


def test_a_loop_pulls_a_drifted_chain_back_onto_the_start():
    # Truth: eight windows walking a square and ending where they started.
    truth = [_sim(0, [0, 0, 0])]
    corners = [[1, 0, 0], [2, 0, 0], [2, 0, 1], [2, 0, 2], [1, 0, 2], [0, 0, 2], [0, 0, 1]]
    truth += [_sim(0, c) for c in corners]
    truth.append(_sim(0, [0.0, 0.0, 0.0]))
    # Each measured step turns 3 degrees too far, so the chain ends away from the start.
    rel = [None] + [compose(inverse(truth[w - 1]), truth[w]) for w in range(1, len(truth))]
    rel = [None] + [compose(_sim(3, [0, 0, 0]), r) for r in rel[1:]]
    chain = [truth[0]]
    for r in rel[1:]:
        chain.append(compose(chain[-1], r))
    end_before = np.linalg.norm(chain[-1][2])
    loop = (0, len(truth) - 1, compose(inverse(truth[0]), truth[-1]))
    fixed = optimize(chain, rel, [loop])
    assert end_before > 0.25
    assert np.linalg.norm(fixed[-1][2]) < 0.1 * end_before
    assert np.allclose(fixed[0][2], 0)  # the first window holds still


def test_without_a_loop_nothing_moves():
    chain = [_sim(0, [0, 0, 0]), _sim(10, [1, 0, 0])]
    assert optimize(chain, [None, chain[1]], []) is chain


def _scene(seed: int, w=480, h=360):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 200, np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.rectangle(
            img,
            (x, y),
            (x + rng.integers(8, 40), y + rng.integers(8, 40)),
            tuple(int(c) for c in rng.integers(0, 255, 3)),
            -1,
        )
    return img


def test_the_same_place_from_a_step_aside_matches_and_another_place_does_not():
    a = _scene(1)
    shifted = cv2.warpAffine(a, np.float32([[1, 0, 25], [0, 1, 10]]), (a.shape[1], a.shape[0]))
    assert shared_matches(a, shifted) >= 40
    assert shared_matches(a, _scene(2)) < 40


def test_a_bare_handful_of_matches_never_counts_as_many():
    a = _scene(3)
    b = cv2.warpAffine(_scene(4), np.float32([[1, 0, 5], [0, 1, 0]]), (a.shape[1], a.shape[0]))
    b[:120, :160] = a[:120, :160]  # a small shared patch: a few true matches only
    assert shared_matches(a, b) <= 200
