import cv2
import numpy as np

from metasight.keyframes import KeyframePolicy, moved, thumb

rng = np.random.default_rng(0)
SCENE = cv2.GaussianBlur((rng.random((480, 1600)) * 255).astype(np.uint8), (0, 0), 4)


def view(x: int, scene=SCENE) -> np.ndarray:
    return scene[:, x : x + 640]


def run(pol, frames, fps=30):
    kept = []
    for i, f in enumerate(frames):
        if pol.offer(f, 1.0, i / fps, i) is not None:
            kept.append(i)
    return kept


def test_turn_is_measured_from_the_slide():
    turn, match = moved(thumb(view(0)), thumb(view(160)), 60.0)  # a quarter of the width
    assert 12 < turn < 18 and match > 0.75


def test_standing_still_keeps_one_frame_until_the_gap():
    kept = run(KeyframePolicy(60.0, max_gap_s=4.0), [view(0)] * 90)  # 3 s
    assert kept == [0]


def test_turning_keeps_a_frame_every_15_degrees():
    frames = [view(x) for x in range(0, 900, 6)]  # 900 px = 84 deg over 5 s
    kept = run(KeyframePolicy(60.0), frames)
    assert 4 <= len(kept) <= 7


def test_a_different_scene_is_kept_and_the_sharpest_wins():
    other = cv2.GaussianBlur((rng.random((480, 1600)) * 255).astype(np.uint8), (0, 0), 4)
    pol = KeyframePolicy(60.0)
    assert pol.offer(view(0), 1.0, 0.0, "a") == "a"
    out = [pol.offer(view(0, other), s, 0.5 + 0.1 * i, i) for i, s in enumerate([1, 5, 2, 1])]
    assert out[-1] == 1
