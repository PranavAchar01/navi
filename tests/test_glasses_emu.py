import math
import random

import cv2
import numpy as np

from metasight.glasses_emu import (
    Link,
    crop_hfov,
    head_track,
    imu_samples,
    transcript_steps,
)


def test_portrait_crop_of_landscape_footage_narrows_the_view():
    cols, hfov = crop_hfov(1280, 720, 105.0)
    assert cols.stop - cols.start == 405
    assert abs(hfov - 44.8) < 0.3
    assert crop_hfov(720, 1280, 61.0)[1] == 61.0  # portrait source used whole


def _texture(w=480, h=270, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h // 6, w // 6), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST), (5, 5), 0)


def test_turning_the_head_left_reads_positive_yaw_on_the_gyro():
    hfov, fps, deg = 105.0, 30.0, 3.0
    img = _texture()
    h, w = img.shape
    f = 0.5 * w / math.tan(math.radians(hfov) / 2)
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    cam, _ = cv2.Rodrigues(np.array([0.0, -math.radians(deg), 0.0]))  # left: about OpenCV -y
    warped = cv2.warpPerspective(img, k @ cam.T @ np.linalg.inv(k), (w, h))
    track = head_track([img, warped], fps, hfov)
    gx, gy, gz = track.omega[0]
    assert abs(gy - math.radians(deg) * fps) < 0.15 * math.radians(deg) * fps
    assert abs(gx) < 0.2 and abs(gz) < 0.2
    assert np.allclose(track.up[0], [0, 1, 0])  # starts level: gravity opposes +y


def test_imu_reads_gravity_on_up_and_runs_at_the_asked_rate():
    img = _texture()
    track = head_track([img, img, img], 30.0, 105.0)
    s = imu_samples(track, 30, random.Random(0))
    assert len(s) == 2 and abs(s[1][0] - 1 / 30) < 1e-9
    ax, ay, az, gx, gy, gz = s[0][1]
    assert 9.0 < ay < 10.5 and abs(ax) < 0.5 and abs(gy) < 0.02


def test_link_steps_resolution_then_frame_rate_and_recovers():
    link = Link("medium", 24, 2.0e6, [(10.0, 5.0, 0.6), (20.0, 5.0, 0.3)])
    link.update(0.0)
    assert (link.res, link.fps) == ("medium", 24)
    link.update(10.0)
    assert (link.res, link.fps) == ("low", 24)  # resolution first
    link.update(20.0)
    link.update(20.1)
    assert (link.res, link.fps) == ("low", 15)  # then frame rate, not below 15
    for t in np.arange(25.0, 40.0, 0.5):
        link.update(float(t))
    assert (link.res, link.fps) == ("medium", 24)


def test_speech_gives_partials_then_one_final():
    steps = transcript_steps("where is my water bottle?", 40.0)
    assert [s[2] for s in steps] == [False] * 5 + [True]
    assert steps[-1][1] == "where is my water bottle?" and steps[0][0] > 40.0
