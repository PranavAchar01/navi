import math

import cv2
import numpy as np

from metasight.heading import Heading, image_yaw


def test_bearing_and_gyro_turn_sign():
    h = Heading(90.0)
    h.anchor(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    # The item is straight ahead; turn the head left 30 degrees and it should sit to the right.
    assert abs(h.to(np.array([0.0, 0.0, 2.0]))[1]) < 1e-6
    for _ in range(30):
        h.gyro(np.array([0.0, math.radians(30) / 0.3, 0.0]), np.array([0.0, 1.0, 0.0]), 0.01, 1)
    dist, bearing = h.to(np.array([0.0, 0.0, 2.0]))
    assert dist == 2.0 and abs(bearing - 30) < 0.5


def test_bad_pose_is_ignored_and_new_pose_blends():
    h = Heading(90.0)
    h.anchor(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert not h.anchor(np.zeros(3), np.array([0.3, -0.95, 0.1]))  # looking at the floor
    assert abs(h.yaw) < 1e-9
    h.anchor(np.zeros(3), np.array([1.0, 0.0, 0.0]))  # a jump of 90 degrees moves it only partly
    assert 0 < h.yaw < math.radians(45)


def test_image_shift_sign_turning_left():
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.random((90, 400)).astype(np.float32), (5, 5), 0)
    prev, cur = scene[:, 100:260], scene[:, 90:250]  # view slides left: the scene moves right
    yaw = image_yaw(np.ascontiguousarray(prev), np.ascontiguousarray(cur), 90.0)
    assert yaw is not None and abs(math.tan(yaw) * 80 / math.tan(math.radians(45)) - 10) < 0.5


def test_late_pose_is_matched_at_its_keyframe_time():
    h = Heading(90.0)
    up = np.array([0.0, 1.0, 0.0])
    h.anchor(np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0)
    # Turn left 90 degrees between t=1 and t=2, then a pose arrives for the keyframe at t=0.5
    # that still faces the start direction: it must not undo the turn.
    for k in range(100):
        h.gyro(np.array([0.0, math.radians(90), 0.0]), up, 0.01, 1 + k * 0.01)
    h.anchor(np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.5)
    assert abs(math.degrees(h.yaw) - 90) < 1.5


def test_image_turn_gain_is_learned_from_map_poses():
    h = Heading(90.0)
    h.gyro_t = -1e9
    for k in range(6):  # the image reads 40 degrees per step, the map 30
        h._add(h.gain * math.radians(40), k + 0.5, math.radians(40))
        h.anchor(
            np.zeros(3),
            np.array([math.sin(math.radians(30 * k)), 0, math.cos(math.radians(30 * k))]),
            k + 0.6,
        )
    assert abs(h.gain - 0.75) < 0.02


def test_camera_sighting_pulls_the_heading_to_agree():
    h = Heading(90.0)
    h.anchor(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    item = np.array([0.0, 0.0, 2.0])  # the map says dead ahead
    assert abs(h.box_bearing((0.7, 0.4, 0.8, 0.6)) - math.degrees(math.atan(0.5))) < 1e-6
    for _ in range(30):
        h.sight(item, 20.0)  # the camera sees it 20 degrees to the right
    assert abs(h.to(item)[1] - 20.0) < 0.5


def test_map_sectors_turn_with_the_head():
    from metasight.server import Server

    srv = Server.__new__(Server)
    srv.heading = Heading(90.0)
    srv.heading.anchor(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    srv.map_yaw = 0.0
    srv.map_sectors = [1.0, 2, 3, 4, 5, 6, 7, 8.0]  # front, front-right, ... front-left
    for _ in range(10):  # turn left 45 degrees
        srv.heading.gyro(
            np.array([0.0, math.radians(450), 0.0]), np.array([0.0, 1.0, 0.0]), 0.01, 1
        )
    assert srv._map_sectors_now() == [8.0, 1.0, 2, 3, 4, 5, 6, 7]
