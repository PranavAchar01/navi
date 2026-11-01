import numpy as np
import open3d as o3d

from metasight.roomprior import RoomPrior
from metasight.safety import Obstacle

K = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]])  # normalised, 90 degrees across


def room(tmp_path) -> RoomPrior:
    """A 6 x 3 x 6 m box around a camera at the origin looking down +z, with a chair against the
    far wall and a tv on it."""
    box = o3d.geometry.TriangleMesh.create_box(6, 3, 6).translate((-3, -1, -3))
    path = tmp_path / "room.ply"
    o3d.io.write_triangle_mesh(str(path), box)
    objs = [
        {"label": "chair", "pos": [0.0, 0.3, 2.8], "views": 50},
        {"label": "tv", "pos": [-1.2, 0.5, 2.9], "views": 50},
        {"label": "laptop", "pos": [0.0, 0.0, -2.0], "views": 50},  # behind the camera
        {"label": "bottle", "pos": [0.5, 0.0, 2.0], "views": 3},  # too few views to count
    ]
    return RoomPrior(objs, path)


def ob(label: str, score: float, box: tuple) -> Obstacle:
    return Obstacle(label, score, box, 2.0)


def test_visible_skips_objects_behind_or_unsure(tmp_path):
    p = room(tmp_path)
    seen = {p.labels[i]: uv for i, uv, _ in p.visible(np.eye(4), K)}
    assert set(seen) == {"chair", "tv"}
    assert np.allclose(seen["chair"], [0.5, 0.5 + 0.3 / 2.8], atol=1e-3)


def test_confirm_keeps_weak_box_only_over_a_saved_object(tmp_path):
    p = room(tmp_path)
    on = ob("chair", 0.25, (0.4, 0.5, 0.6, 0.7))
    off = ob("chair", 0.25, (0.0, 0.0, 0.2, 0.2))
    other = ob("laptop", 0.25, (0.4, 0.5, 0.6, 0.7))
    assert p.apply([on, off, other], np.eye(4), K) == [on]


def test_reject_drops_furniture_far_from_its_saved_place(tmp_path):
    p = room(tmp_path)
    here = ob("chair", 0.5, (0.4, 0.5, 0.6, 0.7))  # its ray meets the wall by the saved chair
    there = ob("chair", 0.5, (0.0, 0.4, 0.1, 0.6))  # meets the wall 1.5 m to the left
    sure = ob("chair", 0.7, (0.0, 0.4, 0.1, 0.6))  # confident boxes are left alone
    unknown = ob("couch", 0.5, (0.0, 0.4, 0.1, 0.6))  # the room has no couch to compare with
    kept = p.apply([here, there, sure, unknown], np.eye(4), K)
    assert kept == [here, sure, unknown]
