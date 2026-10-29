"""Phone mode: ARKit carries the map pose between memory windows, in the map's own axes."""

import asyncio
import math
from collections import deque
from types import SimpleNamespace

import numpy as np

from metasight.heading import Heading
from metasight.server import Server, arkit_pose


def ar_matrix(yaw_deg: float, pos) -> list[float]:
    """ARKit camera-to-world for a phone held upright, turned `yaw_deg` left of -z (ARKit
    looks down its own -z), column-major as the phone sends it."""
    # Upright phone: the landscape sensor's x points down the world, its y to the image's right.
    y = math.radians(yaw_deg)
    fwd = np.array([-math.sin(y), 0.0, -math.cos(y)])  # world, ARKit y up
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    r = np.column_stack([[0.0, -1.0, 0.0], right, -fwd])  # sensor x, y, z in world
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, pos
    return m.T.reshape(-1).tolist()


def test_arkit_pose_is_in_upright_opencv_axes():
    p = arkit_pose(ar_matrix(0.0, [1.0, 1.5, 2.0]))
    assert np.allclose(p[:3, 1], [0, -1, 0])  # image down is world down
    assert np.allclose(p[:3, 2], [0, 0, -1])  # looking where ARKit looks
    assert np.allclose(p[:3, 3], [1.0, 1.5, 2.0])
    assert np.isclose(np.linalg.det(p[:3, :3]), 1.0)


def test_a_memory_pose_ties_arkit_to_the_map_and_arkit_carries_it():
    sent = []

    async def json(msg, _keep=False):
        sent.append(msg)

    srv = SimpleNamespace(
        arposes=deque(maxlen=900),
        ar_align=None,
        ar_pose_ts=0.0,
        ar_sent=0.0,
        pose=None,
        heading=Heading(60.0),
        hub=SimpleNamespace(json=json),
    )
    run = lambda m: asyncio.run(Server.on_arpose(srv, m))  # noqa: E731
    run({"type": "arpose", "seq": 1, "m": ar_matrix(0.0, [0, 1.5, 0]), "track": "tracking"})
    assert srv.pose is None and not sent  # nothing ties ARKit to the map yet

    # The map's world is ARKit's turned 90 degrees and moved: a memory pose for frame 1 says so.
    g = np.eye(4)
    g[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    g[:3, 3] = [5.0, 0.0, -2.0]
    Server._align_arkit(srv, 1, g @ srv.arposes[-1][1])

    run({"type": "arpose", "seq": 2, "m": ar_matrix(30.0, [0, 1.5, -1]), "track": "tracking"})
    want = g @ arkit_pose(ar_matrix(30.0, [0, 1.5, -1]))
    assert np.allclose(srv.pose, want) and srv.ar_pose_ts > 0
    assert np.allclose(srv.heading.pos, want[:3, 3])
    assert math.isclose(srv.heading.yaw, math.atan2(want[0, 2], want[2, 2]), abs_tol=1e-9)
    assert sent and sent[-1]["type"] == "pose"

    before = srv.pose
    run({"type": "arpose", "seq": 3, "m": ar_matrix(60.0, [0, 1.5, -2]), "track": "limited"})
    assert srv.pose is before  # lost tracking moves nothing
