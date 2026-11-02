import math

import numpy as np
import pytest

from metasight import hapsim as h

pytest.importorskip("mujoco")
from metasight.hapsim3d import Sensor3D  # noqa: E402


def test_rendered_depth_sees_a_wall_at_the_right_range_and_the_floor():
    room = h.make_room(1000)
    s = Sensor3D(room)
    try:
        z = s.depth(room.size[0] - 1.5, 2.75, 0.0)  # 1.5 m from the far wall, facing it
        assert z[54, 96] == pytest.approx(1.5 / math.cos(math.radians(15)), abs=0.03)
        strips, sectors = s(room, *room.start, np.random.default_rng(0))
        assert len(strips) == h.STRIPS and len(sectors) == 8
        assert s.prev is not None  # the floor plane was found in the render
    finally:
        s.close()
