"""The floor is found, and is no obstacle, in landscape and portrait (9:16 glasses) frames."""

import math

import numpy as np
import pytest

from metasight.safety import analyse


def floor_depth(w: int, h: int, hfov_deg: float, cam_h: float, pitch_deg: float) -> np.ndarray:
    """Camera z-depth of an empty floor `cam_h` below a camera pitched down by `pitch_deg`."""
    f = 0.5 * w / math.tan(math.radians(hfov_deg) / 2)
    u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    ray = np.stack([u / f, v / f, np.ones_like(u)], -1)  # camera frame: x right, y down, z ahead
    p = math.radians(pitch_deg)
    down = ray[..., 1] * math.cos(p) + ray[..., 2] * math.sin(p)  # world-down part of each ray
    t = np.where(down > 1e-3, cam_h / np.maximum(down, 1e-3), np.nan)  # z = 1 rays: t is depth
    return np.where(t < 30, t, np.nan).astype(np.float32)


@pytest.mark.parametrize("w,h", [(518, 294), (294, 518)])
def test_an_empty_floor_reads_clear(w, h):
    depth = floor_depth(w, h, 40.0, 1.2, 35.0)
    scene, plane = analyse(depth, 1.0, 1.0, 40.0, 1.2, None)
    assert scene.floor and plane is not None
    assert math.isinf(scene.path_m)
    assert all(math.isinf(d) for d in scene.sectors)
