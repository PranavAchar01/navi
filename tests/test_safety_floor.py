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


def bent_floor(w: int, h: int) -> np.ndarray:
    """The empty floor as the depth net reads it on the phone walk: the last 30% of rows, just
    ahead of the feet, come out up to 25% short, so that floor rises 20 to 30 cm off the plane."""
    depth = floor_depth(w, h, 40.0, 1.2, 35.0)
    ramp = np.clip((np.arange(h) - 0.7 * h) / (0.3 * h), 0, 1)[:, None]
    return (depth * (1 - 0.25 * ramp)).astype(np.float32)


def test_the_bent_floor_ahead_of_the_feet_is_no_obstacle():
    scene, plane = analyse(bent_floor(294, 518), 1.0, 1.0, 40.0, 1.2, None)
    assert scene.floor and plane is not None
    assert scene.path_m >= 2.0  # no cue (the plane alone read 0.42 m)


def test_a_box_on_the_bent_floor_still_reads():
    """An upright face stops the floor trace: a 40 cm box 1 m ahead is still in the path."""
    w, h, cam_h, pitch = 294, 518, 1.2, math.radians(35.0)
    depth = bent_floor(w, h)
    f = 0.5 * w / math.tan(math.radians(40.0) / 2)
    u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    ahead = -(v / f) * math.sin(pitch) + math.cos(pitch)  # world-forward part of each z = 1 ray
    down = (v / f) * math.cos(pitch) + math.sin(pitch)
    t = 1.0 / np.maximum(ahead, 1e-3)  # depth where the ray reaches the box face 1 m ahead
    face = (
        (ahead > 1e-3)
        & (cam_h - t * down < 0.4)
        & (cam_h - t * down > 0)
        & (np.abs(u / f * t) < 0.3)
    )
    depth = np.where(face & (t < depth), t, depth).astype(np.float32)
    scene, _ = analyse(depth, 1.0, 1.0, 40.0, 1.2, None)
    assert 0.8 < scene.path_m < 1.2


def test_with_no_floor_an_outline_is_not_cut_flat_at_the_top_cut():
    """No floor in view: the cue ignores the top 30% of rows, but the outline drawn from it ran a
    straight line across the frame there (replay 131247: 115 of 431 outlines)."""
    from metasight.safety import hazards

    w, h = 128, 96
    d = np.full((h, w), 6.0, np.float32)  # far wall, 2.7 m at scale 0.45
    d[5:, 40:88] = 2.0  # a near door-height panel from the top of the view to the bottom
    sc, plane = analyse(d, 0.45, 0.45, 60.0, 1.22, None, None)
    assert plane is None and sc.path_m < 1.0  # the cue is unchanged
    (hz,) = hazards(sc.grid, sc.corridor, keep=1)
    ys = np.array(hz["poly"]).reshape(-1, 2)[:, 1]
    assert ys.min() < 0.1  # the outline reaches the panel's top, not the row cut at 0.3
