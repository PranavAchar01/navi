"""Pure geometry: sector distances, cue levels, and where a point is relative to you.

Map frame: x right, y up, z back (three.js convention), metres. A camera pose is a 4x4
camera-to-map matrix in the OpenCV camera convention (x right, y down, z forward), so the
camera's forward axis is pose[:3, 2] and its up axis is -pose[:3, 1].
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

# Eight belt sectors, clockwise from straight ahead, 45 degrees each.
SECTORS = [
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
]
CAMERA_SECTORS = ["front-left", "front", "front-right"]

# Cue thresholds from the plan: off beyond 2 m, slow pulse 2 to 1 m, fast 1 to 0.5 m, solid below.
OFF_M, SLOW_M, FAST_M = 2.0, 1.0, 0.5


def sector_distances_from_depth(
    depth: np.ndarray, n: int = 3, top_skip: float = 0.3, percentile: float = 5.0
) -> list[float]:
    """Nearest obstacle per vertical strip of a metric depth map.

    The 5th percentile (not the median) so one thin chair leg counts. The top `top_skip` of the
    image is ignored: ceilings and the far end of a corridor are not obstacles.
    """
    band = depth[int(top_skip * depth.shape[0]) :]
    out = []
    for cols in np.array_split(np.arange(band.shape[1]), n):
        strip = band[:, cols]
        strip = strip[np.isfinite(strip) & (strip > 0.05)]
        out.append(float(np.percentile(strip, percentile)) if strip.size else math.inf)
    return out


def grid_strips(grid: np.ndarray, n: int = 15) -> list[float]:
    """Nearest obstacle per vertical strip of a `Scene.grid` (uint8 range in cm / 2, 255 where
    nothing is), in metres; inf for a strip with no obstacle pixel. The panned cues' input."""
    out = []
    for cols in np.array_split(np.arange(grid.shape[1]), n):
        g = grid[:, cols]
        near = g[g != 255]
        out.append(float(near.min()) * 2 / 100 if near.size else math.inf)
    return out


def sector_index(bearing_deg: float, n: int = 8) -> int:
    """Which of n sectors a bearing falls in (0 = ahead, positive = clockwise/right)."""
    width = 360.0 / n
    return int(((bearing_deg + width / 2) % 360.0) // width)


def local_floor(points: np.ndarray, cell: float = 0.3, reach: int = 2) -> np.ndarray:
    """Floor height under each point: the median, over its 30 cm cell and the cells within
    `reach`, of each cell's lowest surface (5th percentile). The map's floor is not flat (it
    tilts with drift), so a single floor level counts the floor itself as an obstacle on one
    side; a single lowest level also catches the noise below the floor. The median over the
    neighbours keeps a table top, whose own cell may show no floor, from becoming the floor.
    """
    ij = np.floor(points[:, [0, 2]] / cell).astype(int)
    ij -= ij.min(axis=0)
    shape = ij.max(axis=0) + 1
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    order = np.lexsort((points[:, 1], flat))
    cells, first, counts = np.unique(flat[order], return_index=True, return_counts=True)
    low = np.full(shape[0] * shape[1], np.nan)
    low[cells] = points[order[first + (counts * 0.05).astype(int)], 1]
    grid = low.reshape(shape)
    pad = np.pad(grid, reach, constant_values=np.nan)
    span = 2 * reach + 1
    with warnings.catch_warnings():  # empty cells far from any point; never read below
        warnings.simplefilter("ignore", RuntimeWarning)
        neigh = np.nanmedian(
            [pad[a : a + shape[0], b : b + shape[1]] for a in range(span) for b in range(span)],
            axis=0,
        )
    return neigh[ij[:, 0], ij[:, 1]]


def map_sector_distances(
    points: np.ndarray,
    pose: np.ndarray,
    max_range: float = 3.0,
    min_h: float = 0.25,
    max_h: float = 1.8,
    floor_y: float | None = None,
    percentile: float = 5.0,
    n: int = 8,
) -> list[float]:
    """Nearest map surface per belt sector around the camera, from the 3D map.

    Only points between `min_h` and `max_h` above the floor count (floor and ceiling are not
    obstacles). The floor is `floor_y` when given, otherwise measured locally under each point.
    This is what drives the side and rear sectors the camera cannot see.
    """
    pos = pose[:3, 3]
    fwd = pose[:3, 2].copy()
    fwd[1] = 0.0
    norm = np.linalg.norm(fwd)
    fwd = fwd / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0])
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    rel = points - pos
    h = points[:, 1] - (floor_y if floor_y is not None else local_floor(points))
    d = np.hypot(rel[:, 0], rel[:, 2])
    keep = (h > min_h) & (h < max_h) & (d < max_range) & (d > 0.15)
    out = [math.inf] * n
    if not keep.any():
        return out
    rel, d = rel[keep], d[keep]
    bearing = np.degrees(np.arctan2(rel @ right, rel @ fwd))
    idx = (((bearing + 180.0 / n) % 360.0) // (360.0 / n)).astype(int)
    for s in range(n):
        sel = d[idx == s]
        if sel.size >= 5:
            out[s] = float(np.percentile(sel, percentile))
    return out


@dataclass(frozen=True)
class Cue:
    mode: str  # off | slow | fast | solid
    intensity: float  # 0..1
    rate_hz: float  # pulses per second, 0 when off or solid


def cue_for_distance(d: float) -> Cue:
    """Distance to haptic/audio cue, per the plan's encoding."""
    if not math.isfinite(d) or d >= OFF_M:
        return Cue("off", 0.0, 0.0)
    if d >= SLOW_M:
        return Cue("slow", 0.35 + 0.25 * (OFF_M - d), 1.5)
    if d >= FAST_M:
        return Cue("fast", 0.6 + 0.3 * (SLOW_M - d) / (SLOW_M - FAST_M), 4.0)
    return Cue("solid", 1.0, 0.0)


def merge_sectors(camera: list[float], from_map: list[float]) -> list[float]:
    """Eight-sector vector: live camera depth for the three front sectors, the map for the rest."""
    merged = list(from_map)
    for name, dist in zip(CAMERA_SECTORS, camera, strict=True):
        merged[SECTORS.index(name)] = dist
    return merged


@dataclass(frozen=True)
class Relative:
    horizontal_m: float
    bearing_deg: float  # 0 ahead, +90 right, +-180 behind
    dy_m: float  # positive = above the camera


def relative_to(point: np.ndarray, pose: np.ndarray) -> Relative:
    pos = pose[:3, 3]
    fwd = pose[:3, 2].copy()
    fwd[1] = 0.0
    norm = np.linalg.norm(fwd)
    fwd = fwd / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0])
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    v = point - pos
    horiz = float(math.hypot(v[0], v[2]))
    bearing = math.degrees(math.atan2(float(v @ right), float(v @ fwd)))
    return Relative(horiz, bearing, float(v[1]))


def gravity_alignment(accel_cam: np.ndarray) -> np.ndarray:
    """Rotation taking the first camera frame (OpenCV, y down) into the y-up map frame.

    `accel_cam` is the accelerometer's gravity axis expressed in OpenCV camera axes, sign
    unknown (iOS and Android disagree). The sign is chosen so the camera's own up (-y) points
    upward, which holds for a phone or glasses held in a normal pose.
    """
    g = np.asarray(accel_cam, dtype=float)
    norm = np.linalg.norm(g)
    if norm < 1e-6:
        up = np.array([0.0, -1.0, 0.0])
    else:
        up = g / norm
        if up @ np.array([0.0, -1.0, 0.0]) < 0:
            up = -up
    # Build R with R @ up = +y; keep the camera's heading: x stays as close to x as possible.
    y_axis = up
    x_axis = np.array([1.0, 0.0, 0.0])
    x_axis = x_axis - (x_axis @ y_axis) * y_axis
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.stack([x_axis, y_axis, z_axis])  # rows: map axes expressed in camera axes
