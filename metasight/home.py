"""The home map on disk: pins, a thinned bank of views and a coarse occupancy grid, all in the
gravity-aligned frame of the last global solve (floor at y = 0). It survives walks. A new walk
puts a few of the saved views into its first global solve, so the same pass that places the walk
also places the house, and the saved pins and grid are carried into the walk's frame by the
transform fitted on those views."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ROOT

OCC_M = 0.10  # occupancy grid cell
HOME_VIEWS = 8  # saved views a solve may carry in with the walk's own bank
SAVED_VIEWS = 36  # views kept on disk: a dozen a room for a three-room floor


def home_dir() -> Path:
    return Path(os.environ.get("METASIGHT_HOME") or ROOT / "data" / "home")


@dataclass
class HomeView:
    ts: float
    jpg: bytes
    pts: np.ndarray  # (h, w, 3) float16, home frame, every STRIDE-th pixel
    mask: np.ndarray  # (h, w) bool
    pose: np.ndarray  # 4x4, home frame


@dataclass
class Home:
    pins: list[dict]
    views: list[HomeView]
    occ: np.ndarray  # (n, 3) int32 cell keys of the occupied space
    walks: int = 0
    updated: float = 0.0
    fixed: dict | None = None  # built from the room mesh: walks never replace its views
    changes: list[dict] = field(default_factory=list)  # room_watch's move flags, mesh frame


def cells(pts: np.ndarray) -> np.ndarray:
    return np.unique(np.floor(pts / OCC_M).astype(np.int32), axis=0)


def cell_centres(keys: np.ndarray) -> np.ndarray:
    return (keys.astype(np.float32) + 0.5) * OCC_M


def load(path: Path | None = None) -> Home | None:
    d = path or home_dir()
    meta = d / "home.json"
    if not meta.is_file():
        return None
    m = json.loads(meta.read_text())
    views: list[HomeView] = []
    vz = d / "views.npz"
    if vz.is_file():
        z = np.load(vz)
        blob, offs = z["jpg"].tobytes(), z["offs"]
        for i in range(len(z["ts"])):
            views.append(
                HomeView(
                    float(z["ts"][i]),
                    blob[offs[i] : offs[i + 1]],
                    z[f"pts{i}"],
                    z[f"mask{i}"],
                    z[f"pose{i}"],
                )
            )
    occ = np.load(d / "occ.npy") if (d / "occ.npy").is_file() else np.empty((0, 3), np.int32)
    return Home(
        m["pins"],
        views,
        occ,
        int(m.get("walks", 0)),
        float(m.get("updated", 0.0)),
        m.get("fixed"),
        m.get("changes", []),
    )


def save(home: Home, path: Path | None = None) -> None:
    """Whole files written beside the old ones and renamed over them, so a crash mid-write
    leaves the last good map."""
    d = path or home_dir()
    d.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "ts": np.array([v.ts for v in home.views], np.float64),
        "jpg": np.frombuffer(b"".join(v.jpg for v in home.views), np.uint8),
        "offs": np.cumsum([0] + [len(v.jpg) for v in home.views]).astype(np.int64),
    }
    for i, v in enumerate(home.views):
        arrays[f"pts{i}"], arrays[f"mask{i}"], arrays[f"pose{i}"] = v.pts, v.mask, v.pose
    tmp = d / "views.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, d / "views.npz")
    with open(d / "occ.tmp.npy", "wb") as f:
        np.save(f, home.occ.astype(np.int32))
    os.replace(d / "occ.tmp.npy", d / "occ.npy")
    meta = {
        "pins": home.pins,
        "walks": home.walks,
        "updated": home.updated,
        "fixed": home.fixed,
        "changes": home.changes,
    }
    (d / "home.tmp.json").write_text(json.dumps(meta))
    os.replace(d / "home.tmp.json", d / "home.json")


def carry(views: list[HomeView], tf) -> list[HomeView]:
    """Saved views moved into a new frame by tf = (scale, rotation, translation), the fit that
    placed a walk in the house. A scan of the hallway solves only the views that look like the
    hallway; the bedroom's views come along through tf, so the next walk that starts in the
    bedroom still finds the house."""
    s_, r_, t_ = tf
    out = []
    for v in views:
        pts = (s_ * v.pts.astype(np.float32).reshape(-1, 3) @ r_.T + t_).reshape(v.pts.shape)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = r_ @ v.pose[:3, :3], s_ * r_ @ v.pose[:3, 3] + t_
        out.append(HomeView(v.ts, v.jpg, pts.astype(np.float16), v.mask, pose))
    return out


def thin(poses: list[np.ndarray], keep: int) -> list[int]:
    """Indices of `keep` views spread over the house: greedy farthest point on position plus a
    little heading, so one corner seen from ten angles does not fill the bank."""
    if len(poses) <= keep:
        return list(range(len(poses)))

    def feat(p: np.ndarray) -> np.ndarray:
        return np.concatenate([p[:3, 3], 0.6 * p[:3, 2]])

    f = np.array([feat(p) for p in poses])
    chosen = [0]
    dist = np.linalg.norm(f - f[0], axis=1)
    while len(chosen) < keep:
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(f - f[i], axis=1))
    return sorted(chosen)


def save_pins(pins: list[dict], path: Path | None = None) -> None:
    """Only the pins changed (between two solves): the views and grid stay as they are."""
    d = path or home_dir()
    meta = json.loads((d / "home.json").read_text())
    meta["pins"], meta["updated"] = pins, time.time()
    (d / "home.tmp.json").write_text(json.dumps(meta))
    os.replace(d / "home.tmp.json", d / "home.json")
