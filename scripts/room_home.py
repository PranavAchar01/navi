"""Turn the room model into the app's home map, so live walks land in the mesh's coordinates.

The home map's views are what a walk's global solve relocalizes against (memory._relocalize).
Here they are the room mesh's anchor views, moved into the mesh frame (floor at y = 0, walls
squared), and the occupancy grid is the mesh itself. The map is marked fixed: a walk refreshes
its pins but never replaces these views with its own, so the mesh stays the reference.

    python scripts/room_home.py data/walks/20261031-004732-glasses --anchors $S/room160 \
        --views 160 --mesh $S/denseall --tag s8 --out data/home-room
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from map_to_mesh import anchor_times  # noqa: E402
from room_dense import world_pts  # noqa: E402

from metasight import home as homemod  # noqa: E402
from metasight.memory import STRIDE  # noqa: E402

VIEW_TRIS = 400_000  # the browser copy of the mesh: about 15 MB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk", help="the walk the anchors came from")
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--views", type=int, required=True)
    ap.add_argument("--mesh", required=True, help="room_dense.py output dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import open3d as o3d

    z = np.load(Path(a.anchors) / "views.npz")
    xf = np.array(json.loads((Path(a.mesh) / f"frame-{a.tag}.json").read_text()))
    ts = anchor_times(Path(a.walk), a.views)
    views = []
    for i in range(len(z["pose"])):
        pts, ok = world_pts(z["depth"][i], z["k"][i], z["pose"][i], z["mask"][i], STRIDE)
        _, jpg = cv2.imencode(
            ".jpg", cv2.cvtColor(z["rgb"][i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        views.append(
            homemod.HomeView(
                float(ts[i]),
                jpg.tobytes(),
                (pts @ xf[:3, :3].T + xf[:3, 3]).astype(np.float16),
                ok,
                xf @ z["pose"][i].astype(float),
            )
        )
    mesh = o3d.io.read_triangle_mesh(str(Path(a.mesh) / f"mesh-{a.tag}.ply"))
    occ = homemod.cells(np.asarray(mesh.vertices))
    out = Path(a.out)
    old = out / "home.json"  # pins he placed stay: the room frame is fixed, so they still fit
    pins = json.loads(old.read_text())["pins"] if old.is_file() else []
    homemod.save(homemod.Home([], views, occ, 0, time.time()), out)
    meta = json.loads(old.read_text())
    meta["pins"] = pins
    meta["fixed"] = {"mesh": str(Path(a.mesh) / f"mesh-{a.tag}.ply"), "frame": xf.tolist()}
    # The same room thinned for the browser (the judge view lays it over the camera).
    thin = mesh.simplify_quadric_decimation(VIEW_TRIS)
    o3d.io.write_triangle_mesh(str(out / "room.tmp.ply"), thin, write_ascii=False)
    (out / "room.tmp.ply").replace(out / "room.ply")
    (out / "home.json").write_text(json.dumps(meta))
    print(f"{len(views)} views, {len(occ)} occupied cells -> {out}")


if __name__ == "__main__":
    main()
