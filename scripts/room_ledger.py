"""Footage per room against the 10 hour benchmark, and which frames are worth keeping.

Each frame of a walk takes the camera pose of the nearest fused frame (cams.jsonl, mesh frame)
within a second; its room is the outline it stands in (data/rooms/rooms.json). The ledger
counts seconds and frames per room and marks a frame needed while its view bucket is not full
(metasight.rooms). Rebuilt from scratch each run, so it always matches the current mesh.

    python scripts/room_ledger.py --mesh $S/denseall --tag s8 --init-room bedroom
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metasight import rooms as roomsmod  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NEAR_S = 1.0


def floor_outline(mesh) -> np.ndarray:
    """Convex hull of the mesh's floor in x, z."""
    v = np.asarray(mesh.vertices)
    floor = v[v[:, 1] < 0.1][:, [0, 2]].astype(np.float32)
    return cv2.convexHull(floor)[:, 0].astype(float)


def sharpness(path: Path) -> float:
    g = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="room_dense.py output dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init-room", help="add this room with the mesh floor as its outline")
    a = ap.parse_args()
    import open3d as o3d

    mdir = Path(a.mesh)
    xf = np.array(json.loads((mdir / f"frame-{a.tag}.json").read_text()))
    rooms_json = roomsmod.ROOMS_DIR / "rooms.json"
    if a.init_room:
        mesh = o3d.io.read_triangle_mesh(str(mdir / f"mesh-{a.tag}.ply"))
        have = json.loads(rooms_json.read_text())["rooms"] if rooms_json.is_file() else []
        have = [r for r in have if r["name"] != a.init_room]
        have.append({"name": a.init_room, "outline": floor_outline(mesh).round(3).tolist()})
        rooms_json.parent.mkdir(parents=True, exist_ok=True)
        rooms_json.write_text(json.dumps({"rooms": have}, indent=1))
    rooms = roomsmod.load()

    fused: dict[str, dict[str, np.ndarray]] = {}
    for ln in (mdir / "cams.jsonl").read_text().splitlines():
        r = json.loads(ln)
        fused.setdefault(r["walk"], {})[r["file"]] = xf @ np.array(r["pose"])
    ledger = roomsmod.Ledger(walks=sorted(fused))
    needed_out = {}
    for walk, poses in sorted(fused.items()):
        wdir = ROOT / "data" / "walks" / walk
        rows = [json.loads(ln) for ln in (wdir / "frames.jsonl").read_text().splitlines() if ln]
        rows.sort(key=lambda r: r["ts"])
        ts = np.array([r["ts"] / 1000 for r in rows])
        have = np.array([r["file"] in poses for r in rows])
        fused_t = ts[have]
        fused_files = [r["file"] for r, h in zip(rows, have, strict=True) if h]
        sharp = np.array([sharpness(wdir / "frames" / r["file"]) for r in rows])
        cut = np.quantile(sharp, 0.3)
        needed, last = [], None
        for i, r in enumerate(rows):
            j = int(np.abs(fused_t - ts[i]).argmin())
            if abs(fused_t[j] - ts[i]) > NEAR_S:
                last = None
                continue
            pose = poses[fused_files[j]]
            room = roomsmod.room_at(rooms, pose[:3, 3]) or "outside"
            dt = ts[i] - last if last is not None else 0.0
            last = ts[i]
            if ledger.add(room, dt, pose, sharp[i] >= cut):
                needed.append(r["file"])
        needed_out[walk] = needed
        print(f"{walk}: {len(rows)} frames, {len(needed)} needed", flush=True)
    ledger.save(roomsmod.ROOMS_DIR / "ledger.json")
    (roomsmod.ROOMS_DIR / "needed.json").write_text(json.dumps(needed_out))
    for room, rep in ledger.report().items():
        print(
            f"{room}: {rep['hours'] * 60:.1f} min of {roomsmod.TARGET_HOURS:.0f} h"
            f" ({rep['of_target'] * 100:.1f}%), {rep['frames']} frames,"
            f" {rep['view_buckets']} view buckets ({rep['full_buckets']} full)"
        )


if __name__ == "__main__":
    main()
