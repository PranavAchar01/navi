"""Slow orbit video of the reconstructed room with labelled object boxes (ceiling cut away).

PYTHONPATH=. .venv/bin/python scripts/recon_orbit.py [--seconds 11] [--turn 180] [--still-only]
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "data/rooms/bedroom/recon-v2/v2.0"
MESH = ROOT / "data/rooms/bedroom/v002/mesh-s1.ply"
W, H, FPS = 1920, 1080, 30
FOV_V = math.radians(45)
ELEV = math.radians(42)
CUT_M = 2.2
LABELLED = {"bed", "desk", "door", "whiteboard", "chair", "nightstand"}
HAZARD_BGR = (40, 110, 240)
COOL_BGR = (200, 140, 40)
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def box_corners(obb: dict) -> np.ndarray:
    """Eight corners: the four top corners, then the four at the bottom."""
    top = np.array(obb["corners_top"], dtype=float)
    bot = top.copy()
    bot[:, 1] = obb["center"][1] - obb["size"][1] / 2
    return np.vstack([top, bot])


class Cam:
    def __init__(self, target: np.ndarray, dist: float, az: float):
        self.f = (H / 2) / math.tan(FOV_V / 2)
        self.eye = target + dist * np.array(
            [math.cos(ELEV) * math.cos(az), math.sin(ELEV), math.cos(ELEV) * math.sin(az)]
        )
        fwd = target - self.eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        # OpenGL-free pinhole: x right, y down, z forward
        self.R = np.stack([right, -up, fwd])

    def extrinsic(self) -> np.ndarray:
        e = np.eye(4)
        e[:3, :3] = self.R
        e[:3, 3] = -self.R @ self.eye
        return e

    def project(self, pts: np.ndarray) -> np.ndarray:
        c = (pts - self.eye) @ self.R.T
        return np.stack([self.f * c[:, 0] / c[:, 2] + W / 2, self.f * c[:, 1] / c[:, 2] + H / 2], 1)


def fit_dist(verts: np.ndarray, target: np.ndarray) -> float:
    """Smallest distance keeping the core 96% of vertices in frame at any azimuth."""
    sub = verts[:: max(1, len(verts) // 40000)]
    lo, hi = np.percentile(sub, [2, 98], axis=0)
    sub = sub[((sub >= lo) & (sub <= hi)).all(1)]
    d = 2.0
    while d < 40:
        ok = True
        for az in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            p = Cam(target, d, az).project(sub)
            if (np.abs(p[:, 0] - W / 2) > 0.46 * W).any() or (
                np.abs(p[:, 1] - H / 2) > 0.46 * H
            ).any():
                ok = False
                break
        if ok:
            return d
        d *= 1.04
    return d


def annotate(img: np.ndarray, cam: Cam, objs: list[dict]) -> None:
    for o in objs:
        c = box_corners(o["obb"])
        p = cam.project(c)
        col = HAZARD_BGR if o["hazard"] else COOL_BGR
        for a, b in EDGES:
            pa, pb = tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int))
            cv2.line(img, pa, pb, col, 2, cv2.LINE_AA)
    for o in objs:
        if o["label"] not in LABELLED:
            continue
        c = box_corners(o["obb"])
        top = c[:4].mean(0)
        x, y = cam.project(top[None])[0]
        text = o["label"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        org = (int(x - tw / 2), int(y - 8))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 5, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    import json

    import open3d as o3d
    from open3d.visualization import rendering

    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--turn", type=float, default=180.0, help="degrees of orbit")
    ap.add_argument("--start", type=float, default=200.0, help="start azimuth, degrees")
    ap.add_argument("--still-only", action="store_true")
    args = ap.parse_args()

    doc = json.loads((OUT / "objects.json").read_text())
    mesh = o3d.io.read_triangle_mesh(str(MESH))
    v = np.asarray(mesh.vertices)
    mesh.remove_vertices_by_mask(v[:, 1] > doc["floor_y"] + CUT_M)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    lo, hi = np.percentile(v[:: max(1, len(v) // 40000)], [2, 98], axis=0)
    target = np.array([(lo[0] + hi[0]) / 2, doc["floor_y"] + 0.8, (lo[2] + hi[2]) / 2])
    dist = fit_dist(v, target)

    r = rendering.OffscreenRenderer(W, H)
    r.scene.set_background([0.96, 0.96, 0.96, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    r.scene.add_geometry("room", mesh, mat)
    k = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1.0]])
    f = (H / 2) / math.tan(FOV_V / 2)
    k = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1.0]])
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, k)

    def frame(az: float) -> np.ndarray:
        cam = Cam(target, dist, az)
        r.setup_camera(intr, cam.extrinsic())
        img = cv2.cvtColor(np.asarray(r.render_to_image()), cv2.COLOR_RGB2BGR)
        annotate(img, cam, doc["objects"])
        return img

    n = int(args.seconds * FPS)
    a0 = math.radians(args.start)
    span = math.radians(args.turn)
    # ease in and out so the orbit starts and ends gently
    az_at = lambda i: a0 + span * (0.5 - 0.5 * math.cos(math.pi * i / (n - 1)))  # noqa: E731
    cv2.imwrite(str(OUT / "orbit.png"), frame(az_at(n // 3)))
    if args.still_only:
        return
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / "orbit.mp4")],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for i in range(n):
        ff.stdin.write(frame(az_at(i)).tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    from room_watch import take_slot

    slot = take_slot()
    try:
        main()
    finally:
        subprocess.run(["rm", "-r", str(slot)], check=False)
