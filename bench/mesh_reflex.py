"""The reflex with and without the room model, on the same frames.

For fused walk frames (camera known in the mesh, cams.jsonl), three depth sources go through
the same safety.analyse the live reflex uses:
  net    the live depth network at the live scale (what runs today)
  net+m  the same network output, rescaled per frame to the room model's depth
  mesh   the room model's own depth from that camera (a ray cast, no network)
The mesh is the reference. Reported: depth error, the path distance error that drives the cue,
false cues (says near, room says clear) and misses (room says near, says clear), and the time
each source takes per frame.

    python bench/mesh_reflex.py --mesh $S/denseall --tag s8 --n 300
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from room_mesh import H, W  # noqa: E402

from metasight.config import Settings  # noqa: E402
from metasight.safety import SafetyLoop, analyse  # noqa: E402

# Fixed scales the app could ship with (no room model at run time).
FIXED = {"net@0.40": 0.40, "net@0.45": 0.45, "net@0.50": 0.50, "net@0.55": 0.55}
NEAR_M, CLEAR_M = 1.0, 1.5  # a cue fires under NEAR_M; a reading over CLEAR_M is "clear"


class MeshDepth:
    def __init__(self, mesh):
        import open3d as o3d

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.o3c = o3d.core

    def __call__(self, pose: np.ndarray, k: np.ndarray) -> np.ndarray:
        v, u = np.mgrid[0:H, 0:W]
        d = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones((H, W))], -1)
        dw = d.reshape(-1, 3) @ pose[:3, :3].T  # unnormalised: t_hit is the z depth
        rays = np.concatenate([np.broadcast_to(pose[:3, 3], dw.shape), dw], 1).astype(np.float32)
        t = self.scene.cast_rays(self.o3c.Tensor(rays))["t_hit"].numpy().reshape(H, W)
        return np.where(np.isfinite(t), t, 0.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    import open3d as o3d

    mdir = Path(a.mesh)
    xf = np.array(json.loads((mdir / f"frame-{a.tag}.json").read_text()))
    rows = [json.loads(ln) for ln in (mdir / "cams.jsonl").read_text().splitlines() if ln]
    rows = [r for r in rows if "k" in r]
    by_walk: dict[str, list[dict]] = {}
    for r in rows:
        by_walk.setdefault(r["walk"], []).append(r)
    per = max(1, a.n // len(by_walk))  # every walk gets its share, so one long walk cannot decide
    rows = [
        w[i]
        for w in by_walk.values()
        for i in np.linspace(0, len(w) - 1, min(per, len(w))).astype(int)
    ]
    md = MeshDepth(o3d.io.read_triangle_mesh(str(mdir / f"mesh-{a.tag}.ply")))
    s = Settings.load()
    loop = SafetyLoop(s)
    eye = s.eye_height_m
    res: dict[str, dict[str, list]] = {
        k: {"absrel": [], "path_err": [], "ms": []} for k in ("net", "net+m", "mesh", *FIXED)
    }
    cues = {k: {"false": 0, "miss": 0, "near_ref": 0} for k in ("net", "net+m", *FIXED)}
    used = 0
    scales: dict[str, list[float]] = {}
    for r in rows:
        bgr = cv2.imread(str(ROOT / "data/walks" / r["walk"] / "frames" / r["file"]))
        pose, k = xf @ np.array(r["pose"]), np.array(r["k"])
        hfov = math.degrees(2 * math.atan(W / 2 / k[0, 0]))
        t0 = time.perf_counter()
        ref = md(pose, k)
        res["mesh"]["ms"].append(1000 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        raw = loop.estimate_depth(bgr)
        res["net"]["ms"].append(1000 * (time.perf_counter() - t0))
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
        ok = (ref > 0.2) & (ref < 6.0) & (raw > 0.05)
        if ok.mean() < 0.5:
            continue
        used += 1
        m_scale = float(np.median(ref[ok] / raw[ok]))
        scales.setdefault(r["walk"], []).append(m_scale)
        cam_h = float(pose[1, 3])  # the room knows how high the camera is
        scene_ref, _ = analyse(ref, 1.0, 1.0, hfov, cam_h, None)
        variants = [("net", s.depth_scale, eye), ("net+m", m_scale, cam_h)]
        variants += [(name, v, eye) for name, v in FIXED.items()]
        for name, scale, h in variants:
            est = raw * scale
            res[name]["absrel"].append(float(np.median(np.abs(est[ok] - ref[ok]) / ref[ok])))
            scene, _ = analyse(raw, scale, scale, hfov, h, None)
            p, q = scene.path_m, scene_ref.path_m
            if math.isfinite(p) and math.isfinite(q):
                res[name]["path_err"].append(abs(p - q))
            c = cues[name]
            c["near_ref"] += q < NEAR_M
            c["false"] += p < NEAR_M and not q < CLEAR_M
            c["miss"] += q < NEAR_M and not p < CLEAR_M
    out = {"frames": used, "mesh": str(mdir / f"mesh-{a.tag}.ply")}
    for name, v in res.items():
        out[name] = {key: round(float(np.median(x)), 3) for key, x in v.items() if x} | (
            {"cues": cues[name]} if name in cues else {}
        )
    out["room_scale_by_walk"] = {
        w: [round(float(q), 2) for q in np.quantile(v, [0.1, 0.5, 0.9])] for w, v in scales.items()
    }
    print(json.dumps(out, indent=1))
    (ROOT / "bench/results/mesh-reflex.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
