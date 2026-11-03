"""Per-object textures for the sim set, sampled from the walk footage.

Each object's oriented box gets six face images. A texel is a point on a box face; it takes its
colour from every chosen frame whose ray to it passes through no other surface first and lands
on mesh inside the box (so a chair's texel shows the chair, never the wall behind it). Frames
are matched for exposure with one gain each (median luminance ratio against the first-pass
median), then the texel is the median over frames. Unseen texels are inpainted from their face,
or take the object's median colour on a face nothing saw. Shading is not removed: this is the
footage's colour under the room's light, exposure-matched, which is what the set wants.

Frames: the object's best mask frames first, then trusted frames ranked by the box's projected
area, at most one per second of a walk. Every fourth frame is held out to score the texture
against a single flat colour (mean abs RGB error on held-out samples, 0 to 255).

Face images: face_<name>.png, viewed from outside with the box's up (+y) at the top for the
four sides; top viewed from above with rows along +z. albedo.json gives each face's world
corners (top-left, top-right, bottom-right, bottom-left), so UVs need no convention guessing.

    PYTHONPATH=. .venv/bin/python scripts/recon_albedo.py --version v2.1 \\
        --cams data/rooms/bedroom/recon-v2/v2.1/cams.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from recon_v2 import ROOM, WALKS, H, W, cast, load_cams, scene_of, walk_ts  # noqa: E402

PX_PER_M = 220
MAX_PX = 384
MIN_PX = 8
N_FRAMES = 32
MAX_DIST = 4.0


def axes(yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ex = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    return ex, np.array([0.0, 1.0, 0.0]), np.array([-np.sin(yaw), 0.0, np.cos(yaw)])


def faces(b: dict) -> list[dict]:
    """Six faces: name, outward normal, centre, right and down axes (unit), extent along each."""
    c, s = np.asarray(b["center"]), np.asarray(b["size"])
    ex, ey, ez = axes(b["yaw_rad"])
    out = []

    def face(name, n, c, r, d, w, h):
        out.append({"name": name, "n": n, "c": c, "r": r, "d": d, "w": w, "h": h})

    for name, n, half in [("+x", ex, s[0]), ("-x", -ex, s[0]), ("+z", ez, s[2]), ("-z", -ez, s[2])]:
        w = s[2] if name[1] == "x" else s[0]
        face(name, n, c + n * half / 2, np.cross(-n, ey), -ey, w, s[1])
    face("+y", ey, c + ey * s[1] / 2, ex, ez, s[0], s[2])
    face("-y", -ey, c - ey * s[1] / 2, ex, -ez, s[0], s[2])
    return out


def texels(f: dict) -> tuple[np.ndarray, int, int]:
    nu = int(np.clip(round(f["w"] * PX_PER_M), MIN_PX, MAX_PX))
    nv = int(np.clip(round(f["h"] * PX_PER_M), MIN_PX, MAX_PX))
    u = ((np.arange(nu) + 0.5) / nu - 0.5) * f["w"]
    v = ((np.arange(nv) + 0.5) / nv - 0.5) * f["h"]
    uu, vv = np.meshgrid(u, v)
    p = f["c"] + uu[..., None] * f["r"] + vv[..., None] * f["d"]
    return p.reshape(-1, 3), nu, nv


def inside(pts: np.ndarray, b: dict, grow: float = 0.05) -> np.ndarray:
    c, s = np.asarray(b["center"]), np.asarray(b["size"]) / 2 + grow
    ex, _, ez = axes(b["yaw_rad"])
    d = pts - c
    return (np.abs(d @ ex) <= s[0]) & (np.abs(d[:, 1]) <= s[1]) & (np.abs(d @ ez) <= s[2])


def project(cam: dict, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates in the full frame, and depth along the optical axis."""
    pose = cam["pose"]
    pc = (pts - pose[:3, 3]) @ pose[:3, :3]
    z = pc[:, 2]
    uv = pc[:, :2] / np.maximum(z, 1e-6)[:, None] * cam["k"][[0, 1], [0, 1]] + cam["k"][:2, 2]
    return uv, z


def choose(o: dict, cams: dict, lookup: dict) -> list[dict]:
    b = o["obb"]
    ex, ey, ez = axes(b["yaw_rad"])
    s = np.asarray(b["size"]) / 2
    corners = np.array(
        [
            b["center"] + i * s[0] * ex + j * s[1] * ey + k * s[2] * ez
            for i in (-1, 1)
            for j in (-1, 1)
            for k in (-1, 1)
        ]
    )
    picked, used = [], set()
    for key in o["seen"]["best_frames"]:
        if key in lookup and len(picked) < N_FRAMES // 3:
            picked.append(lookup[key])
            used.add(key)
    scored = []
    for w, rows in cams.items():
        for r in rows[::3]:
            if not r["trusted"]:
                continue
            uv, z = project(r, corners)
            if (z < 0.2).any() or np.linalg.norm(r["pose"][:3, 3] - b["center"]) > MAX_DIST:
                continue
            lo, hi = np.clip(uv.min(0), 0, [W, H]), np.clip(uv.max(0), 0, [W, H])
            area = float(np.prod(hi - lo))
            if area > 400:
                scored.append((area, w, r))
    last: dict[str, list[int]] = {}
    for key in used:
        w, f = key.split("/")
        last.setdefault(w, []).append(walk_ts(w, f))
    for _a, w, r in sorted(scored, key=lambda t: -t[0]):
        if len(picked) >= N_FRAMES:
            break
        t = walk_ts(w, r["file"])
        if any(abs(t - u) < 1000 for u in last.get(w, [])):
            continue
        last.setdefault(w, []).append(t)
        picked.append(r)
    return picked


def surface(pts: np.ndarray, n: np.ndarray, b: dict, scene) -> np.ndarray:
    """The mesh point behind each texel: a ray from just outside the face straight in, its first
    hit kept when it is inside the box. NaN rows where the ray finds nothing of the object."""
    org = pts + n * 0.05
    t, _ = cast(scene, org, np.repeat(-n[None], len(pts), 0))
    hit = org - n * np.where(np.isfinite(t), t, 0)[:, None]
    ok = np.isfinite(t) & inside(hit, b, grow=0.02)
    return np.where(ok[:, None], hit, np.nan)


def sample(cam: dict, img: np.ndarray, surf: np.ndarray, n: np.ndarray, scene) -> np.ndarray:
    """RGB per surface point from this frame, NaN where facing away, off frame or occluded."""
    out = np.full((len(surf), 3), np.nan, np.float32)
    have = np.flatnonzero(np.isfinite(surf[:, 0]))
    pts = surf[have]
    org = cam["pose"][:3, 3]
    to = pts - org
    dist = np.linalg.norm(to, axis=1)
    dirs = to / dist[:, None]
    facing = (-dirs @ n) > 0.2
    uv, z = project(cam, pts)
    sx, sy = img.shape[1] / W, img.shape[0] / H
    px = uv * [sx, sy]
    ok = facing & (z > 0.1) & (px[:, 0] >= 0) & (px[:, 0] < img.shape[1] - 1) & (px[:, 1] >= 0)
    ok &= px[:, 1] < img.shape[0] - 1
    if not ok.any():
        return out
    t, _ = cast(scene, np.repeat(org[None], ok.sum(), 0), dirs[ok])
    good = np.isfinite(t) & (t > dist[ok] - 0.02 - 0.01 * dist[ok])
    idx = np.flatnonzero(ok)[good]
    out[have[idx]] = bilinear(img, px[idx])[:, ::-1]
    return out


def bilinear(img: np.ndarray, p: np.ndarray) -> np.ndarray:
    x0, y0 = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
    fx, fy = (p[:, 0] - x0)[:, None], (p[:, 1] - y0)[:, None]
    im = img.astype(np.float32)
    top = im[y0, x0] * (1 - fx) + im[y0, x0 + 1] * fx
    bot = im[y0 + 1, x0] * (1 - fx) + im[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def lum(x: np.ndarray) -> np.ndarray:
    return x @ np.array([0.299, 0.587, 0.114], np.float32)


def fuse(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frames, texels, 3) with NaN gaps -> per-frame gains and the gain-matched median."""
    with np.errstate(all="ignore"):
        first = np.nanmedian(stack, 0)
        gains = np.ones(len(stack), np.float32)
        for i, s in enumerate(stack):
            ok = np.isfinite(s[:, 0]) & np.isfinite(first[:, 0]) & (lum(s) > 8)
            if ok.sum() >= 20:
                gains[i] = np.clip(np.median(lum(first[ok]) / lum(s[ok])), 0.5, 2.0)
        return gains, np.nanmedian(stack * gains[:, None, None], 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--cams", help="recon_reloc.py cams file (default: the dense run's)")
    ap.add_argument("--only", nargs="*", help="object ids to do (default: all)")
    a = ap.parse_args()
    import open3d as o3d

    vdir = ROOM / "recon-v2" / a.version
    doc = json.loads((vdir / "objects.json").read_text())
    mesh_path = Path(doc["source"]["mesh"])
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = scene_of(mesh)
    cams = load_cams(
        mesh_path.parent, mesh_path.stem.split("-")[-1], Path(a.cams) if a.cams else None
    )
    lookup = {f"{w}/{r['file']}": r for w, rows in cams.items() for r in rows}
    out_dir = vdir / "albedo"
    out_dir.mkdir(exist_ok=True)
    cache: dict[str, np.ndarray] = {}
    report = []
    for o in doc["objects"]:
        if a.only and o["id"] not in a.only:
            continue
        frames = choose(o, cams, lookup)
        imgs = []
        for r in frames:
            key = f"{r['walk']}/{r['file']}"
            if key not in cache:
                cache[key] = cv2.imread(str(WALKS / r["walk"] / "frames" / r["file"]))
            imgs.append(cache[key])
        held = np.arange(len(frames)) % 4 == 3
        odir = out_dir / o["id"]
        odir.mkdir(exist_ok=True)
        rec = {"id": o["id"], "class": o["class"], "frames": len(frames), "faces": {}}
        err_tex, err_flat, all_med = [], [], []
        face_imgs = {}
        for f in faces(o["obb"]):
            pts, nu, nv = texels(f)
            surf = surface(pts, f["n"], o["obb"], scene)
            stack = np.stack(
                [sample(r, im, surf, f["n"], scene) for r, im in zip(frames, imgs, strict=True)]
            )
            _, tex = fuse(stack)
            # Held-out check: fuse without the held-out frames, then compare on their samples.
            if held.any() and (~held).sum() >= 2:
                _, train = fuse(stack[~held])
                with np.errstate(all="ignore"):
                    flat = np.nanmedian(train, 0)
                for s in stack[held]:
                    ok = np.isfinite(s[:, 0]) & np.isfinite(train[:, 0])
                    if ok.sum() < 20:
                        continue
                    g = np.clip(np.median(lum(train[ok]) / np.maximum(lum(s[ok]), 1)), 0.5, 2.0)
                    err_tex.append(np.abs(s[ok] * g - train[ok]).mean(1))
                    err_flat.append(np.abs(s[ok] * g - flat).mean(1))
            seen = np.isfinite(tex[:, 0])
            face_imgs[f["name"]] = (tex, seen, nu, nv, f)
            if seen.any():
                all_med.append(tex[seen])
        fill = (
            np.median(np.concatenate(all_med), 0)
            if all_med
            else np.array([128, 128, 128], np.float32)
        )
        for name, (tex, seen, nu, nv, f) in face_imgs.items():
            img = np.where(seen[:, None], tex, fill).reshape(nv, nu, 3)
            img = np.clip(img, 0, 255).astype(np.uint8)[..., ::-1]
            hole = (~seen).reshape(nv, nu).astype(np.uint8)
            if 0 < hole.mean() < 0.9:
                img = cv2.inpaint(img, hole, 5, cv2.INPAINT_TELEA)
            cv2.imwrite(str(odir / f"face_{name}.png"), img)
            hw, hh = f["r"] * f["w"] / 2, f["d"] * f["h"] / 2
            corners = [f["c"] - hw - hh, f["c"] + hw - hh, f["c"] + hw + hh, f["c"] - hw + hh]
            rec["faces"][name] = {
                "file": f"{o['id']}/face_{name}.png",
                "px": [nu, nv],
                "coverage": round(float(seen.mean()), 3),
                "corners_tl_tr_br_bl": np.round(corners, 4).tolist(),
            }
        rec["mean_rgb"] = np.round(fill).astype(int).tolist()
        if err_tex:
            et, ef = np.concatenate(err_tex), np.concatenate(err_flat)
            rec["heldout"] = {"samples": int(len(et)), "mae_texture": round(float(et.mean()), 1)}
            rec["heldout"]["mae_flat"] = round(float(ef.mean()), 1)
        report.append(rec)
        print(
            o["id"],
            len(frames),
            rec.get("heldout"),
            {k: v["coverage"] for k, v in rec["faces"].items()},
            flush=True,
        )
    ok = [r for r in report if "heldout" in r]
    n = sum(r["heldout"]["samples"] for r in ok)
    summary = {
        "objects": len(report),
        "scored": len(ok),
        "mae_texture": round(
            sum(r["heldout"]["mae_texture"] * r["heldout"]["samples"] for r in ok) / max(n, 1), 1
        ),
        "mae_flat": round(
            sum(r["heldout"]["mae_flat"] * r["heldout"]["samples"] for r in ok) / max(n, 1), 1
        ),
    }
    out = {
        "version": a.version,
        "frame": doc["frame"],
        "note": "footage colour, exposure-matched per frame; room shading is not removed",
        "summary": summary,
        "objects": report,
    }
    (out_dir / "albedo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
