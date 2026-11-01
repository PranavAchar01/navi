"""Refine a splat dataset's camera poses with COLMAP bundle adjustment, keeping the room frame.

The dataset's poses come from the dense run and drift a little frame to frame, which blurs the
splats. SIFT matches between frames that already look at the same place pin the poses down;
triangulating with the known poses and bundle-adjusting (intrinsics fixed, robust loss) moves
each camera by whatever the pixels disagree with. The result is aligned back onto the original
cameras (Umeyama on centres), so room.ply and the saved objects stay valid.

    python scripts/splat_refine_poses.py data/rooms/bedroom/splat-x1 data/rooms/bedroom/splat-x3
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

GL = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes to nerfstudio's OpenGL ones
MIN_OBS = 30  # a frame with fewer triangulated points than this is left where it was
MAX_ANGLE = 60.0  # degrees between two cameras' viewing directions for a pair to be matched
MAX_REPROJ = 4.0  # px; a point whose mean error is above this after BA is dropped


def copy_dataset(src: Path, dst: Path) -> None:
    if dst.exists():
        raise SystemExit(f"{dst} exists; pass a fresh output directory")
    shutil.copytree(src, dst, symlinks=False)  # Brush reads real files


def load_poses(tf: dict) -> list[np.ndarray]:
    """OpenCV camera-to-world 4x4 per frame."""
    return [np.array(f["transform_matrix"], float) @ GL for f in tf["frames"]]


def write_masks(src: Path, tf: dict, out: Path) -> None:
    """COLMAP wants <image path>.png with black = ignore; ours are <stem>.png, maybe grey."""
    out.mkdir(parents=True, exist_ok=True)
    for f in tf["frames"]:
        if "mask_path" not in f:
            continue
        m = cv2.imread(str(src / f["mask_path"]), cv2.IMREAD_GRAYSCALE)
        dst = out / (f["file_path"] + ".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), np.where(m > 127, 255, 0).astype(np.uint8))


def extract(db: Path, root: Path, tf: dict, masked: bool) -> None:
    names = [f["file_path"] for f in tf["frames"]]
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "PINHOLE"
    if masked:
        ro.mask_path = root / "colmap" / "masks"
    pycolmap.extract_features(
        db, root, names, camera_mode=pycolmap.CameraMode.PER_IMAGE, reader_options=ro
    )
    # the reader guesses a focal length from the file; ours are known, so overwrite each camera
    with pycolmap.Database.open(db) as d:
        for f in tf["frames"]:
            im = d.read_image_with_name(f["file_path"])
            cam = d.read_camera(im.camera_id)
            cam.width, cam.height = int(f["w"]), int(f["h"])
            cam.params = [f["fl_x"], f["fl_y"], f["cx"], f["cy"]]
            cam.has_prior_focal_length = True
            d.update_camera(cam)


def pick_pairs(poses: list[np.ndarray], k: int) -> list[tuple[int, int]]:
    c = np.array([p[:3, 3] for p in poses])
    z = np.array([p[:3, 2] for p in poses])  # OpenCV viewing direction
    d = np.linalg.norm(c[:, None] - c[None], axis=2)
    cos = z @ z.T
    ok = cos > np.cos(np.radians(MAX_ANGLE))
    pairs = set()
    for i in range(len(poses)):
        cand = [j for j in np.argsort(d[i]) if j != i and ok[i, j]][:k]
        pairs.update((min(i, j), max(i, j)) for j in cand)
    return sorted(pairs)


def match(db: Path, names: list[str], pairs: list[tuple[int, int]], out: Path) -> None:
    lst = out / "pairs.txt"
    lst.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in pairs))
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = str(lst)
    pycolmap.match_image_pairs(db, pairing_options=po)


def build_recon(db: Path, tf: dict, poses: list[np.ndarray]) -> pycolmap.Reconstruction:
    rec = pycolmap.Reconstruction()
    with pycolmap.Database.open(db) as d:
        for cam in d.read_all_cameras():
            rec.add_camera(cam)
        for rig in d.read_all_rigs():
            rec.add_rig(rig)
        for f, pose in zip(tf["frames"], poses, strict=True):
            im = d.read_image_with_name(f["file_path"])
            kp = d.read_keypoints(im.image_id)[:, :2]
            new = pycolmap.Image(f["file_path"], kp.astype(np.float64), im.camera_id, im.image_id)
            rec.add_image_with_trivial_frame(new, pycolmap.Rigid3d(np.linalg.inv(pose)[:3]))
    return rec


def triangulate(rec: pycolmap.Reconstruction, db: Path, root: Path, work: Path):
    op = pycolmap.IncrementalPipelineOptions()
    op.triangulation.min_angle = 3.0  # shallower tracks drift to infinity in BA
    return pycolmap.triangulate_points(rec, db, root, work, clear_points=True, options=op)


def mean_err(rec: pycolmap.Reconstruction) -> float:
    return float(rec.compute_mean_reprojection_error())


def drop_outliers(rec: pycolmap.Reconstruction) -> int:
    bad = [i for i, p in rec.points3D.items() if p.error > MAX_REPROJ or p.track.length() < 2]
    for i in bad:
        rec.delete_point3D(i)
    return len(bad)


def bundle(rec: pycolmap.Reconstruction) -> None:
    """Pose + point BA, intrinsics fixed, Huber loss. The gauge is fixed on two cameras and the
    result is aligned back onto the originals afterwards."""
    o = pycolmap.BundleAdjustmentOptions()
    o.refine_focal_length = o.refine_principal_point = o.refine_extra_params = False
    o.print_summary = False
    o.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    o.ceres.loss_function_scale = 1.0  # px
    cfg = pycolmap.BundleAdjustmentConfig()
    for i in rec.reg_image_ids():
        cfg.add_image(i)
    for i in rec.point3D_ids():
        cfg.add_variable_point(i)
    for cam in rec.cameras:
        cfg.set_constant_cam_intrinsics(cam)
    cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    pycolmap.create_default_bundle_adjuster(o, cfg, rec).solve()
    rec.update_point_3d_errors()  # the stored errors are stale after a solve


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity s, R, t with dst ~ s R src + t."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd(b.T @ a / len(src))
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    r = u @ s @ vt
    sc = (sv * np.diag(s)).sum() / a.var(0).sum()
    return float(sc), r, md - sc * r @ ms


def rot_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(a.T @ b) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def refined_poses(rec: pycolmap.Reconstruction, tf: dict, poses: list[np.ndarray], db: Path):
    """Camera-to-world poses back in the original frame; frames BA could not pin keep theirs."""
    with pycolmap.Database.open(db) as d:
        ids = [d.read_image_with_name(f["file_path"]).image_id for f in tf["frames"]]
    new, used = list(poses), []
    for k, i in enumerate(ids):
        im = rec.images[i]
        if im.has_pose and im.num_points3D >= MIN_OBS:
            new[k] = np.linalg.inv(np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]]))
            used.append(k)
    sc, r, t = umeyama(
        np.array([new[k][:3, 3] for k in used]), np.array([poses[k][:3, 3] for k in used])
    )
    for k in used:
        new[k][:3, :3] = r @ new[k][:3, :3]
        new[k][:3, 3] = sc * r @ new[k][:3, 3] + t
    return new, used, (sc, r, t), ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-pairs-per-image", type=int, default=15)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed-points", action="store_true", help="write the BA points to points.ply")
    a = ap.parse_args()
    t0 = time.time()

    copy_dataset(a.src, a.out)
    root = a.out
    work = root / "colmap"
    work.mkdir()
    db = work / "database.db"
    tf = json.loads((root / "transforms.json").read_text())
    names = [f["file_path"] for f in tf["frames"]]
    poses = load_poses(tf)
    masked = any("mask_path" in f for f in tf["frames"])
    if masked:
        write_masks(root, tf, work / "masks")

    extract(db, root, tf, masked)
    pairs = pick_pairs(poses, a.max_pairs_per_image)
    match(db, names, pairs, work)
    with pycolmap.Database.open(db) as d:
        verified = d.num_verified_image_pairs()
    print(f"pairs chosen {len(pairs)}, verified {verified}")

    rec = build_recon(db, tf, poses)
    rec = triangulate(rec, db, root, work)
    before = mean_err(rec)
    print(f"initial: {rec.num_points3D()} points, mean reproj {before:.3f} px")
    for it in range(a.iters):
        bundle(rec)
        n = drop_outliers(rec)
        bundle(rec)
        print(
            f"iter {it + 1}: BA, dropped {n}, {rec.num_points3D()} points, "
            f"mean reproj {mean_err(rec):.3f} px"
        )
        if it + 1 < a.iters:
            rec = triangulate(rec, db, root, work)  # re-triangulate with the refined poses
    after = mean_err(rec)

    new, used, gauge, _ = refined_poses(rec, tf, poses, db)
    sc = gauge[0]
    dc = np.array([np.linalg.norm(new[k][:3, 3] - poses[k][:3, 3]) for k in used]) * 100
    dr = np.array([rot_deg(new[k][:3, :3], poses[k][:3, :3]) for k in used])
    print(
        f"sim3 scale {sc:.5f}; registered {len(used)}/{len(poses)}, kept {len(poses) - len(used)}"
    )
    print(f"position change cm: median {np.median(dc):.2f} max {dc.max():.2f}")
    print(f"rotation change deg: median {np.median(dr):.3f} max {dr.max():.3f}")
    # the BA's own error is in its free gauge; re-triangulating from the aligned poses is the
    # same measure as "before", so the two are comparable
    check = triangulate(build_recon(db, tf, new), db, root, work)
    print(
        f"mean reproj before {before:.3f} px, after BA {after:.3f} px, "
        f"re-triangulated from refined poses {mean_err(check):.3f} px ({check.num_points3D()} pts)"
    )

    for f, p in zip(tf["frames"], new, strict=True):
        f["transform_matrix"] = (p @ GL).tolist()
    (root / "transforms.json").write_text(json.dumps(tf, indent=2))
    rec.write(work)  # still in COLMAP's gauge; the sim3 maps it back to the room frame
    np.savez(work / "gauge.npz", s=sc, R=gauge[1], t=gauge[2])
    if a.seed_points:
        pts = list(rec.points3D.values())
        xyz = np.array([p.xyz for p in pts]) @ gauge[1].T * sc + gauge[2]
        rgb = np.array([p.color for p in pts], np.uint8)
        write_ply(root / "points.ply", xyz, rgb)
    print(f"done in {time.time() - t0:.0f}s")


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    head = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rec = np.zeros(len(xyz), [("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = xyz, rgb
    path.write_bytes(head.encode() + rec.tobytes())


if __name__ == "__main__":
    main()
