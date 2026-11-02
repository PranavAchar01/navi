"""Quality of a room map, for recon_v2.py's score step.

continuity   per walk, the share of consecutive frames whose camera jumps over 30 cm (a head
             moves centimetres a frame; a jump is a frame registered in the wrong place)
xwalk        SIFT keypoints of a frame from one walk, lifted onto the mesh through its camera
             and projected into a frame of another walk that looks at the same place (DINO
             retrieval), against where the match really is: pixels at 504 x 896. Mesh and both
             poses have to be right for this to be small.
heldout      frames of bedroom walks the map never fused, placed against the map by PnP from
             such lifted matches: share placed and the inliers' reprojection error.
scale        heights the map measures against standard sizes (desk top, door, ceiling).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metasight import recon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WALKS = ROOT / "data/walks"
FW, FH = 504, 896  # stored frame size
W, H = 294, 518
HELDOUT = [
    "20261029-182525-glasses",
    "20261102-023733-glasses",
    "20261102-025349-glasses",
    "20261102-030055-glasses",
]
REF = {"desk_top_m": 0.74, "door_height_m": 2.03, "ceiling_8ft_m": 2.44, "ceiling_9ft_m": 2.74}
_sift = cv2.SIFT_create(2000)


def k_full(k: np.ndarray) -> np.ndarray:
    s = np.diag([FW / W, FH / H, 1.0])
    return s @ k


def feats(path: Path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return _sift.detectAndCompute(g, None)


def match(d1, d2) -> list[tuple[int, int]]:
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return []
    m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    return [
        (a.queryIdx, a.trainIdx)
        for a, b in (p for p in m if len(p) == 2)
        if a.distance < 0.75 * b.distance
    ]


def lift(scene, cam: dict, uv: np.ndarray) -> np.ndarray:
    """Mesh points seen at full-size pixels uv from this camera (nan where the ray misses)."""
    from recon_v2 import cast

    k = k_full(cam["k"])
    d = np.column_stack(
        [(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))]
    )
    d = d @ cam["pose"][:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t, _ = cast(scene, np.repeat(cam["pose"][None, :3, 3], len(d), 0), d)
    p = cam["pose"][:3, 3] + d * t[:, None]
    p[~np.isfinite(t) | (t > 6)] = np.nan
    return p


def project(cam_pose: np.ndarray, k: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(cam_pose)
    q = p @ w2c[:3, :3].T + w2c[:3, 3]
    uv = q[:, :2] / q[:, 2:3] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    return uv, q[:, 2]


def embed(paths: list[Path]) -> np.ndarray:
    from room_dense import embed as dino

    out = []
    for i in range(0, len(paths), 64):
        ims = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths[i : i + 64]]
        out.append(dino(ims))
    return np.concatenate(out)


def score(cams: dict, mesh, scene, out: Path, n_ref: int = 1200, n_q: int = 160) -> dict:
    rng = np.random.default_rng(0)
    cont = {
        w: round(recon.continuity(np.array([r["pose"][:3, 3] for r in rows])), 3)
        for w, rows in cams.items()
    }
    trusted = {
        w: round(float(np.mean([r["trusted"] for r in rows])), 3) for w, rows in cams.items()
    }
    pool = [r for rows in cams.values() for r in rows if r["trusted"]]
    refs = [pool[i] for i in sorted(rng.choice(len(pool), min(n_ref, len(pool)), replace=False))]
    ref_paths = [WALKS / r["walk"] / "frames" / r["file"] for r in refs]
    e_ref = embed(ref_paths)
    k_glasses = np.median(
        [r["k"] for rows in cams.values() for r in rows if "glasses" in r["walk"]], 0
    )

    # Cross-walk: each ref against its best match from another walk.
    xerr, xpairs = [], 0
    for i in rng.choice(len(refs), min(150, len(refs)), replace=False):
        sim = e_ref @ e_ref[i]
        other = [j for j in np.argsort(-sim)[:40] if refs[j]["walk"] != refs[i]["walk"]]
        if not other or sim[other[0]] < 0.6:
            continue
        j = other[0]
        ka, da = feats(ref_paths[i])
        kb, db = feats(ref_paths[j])
        mm = match(da, db)
        if len(mm) < 20:
            continue
        ua = np.array([ka[a].pt for a, _ in mm])
        ub = np.array([kb[b].pt for _, b in mm])
        p = lift(scene, refs[i], ua)
        ok = np.isfinite(p).all(1)
        uv, z = project(refs[j]["pose"], k_full(refs[j]["k"]), p[ok])
        e = np.linalg.norm(uv - ub[ok], axis=1)[z > 0.1]
        if len(e) >= 10:
            xerr.append(float(np.median(e)))
            xpairs += 1

    # Held-out walks: PnP against lifted matches from the 3 nearest refs.
    q_paths = []
    for w in HELDOUT:
        fs = sorted((WALKS / w / "frames").glob("*.jpg"))
        if fs:
            q_paths += [fs[int(i)] for i in np.linspace(0, len(fs) - 1, n_q // len(HELDOUT))]
    e_q = embed(q_paths)
    kq = k_full(k_glasses)
    placed, rms, inl = 0, [], []
    for qi, qp in enumerate(q_paths):
        kq_, dq = feats(qp)
        P, U = [], []
        for j in np.argsort(-(e_ref @ e_q[qi]))[:3]:
            kr, dr = feats(ref_paths[j])
            mm = match(dr, dq)
            if len(mm) < 12:
                continue
            p = lift(scene, refs[j], np.array([kr[a].pt for a, _ in mm]))
            ok = np.isfinite(p).all(1)
            P.append(p[ok])
            U.append(np.array([kq_[b].pt for _, b in mm])[ok])
        if not P or sum(len(p) for p in P) < 20:
            continue
        P, U = np.concatenate(P), np.concatenate(U)
        good, rv, tv, idx = cv2.solvePnPRansac(
            P.astype(np.float64),
            U.astype(np.float64),
            kq,
            None,
            reprojectionError=8.0,
            iterationsCount=3000,
            confidence=0.999,
        )
        if not good or idx is None or len(idx) < 30:
            continue
        idx = idx[:, 0]
        proj, _ = cv2.projectPoints(P[idx], rv, tv, kq, None)
        e = np.linalg.norm(proj[:, 0] - U[idx], axis=1)
        placed += 1
        rms.append(float(np.sqrt(np.mean(e**2))))
        inl.append(len(idx) / len(P))

    objs = json.loads((out / "objects.json").read_text()) if (out / "objects.json").exists() else {}
    scale = scale_check(objs)
    summary = {
        "frames_fused": sum(len(v) for v in cams.values()),
        "trusted_share": round(float(np.mean([r["trusted"] for v in cams.values() for r in v])), 3),
        "xwalk_reproj_px_median": round(float(np.median(xerr)), 2) if xerr else None,
        "xwalk_pairs": xpairs,
        "heldout_placed": f"{placed}/{len(q_paths)}",
        "heldout_rms_px_median": round(float(np.median(rms)), 2) if rms else None,
        "heldout_inlier_share_median": round(float(np.median(inl)), 2) if inl else None,
        "scale": scale,
        "objects": len(objs.get("objects", [])),
    }
    return {"summary": summary, "continuity_jump_share": cont, "trusted_share": trusted}


def scale_check(doc: dict) -> dict:
    """Measured heights over standard ones: 1.0 means the map is metric."""
    out: dict = {}
    objs = doc.get("objects", [])
    desks = [o for o in objs if o["class"] == "desk"]
    if desks:
        d = max(desks, key=lambda o: o["seen"]["frames"])
        out["desk_top_m"] = d["top_height_m"]
        out["desk_ratio"] = round(d["top_height_m"] / REF["desk_top_m"], 3)
    doors = [o for o in objs if o["class"] == "door" and o["size_m"]["height"] > 1.5]
    if doors:
        h = float(np.median([o["size_m"]["height"] for o in doors]))
        out["door_height_m"] = round(h, 3)
        out["door_ratio"] = round(h / REF["door_height_m"], 3)
    if "ceiling_height_m" in doc:
        c = doc["ceiling_height_m"]
        out["ceiling_m"] = c
        out["ceiling_ratio_8ft"] = round(c / REF["ceiling_8ft_m"], 3)
        out["ceiling_ratio_9ft"] = round(c / REF["ceiling_9ft_m"], 3)
    return out
