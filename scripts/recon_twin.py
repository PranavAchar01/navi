"""Turn a recon-v2 version into a living object twin, and measure what the twin buys.

Reads the version's label output (seen.npz: every mask's frame, class and mesh triangles;
voted_frames.json) and re-votes it walk by walk, so each walk gives its own observations of each
object. metasight.twin folds the walks in time order into tracks with moved / added / missing /
removed events. Written as twin.json.

Then three measurements, written as twin-metrics.json:
  hazards   For every voted frame, the hazards within 1 m of the camera (horizontal distance
            to the box footprint), and how many of those are outside the camera's view. A
            frame-only detector cannot know those; the twin does.
  where     "Where is my X", causal: for each walk, the twin built from the earlier walks only
            answers every object that walk saw. Hit when the most recent answer (top-1) or any
            answer is within 0.5 m (floor plane) of where the walk found it. Baseline: the old
            static home/objects.json, built from every walk including the one asked about.
  sharpen   Object size from the first k masks of an object versus from all of them: the
            median relative error of the box dimensions for k = 1, 2, 4 ... 64.

    PYTHONPATH=. .venv/bin/python scripts/recon_twin.py --version v2.1 [--cams PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from recon_score import project  # noqa: E402
from recon_v2 import (  # noqa: E402
    CLASSES,
    ROOM,
    H,
    W,
    cast,
    clusters,
    load_cams,
    scene_of,
    walk_ts,
)

from metasight import recon, twin  # noqa: E402

MIN_FRAMES = 3  # an observation needs masks in this many frames of the walk
NEAR_M = 1.0
LOOK_FRAMES = 5  # a spot counts as looked at after this many unoccluded views
HIT_M = 0.5
# home/objects.json (COCO labels) to the open-vocabulary classes, for the baseline.
COCO = {
    "bed": "bed",
    "chair": "chair",
    "laptop": "laptop",
    "tv": "computer monitor",
    "keyboard": "keyboard",
    "backpack": "backpack",
    "handbag": "bag",
    "potted plant": "plant",
    "dining table": "desk",
    "suitcase": "bag",
}


def observe(instances, ntri, s, starts, frames, cams_by, verts, tris) -> dict[str, list[dict]]:
    """Each walk's sightings of the map's object instances.

    Instances come from the votes of every walk together (the ConceptGraphs pattern: fuse
    first, then ask each observation which instance it belongs to), so a walk cannot invent
    an object out of a few stray masks. A walk sees an instance when its masks of the same
    class family cover at least 10 of the instance's triangles in MIN_FRAMES frames; the
    sighting's box is fitted to just the triangles that walk's masks covered.
    """
    owner = np.full(ntri, -1, np.int64)
    for n, (_name, ids, _pts, _box) in enumerate(instances):
        owner[ids] = n
    fam = [twin.family(name) for name, *_ in instances]
    hits: dict[tuple[str, int], tuple[set[int], list[np.ndarray]]] = {}
    for r in range(len(s["frame"])):
        ids = s["ids"][starts[r] : starts[r + 1]]
        own = owner[ids]
        cnt = np.bincount(own[own >= 0], minlength=len(instances))
        f = int(s["frame"][r])
        for n in np.flatnonzero(cnt >= 10):
            if fam[n] != twin.family(CLASSES[s["cls"][r]]):
                continue
            fr, parts = hits.setdefault((frames[f]["walk"], int(n)), (set(), []))
            fr.add(f)
            parts.append(ids[own == n])
    out: dict[str, list[dict]] = {}
    for (walk, n), (fr, parts) in hits.items():
        if len(fr) < MIN_FRAMES:
            continue
        ids = np.unique(np.concatenate(parts))
        box = recon.upright_box(verts[tris[ids].reshape(-1)])
        ts = sorted(walk_ts(walk, cams_by[f]["file"]) for f in fr)
        out.setdefault(walk, []).append(
            {
                "class": instances[n][0],
                "instance": n,
                "pos": np.round(box["center"], 3).tolist(),
                "size": np.round(box["size"], 3).tolist(),
                "yaw_rad": round(float(box["yaw"]), 3),
                "frames": len(fr),
                "ts_ms": ts[len(ts) // 2],
            }
        )
    return out


def as_obb(o: dict) -> dict:
    return {"obb": {"center": o["pos"], "size": o["size"], "yaw_rad": o.get("yaw_rad", 0.0)}}


def footprint_dist(c: np.ndarray, o: dict) -> float:
    """Horizontal distance from a point to an object's oriented footprint (0 inside)."""
    b = o["obb"]
    yaw = b["yaw_rad"]
    d = c[[0, 2]] - np.asarray(b["center"])[[0, 2]]
    ax = np.array([np.cos(yaw), np.sin(yaw)])
    az = np.array([-np.sin(yaw), np.cos(yaw)])
    half = np.array([b["size"][0], b["size"][2]]) / 2
    q = np.abs([d @ ax, d @ az]) - half
    return float(np.linalg.norm(np.maximum(q, 0)))


def in_view(cam: dict, o: dict) -> bool:
    b = o["obb"]
    top = np.asarray(b["corners_top"])
    bot = top.copy()
    bot[:, 1] = b["center"][1] - b["size"][1] / 2
    pts = np.vstack([top, bot, [b["center"]]])
    uv, z = project(cam["pose"], cam["k"], pts)
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return bool(ok.any())


def hazards(doc, cams_by, fidx) -> dict:
    hz = [o for o in doc["objects"] if o.get("hazard")]
    near = out = 0
    by: dict[str, list[int]] = {}
    for f in fidx:
        cam = cams_by[f]
        c = cam["pose"][:3, 3]
        for o in hz:
            if footprint_dist(c, o) > NEAR_M:
                continue
            seen = in_view(cam, o)
            near += 1
            out += not seen
            by.setdefault(o["class"], [0, 0])
            by[o["class"]][0] += 1
            by[o["class"]][1] += not seen
    return {
        "frames": len(fidx),
        "hazards_within_1m": near,
        "out_of_view": out,
        "out_of_view_share": round(out / max(near, 1), 3),
        "by_class": {k: {"near": v[0], "out_of_view": v[1]} for k, v in sorted(by.items())},
    }


def flat(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[2] - b[2]))


def where(walks, visible, baseline) -> dict:
    """Each walk asks for every object it saw. Answers come from the earlier walks only.

    twin: the tracks' latest positions (top-1 = the most recently seen of that class family).
    scan_once: the same tracks frozen at their first sighting, i.e. a map scanned once and
    never updated. home: the old static home/objects.json (COCO labels), which was built from
    every walk, the asked one included, so it is an upper bound for a static map.
    Rates are over the objects whose instance an earlier walk had seen ("known"); objects no
    earlier walk saw cannot be answered by any map.
    """
    c = {"asked": 0, "known": 0, "twin_top1": 0, "twin_any": 0, "scan_once_any": 0}
    c |= {"twin_top1_extent": 0, "point_top1_err_m": [], "extent_top1_err_m": []}
    cc = {"known_coco": 0, "twin_any_coco": 0, "home_any_coco": 0}
    per = []
    for i in range(1, len(walks)):
        tracks = twin.build_tracks(walks[:i], visible)
        w, _, obs = walks[i]
        h = 0
        for o in obs:
            c["asked"] += 1
            earlier = {sg["instance"] for t in tracks for sg in t["sightings"]}
            if o["instance"] not in earlier:
                continue
            c["known"] += 1
            r = twin.reach(o)
            ans = twin.where_is(tracks, o["class"])
            fam = twin.family(o["class"])
            frozen = [t for t in tracks if twin.family(t["class"]) == fam]
            top1 = bool(ans) and flat(ans[0]["pos"], o["pos"]) < r
            hit = any(flat(a["pos"], o["pos"]) < r for a in ans)
            c["twin_top1"] += top1
            if ans:
                # Guidance target as a point (the old pins) vs as the last box (pose + extent):
                # how far the true object is from what the wearer is walked to.
                ext = footprint_dist(np.asarray(o["pos"], float), as_obb(ans[0]["last"]))
                c["point_top1_err_m"].append(flat(ans[0]["pos"], o["pos"]))
                c["extent_top1_err_m"].append(ext)
                c["twin_top1_extent"] += ext < HIT_M
            c["twin_any"] += hit
            c["scan_once_any"] += any(flat(t["sightings"][0]["pos"], o["pos"]) < r for t in frozen)
            h += top1
            if any(twin.family(v) == fam for v in COCO.values()):
                bl = [b for b in baseline if twin.family(COCO.get(b["label"], "")) == fam]
                cc["known_coco"] += 1
                cc["twin_any_coco"] += hit
                cc["home_any_coco"] += any(flat(b["pos"], o["pos"]) < r for b in bl)
        per.append({"walk": w, "asked": len(obs), "top1_hits": h})
    k, kc = max(c["known"], 1), max(cc["known_coco"], 1)
    return {
        "asked": c["asked"],
        "known": c["known"],
        "twin_top1": round(c["twin_top1"] / k, 3),
        "twin_top1_extent_within_0.5m": round(c["twin_top1_extent"] / k, 3),
        "top1_median_err_m_point": round(float(np.median(c["point_top1_err_m"] or [0])), 3),
        "top1_median_err_m_extent": round(float(np.median(c["extent_top1_err_m"] or [0])), 3),
        "twin_any": round(c["twin_any"] / k, 3),
        "scan_once_any": round(c["scan_once_any"] / k, 3),
        "known_coco": cc["known_coco"],
        "twin_any_coco": round(cc["twin_any_coco"] / kc, 3),
        "home_any_coco": round(cc["home_any_coco"] / kc, 3),
        "per_walk": per,
    }


def sharpen(instances, s, starts, verts, tris) -> dict:
    """Size from the first k masks of each object vs from all its masks."""
    ks = [1, 2, 4, 8, 16, 32, 64]
    errs: dict[int, list[float]] = {k: [] for k in ks}
    order = np.argsort(s["frame"], kind="stable")  # frames are voted in walk/time order
    for name, ids, _pts, full in instances:
        ci = CLASSES.index(name)
        own = np.zeros(len(tris), bool)
        own[ids] = True
        masks = [
            s["ids"][starts[r] : starts[r + 1]]
            for r in order
            if s["cls"][r] == ci and own[s["ids"][starts[r] : starts[r + 1]]].sum() >= 10
        ]
        if len(masks) < 8:
            continue
        ref = np.sort(full["size"])
        acc = np.zeros(len(tris), bool)
        done = 0
        for k in ks:
            if k > len(masks):
                break
            for m in masks[done:k]:
                acc[m[own[m]]] = True
            done = k
            box = recon.upright_box(verts[tris[acc].reshape(-1)])
            errs[k].append(float(np.median(np.abs(np.sort(box["size"]) - ref) / ref)))
    return {
        str(k): {"objects": len(v), "median_rel_err": round(float(np.median(v)), 3)}
        for k, v in errs.items()
        if v
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--cams", help="cams file the version was voted with (recon_reloc output)")
    a = ap.parse_args()
    import open3d as o3d

    out = ROOM / "recon-v2" / a.version
    doc = json.loads((out / "objects.json").read_text())
    mesh_path = Path(doc["source"]["mesh"])
    mesh_dir, tag = mesh_path.parent, mesh_path.stem.split("-", 1)[1]
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = scene_of(mesh)
    verts, tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    cams = load_cams(mesh_dir, tag, Path(a.cams) if a.cams else None)
    look = {(r["walk"], r["file"]): r for rows in cams.values() for r in rows}
    frames = json.loads((out / "voted_frames.json").read_text())
    cams_by = [look[(f["walk"], f["file"])] for f in frames]
    s = dict(np.load(out / "seen.npz"))
    starts = np.concatenate([[0], np.cumsum(s["n"])])

    by_walk: dict[str, list[int]] = {}
    for i, f in enumerate(frames):
        by_walk.setdefault(f["walk"], []).append(i)
    v = np.load(out / "votes.npz")
    votes, vis = v["votes"], v["vis"]
    instances = clusters(votes, vis, verts, tris)
    seen_by = observe(instances, len(tris), s, starts, frames, cams_by, verts, tris)
    walks = []
    for w, fidx in by_walk.items():
        obs = seen_by.get(w, [])
        t0 = min(walk_ts(w, frames[i]["file"]) for i in fidx)
        walks.append((w, t0, obs))
        print(w, len(fidx), "frames", len(obs), "observations", flush=True)
    walks.sort(key=lambda x: x[1])

    rig = {}
    for w, fidx in by_walk.items():
        poses = np.array([cams_by[i]["pose"] for i in fidx])
        w2c = np.linalg.inv(poses)
        rig[w] = (w2c, np.array([cams_by[i]["k"] for i in fidx]), poses[:, :3, 3])
    cache: dict[tuple[int, str], bool] = {}

    def visible(o: dict, w: str) -> bool:
        """The camera looked at this spot, unblocked, in LOOK_FRAMES frames of the walk."""
        key = (id(o), w)
        if key not in cache:
            w2c, k, org = rig[w]
            p = np.asarray(o["pos"], float)
            q = np.einsum("nij,j->ni", w2c[:, :3, :3], p) + w2c[:, :3, 3]
            z = q[:, 2]
            u = q[:, 0] / np.maximum(z, 1e-6) * k[:, 0, 0] + k[:, 0, 2]
            v = q[:, 1] / np.maximum(z, 1e-6) * k[:, 1, 1] + k[:, 1, 2]
            sel = (z > 0.3) & (z < 4.0) & (u > 0) & (u < W) & (v > 0) & (v < H)
            n = 0
            if sel.any():
                d = p - org[sel]
                dist = np.linalg.norm(d, axis=1)
                t, _ = cast(scene, org[sel], d / dist[:, None])
                n = int((t > dist - twin.reach(o) - 0.15).sum())
            cache[key] = n >= LOOK_FRAMES
        return cache[key]

    tracks = twin.build_tracks(walks, visible)
    for i, t in enumerate(sorted(tracks, key=lambda t: -len(t["sightings"]))):
        t["id"] = f"{t['class'].replace(' ', '_')}_t{i:02d}"
        near = [
            o
            for o in doc["objects"]
            if twin.family(o.get("detector_class", o["class"])) == twin.family(t["class"])
            and flat(o["pos"], t["pos"]) < HIT_M
        ]
        t["object_id"] = min(near, key=lambda o: flat(o["pos"], t["pos"]))["id"] if near else None
        t.pop("last")
        t.pop("missing_walks")
    tracks.sort(key=lambda t: t["id"])
    events = sorted(
        (dict(e, track=t["id"]) for t in tracks for e in t["events"]), key=lambda e: e["ts_ms"]
    )
    (out / "twin.json").write_text(
        json.dumps(
            {
                "version": a.version,
                "frame": doc["frame"],
                "walks": [
                    {"walk": w, "start_ts_ms": t0, "observations": len(o)} for w, t0, o in walks
                ],
                "tracks": tracks,
                "events": events,
            },
            indent=1,
        )
    )
    baseline = json.loads((ROOM / "home/objects.json").read_text())
    bed = [b for b in baseline if b["label"] == "bed"]
    mine = [o for o in doc["objects"] if o["class"] == "bed"]
    metrics = {
        "tracks": len(tracks),
        "events": {
            k: sum(e["type"] == k for e in events)
            for k in ["first_seen", "added", "moved", "missing", "removed"]
        },
        "hazards": hazards(doc, cams_by, list(range(len(frames)))),
        "where": where(walks, visible, baseline),
        "sharpen": sharpen(instances, s, starts, verts, tris),
        "baseline_frame_check_bed_m": round(flat(bed[0]["pos"], mine[0]["pos"]), 3)
        if bed and mine
        else None,
    }
    (out / "twin-metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k != "where"}, indent=1))
    print(json.dumps({k: v for k, v in metrics["where"].items() if k != "per_walk"}))


if __name__ == "__main__":
    import shutil

    from room_watch import take_slot

    slot = take_slot()
    try:
        main()
    finally:
        shutil.rmtree(slot, ignore_errors=True)
