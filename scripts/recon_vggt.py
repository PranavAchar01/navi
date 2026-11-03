"""VGGT-1B against MapAnything for camera poses, under room_dense.py's chunking and alignment.

Each walk is subsampled (every --stride frame) and interleaved into chunks of NEW frames plus
ANCHORS room frames exactly as room_dense.py does; only the model that reconstructs a chunk
changes. VGGT's units are arbitrary, so a chunk is first brought to metres by the median ratio of
the stored anchor depth to VGGT's anchor depth, then pinned into the room by room_dense.fit_chunk
with its own gates (6 cm fit residual, scale 0.5 to 2). MapAnything's side is the bedroom v003
cams.jsonl, restricted to the same frames.

    PYTHONPATH=. .venv/bin/python scripts/recon_vggt.py run <walk> [<walk> ...] [--stride 2]
    PYTHONPATH=. .venv/bin/python scripts/recon_vggt.py score <walk> [<walk> ...]

Writes data/rooms/bedroom/vggt/cams.jsonl (same rows as room_dense), chunks.jsonl and score.json.

RESULT: filled in after the run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_dense import ANCHORS, FIT_MAX_M, NEW, embed, fit_chunk, umeyama  # noqa: E402
from room_mesh import H, W, pool_frames  # noqa: E402

from metasight import recon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WALKS = ROOT / "data/walks"
ROOM = ROOT / "data/rooms/bedroom"
OUT = ROOM / "vggt"
MA_CAMS = ROOM / "v003/cams.jsonl"
ANCHOR_DIR = ROOM / "anchors"
ANCHOR_WALKS = ["20261031-004732-glasses"]  # what v003 was built with (its log.txt)
VIEWS = 160
CONF_PCT = 20  # drop VGGT's least confident fifth of depth pixels, as MapAnything's run does


def free_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    line = next((ln for ln in out.splitlines() if "free percentage" in ln), "0%")
    return int(line.split()[-1].rstrip("%"))


def take_slot() -> Path:
    """First free heavy-lock slot (a dead owner's slot is stale), once memory and disk allow."""
    while free_pct() < 10 or shutil.disk_usage(ROOT).free < 15 * 2**30:
        time.sleep(30)
    n = 1
    while True:
        slot = ROOT.parent / (".heavy-lock" if n == 1 else f".heavy-lock-{n}")
        try:
            slot.mkdir()
        except FileExistsError:
            try:
                os.kill(int((slot / "owner").read_text()), 0)
            except (ProcessLookupError, ValueError, FileNotFoundError):
                (slot / "owner").unlink(missing_ok=True)
                slot.rmdir()
                continue
            n += 1
            continue
        (slot / "owner").write_text(str(os.getpid()))
        return slot


def release(slot: Path) -> None:
    (slot / "owner").unlink(missing_ok=True)
    slot.rmdir()


def load_model():
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.point_head = model.track_head = None  # poses and depth only
    return model.to("mps").eval()


def reconstruct(paths: list[Path], model) -> list[dict]:
    """room_mesh.reconstruct's view dicts from VGGT: camera-to-world (OpenCV axes), depth, mask
    and intrinsics at 294 x 518, in VGGT's own units with the first frame at the origin."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    rgbs = [cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (W, H)), cv2.COLOR_BGR2RGB) for p in paths]
    x = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float().div(255).to("mps")
    with torch.no_grad(), torch.autocast("mps", dtype=torch.float16):
        pred = model(x)
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
    ext, intr = ext[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()
    conf = pred["depth_conf"][0].float().cpu().numpy()
    del pred
    thr = np.percentile(conf, CONF_PCT)
    out = []
    for i, rgb in enumerate(rgbs):
        pose = np.eye(4)
        pose[:3, :3] = ext[i, :, :3].T
        pose[:3, 3] = -ext[i, :, :3].T @ ext[i, :, 3]
        out.append(dict(pose=pose, depth=depth[i], mask=conf[i] > thr, k=intr[i], rgb=rgb))
    return out


def to_metres(views: list[dict], anchors: list[dict]) -> float:
    """Metres per VGGT unit: median of stored anchor depth over VGGT's anchor depth. Applied
    before fit_chunk, so its metric depth window and scale gate mean what they do for
    MapAnything."""
    ratios = []
    for v, a in zip(views, anchors, strict=True):
        ok = v["mask"] & a["mask"].astype(bool) & (a["depth"] > 0.1) & (v["depth"] > 1e-4)
        ratios.append(a["depth"][ok] / v["depth"][ok])
    return float(np.median(np.concatenate(ratios)))


def walk_frames(walk: Path, stride: int, skip: set[Path]) -> list[Path]:
    """room_dense's chunk order: every stride-th frame, interleaved so each chunk spans the walk."""
    rows = [json.loads(ln) for ln in (walk / "frames.jsonl").read_text().splitlines() if ln]
    todo = [p for p in [walk / "frames" / r["file"] for r in rows][::stride] if p not in skip]
    n = -(-len(todo) // NEW)
    return [q for i in range(n) for q in todo[i::n]]


def run(a: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ANCHOR_DIR / "views.npz")
    keys = ("pose", "depth", "mask", "k", "rgb")
    anchor_views = [{k: z[k][i] for k in keys} for i in range(len(z["pose"]))]
    anchor_paths = pool_frames([WALKS / w for w in ANCHOR_WALKS], VIEWS)
    assert len(anchor_paths) == len(anchor_views)
    anchor_of = dict(zip(anchor_paths, anchor_views, strict=True))
    cams_path = OUT / "cams.jsonl"
    done = set()
    if cams_path.exists():
        done = {
            (r["walk"], r["chunk_start"], r["stride"])
            for r in map(json.loads, (OUT / "chunks.jsonl").read_text().splitlines())
        }
    slot = take_slot()
    try:
        anchor_emb = embed([v["rgb"] for v in anchor_views])
        t = time.time()
        model = load_model()
        print(f"model loaded in {time.time() - t:.0f}s", flush=True)
        log = (OUT / "chunks.jsonl").open("a")
        cams = cams_path.open("a")
        for name in a.walk:
            todo = walk_frames(WALKS / name, a.stride, set(anchor_paths))
            for c in range(0, len(todo), a.new):
                if (name, c, a.stride) in done:
                    continue
                while free_pct() < 10:
                    time.sleep(30)
                new = todo[c : c + a.new]
                rgbs = [cv2.cvtColor(cv2.resize(cv2.imread(str(q)), (W, H)), 4) for q in new]
                sim = (embed(rgbs) @ anchor_emb.T).max(0)
                anc = [anchor_paths[i] for i in np.argsort(-sim)[: a.anchors]]
                t = time.time()
                views = reconstruct(anc + new, model)
                dt = time.time() - t
                mem = torch.mps.driver_allocated_memory() / 2**30
                torch.mps.empty_cache()
                k = to_metres(views[: len(anc)], [anchor_of[p] for p in anc])
                for v in views:
                    v["depth"] = v["depth"] * k
                    v["pose"][:3, 3] *= k
                s, r, tr, res = fit_chunk(views[: len(anc)], [anchor_of[p] for p in anc])
                ok = res < FIT_MAX_M and 0.5 < s < 2.0
                row = dict(walk=name, stride=a.stride, chunk_start=c, n=len(new), views=len(views))
                row |= dict(sec=round(dt, 1), gib=round(mem, 1), unit_m=k, scale=s, res_m=res)
                log.write(json.dumps(row | dict(used=ok)) + "\n")
                log.flush()
                print(
                    f"{name} {c}+{len(new)}: {dt:.0f}s ({dt / len(views):.2f} s/frame),"
                    f" {mem:.1f} GiB, fit {res * 100:.1f} cm, scale {s:.2f},"
                    f" {'used' if ok else 'DROPPED'}",
                    flush=True,
                )
                if not ok:
                    continue
                for q, v in zip(new, views[len(anc) :], strict=True):
                    pose = np.eye(4)
                    pose[:3, :3] = r @ v["pose"][:3, :3]
                    pose[:3, 3] = s * r @ v["pose"][:3, 3] + tr
                    cams.write(
                        json.dumps(
                            dict(walk=name, file=q.name, pose=pose.tolist(), k=v["k"].tolist())
                        )
                        + "\n"
                    )
                cams.flush()
    finally:
        release(slot)


def ts_of(file: str) -> float:
    return float(file.split("-")[0])


def load(path: Path, walk: str, files: set[str] | None = None) -> dict[str, np.ndarray]:
    """Camera centres of one walk by file, in time order (the last row of a file wins)."""
    by = {}
    for ln in path.read_text().splitlines():
        r = json.loads(ln)
        if r["walk"] == walk and (files is None or r["file"] in files):
            by[r["file"]] = np.array(r["pose"])[:3, 3]
    return dict(sorted(by.items(), key=lambda kv: ts_of(kv[0])))


def arkit_residual(walk: str, pos: dict[str, np.ndarray]) -> float | None:
    """Median camera-centre distance (m) between the model and ARKit after a similarity fit."""
    p = WALKS / walk / "poses.jsonl"
    if not p.exists() or len(pos) < 3:
        return None
    ar = {}
    for ln in p.read_text().splitlines():
        r = json.loads(ln)
        if r.get("type") == "arpose":
            ar[round(r["ts"])] = np.array(r["m"]).reshape(4, 4).T[:3, 3]  # column-major
    rows = {f: ar[round(ts)] for f in pos if (ts := ts_from_frames(walk).get(f)) in ar}
    if len(rows) < 3:
        return None
    # ARKit onto the model, residual in the room's metres. The other way round a scrambled
    # trajectory wins by shrinking to a point (scale 0.08 for MapAnything here), which only
    # measures ARKit's own spread (0.5 m on this walk).
    src = np.array(list(rows.values()))
    dst = np.array([pos[f] for f in rows])
    s, r, t = umeyama(src, dst)
    return float(np.median(np.linalg.norm(src * s @ r.T + t - dst, axis=1)))


_ts: dict[str, dict[str, int]] = {}


def ts_from_frames(walk: str) -> dict[str, int]:
    if walk not in _ts:
        rows = (WALKS / walk / "frames.jsonl").read_text().splitlines()
        _ts[walk] = {(r := json.loads(ln))["file"]: round(r["ts"]) for ln in rows if ln}
    return _ts[walk]


def metrics(pos: dict[str, np.ndarray], tried: int, walk: str) -> dict:
    p = np.array(list(pos.values())) if pos else np.zeros((0, 3))
    res = arkit_residual(walk, pos)
    return dict(
        placed=len(p),
        tried=tried,
        jump_share=round(recon.continuity(p), 3),
        trusted_share=round(float(recon.trusted(p).mean()), 3) if len(p) else None,
        arkit_median_m=None if res is None else round(res, 3),
    )


def score(a: argparse.Namespace) -> None:
    out = {}
    for name in a.walk:
        log = OUT / "chunks.jsonl"
        rows = log.read_text().splitlines() if log.exists() else []
        chunks = [r for r in map(json.loads, rows) if r["walk"] == name and r["stride"] == a.stride]
        tried = {q.name for q in walk_frames(WALKS / name, a.stride, set())}
        vg = load(OUT / "cams.jsonl", name) if (OUT / "cams.jsonl").exists() else {}
        ma_same = load(MA_CAMS, name, tried)
        ma_all = load(MA_CAMS, name)
        out[name] = dict(
            vggt=metrics(vg, len(tried), name),
            mapanything_same_frames=metrics(ma_same, len(tried), name),
            mapanything_all=metrics(ma_all, len(ts_from_frames(name)), name),
            vggt_sec_per_frame=round(
                sum(c["sec"] for c in chunks) / max(1, sum(c["views"] for c in chunks)), 3
            ),
            vggt_peak_gib=max((c["gib"] for c in chunks), default=None),
            vggt_chunks_used=f"{sum(c['used'] for c in chunks)}/{len(chunks)}",
        )
        print(name, json.dumps(out[name]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("walk", nargs="+")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--new", type=int, default=NEW)
    ap.add_argument("--anchors", type=int, default=ANCHORS)
    a = ap.parse_args()
    run(a) if a.mode == "run" else score(a)


if __name__ == "__main__":
    main()
