"""The memory loop: MapAnything windows become one metric house map with pinned objects.

Keyframes arrive at about 1 to 2 per second. Each window re-sees the last `overlap` keyframes of
the previous one, so a similarity transform fitted on those shared pixels stitches the windows
into a single map. Everything here works on plain arrays; the models are injected so the same
engine runs inside the server's worker process and in offline scripts.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .align import (
    VoxelSet,
    apply_sim3,
    pose_through_sim3,
    robust_similarity,
)
from .config import Settings
from .geometry import gravity_alignment, map_sector_distances
from .loops import (
    AGREE_SHARE,
    AGREE_TURN_DEG,
    GAP_S,
    MAX_SHIFT_M,
    MAX_TURN_DEG,
    MIN_INLIERS,
    MIN_SHIFT_M,
    NEAR_M,
    SIM_MIN,
    Sim3,
    compose,
    inverse,
    optimize,
    shared_matches,
    turn_deg,
)
from .pins import PinStore
from .providers.finder import LANDMARKS, Finder


@dataclass
class Keyframe:
    seq: int
    ts: float
    bgr: np.ndarray  # already at the models' working resolution
    depth: np.ndarray | None = None  # safety-loop metric depth, float16, same size as bgr
    depth_scale: float = 1.0  # scale that had been applied to `depth`
    gravity: np.ndarray | None = None  # gravity axis in OpenCV camera axes, sign unknown
    taps: list[tuple[str, float, float]] = field(default_factory=list)  # (label, nx, ny)
    hi: np.ndarray | None = None  # the same frame larger, for the item finder


@dataclass
class _Placed:
    kf: Keyframe
    pts: np.ndarray  # (H, W, 3) global
    mask: np.ndarray  # (H, W) bool
    pose: np.ndarray  # 4x4 global


STITCH_OK = 0.05  # stitch cost (share of scene distance) of a join trusted fully
STITCH_FLOOR = 0.05  # trust in a join with no overlap at all (the transform was carried over)
TEXTURE_MIN = 25  # FAST corners at 320 px wide; a blank wall or door up close has 2 to 22
STRIDE = 2  # pixel step of the geometry each keyframe keeps for re-placing after a loop


@dataclass
class _Stored:
    """What a keyframe keeps so the map can be rebuilt when its window moves."""

    seq: int
    ts: float
    w: int  # window index; the window's similarity transform places everything below
    local: np.ndarray  # (h, w, 3) float16, the window's own frame, every STRIDE-th pixel
    mask: np.ndarray  # (h, w) bool
    cols: np.ndarray  # (h, w, 3) uint8 RGB
    pose: np.ndarray  # 4x4 in the window's frame
    jpg: bytes
    desc: np.ndarray | None


@dataclass
class _Obs:
    w: int
    local: np.ndarray  # position in the window's frame
    label: str
    kind: str
    ts: float
    score: float
    thumb: str
    source: str


_fast = cv2.FastFeatureDetector_create(25)


def texture(bgr: np.ndarray) -> int:
    """Corner count: how much a frame gives the map to hold on to."""
    h = round(bgr.shape[0] * 320 / bgr.shape[1])
    return len(_fast.detect(cv2.cvtColor(cv2.resize(bgr, (320, h)), cv2.COLOR_BGR2GRAY)))


def _thumb(bgr: np.ndarray, box: tuple[float, float, float, float], size: int = 96) -> str:
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box
    pw, ph = 0.3 * (x1 - x0) + 12, 0.3 * (y1 - y0) + 12
    crop = bgr[
        max(0, int(y0 - ph)) : min(h, int(y1 + ph)), max(0, int(x0 - pw)) : min(w, int(x1 + pw))
    ]
    if crop.size == 0:
        return ""
    s = size / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
    ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(jpg.tobytes()).decode() if ok else ""


class MapAnythingRunner:
    """Thin wrapper over the model so the engine can be tested with a fake."""

    def __init__(self, device: str = "mps"):
        import torch
        from mapanything.models import MapAnything
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

        self._torch = torch
        self.device = torch.device(device)
        self.model = MapAnything.from_pretrained("facebook/map-anything-apache").to(self.device)
        self.model.eval()
        norm = IMAGE_NORMALIZATION_DICT["dinov2"]
        self._mean = np.array(norm.mean, np.float32)
        self._std = np.array(norm.std, np.float32)

    def infer(self, frames_bgr: list[np.ndarray]) -> list[dict]:
        """Per frame: pts (H,W,3), mask (H,W), pose 4x4, depth_z (H,W), rgb (H,W,3) 0..1."""
        torch = self._torch
        views = []
        for i, bgr in enumerate(frames_bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            x = torch.from_numpy((rgb - self._mean) / self._std).permute(2, 0, 1)[None]
            views.append(
                dict(
                    img=x,
                    true_shape=np.int32([bgr.shape[:2]]),
                    idx=i,
                    instance=str(i),
                    data_norm_type=["dinov2"],
                )
            )
        with torch.no_grad():
            preds = self.model.infer(
                views,
                memory_efficient_inference=False,
                use_amp=True,
                amp_dtype="fp16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=True,
                confidence_percentile=20,
            )
        torch.mps.synchronize()
        out = []
        for p, bgr in zip(preds, frames_bgr, strict=True):
            out.append(
                dict(
                    pts=p["pts3d"][0].float().cpu().numpy(),
                    mask=p["mask"][0].squeeze(-1).cpu().numpy().astype(bool),
                    pose=p["camera_poses"][0].float().cpu().numpy(),
                    depth_z=p["depth_z"][0].squeeze(-1).float().cpu().numpy(),
                    rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255,
                )
            )
        torch.mps.empty_cache()
        return out


class MemoryEngine:
    def __init__(
        self,
        s: Settings,
        runner,
        finder: Finder,
        landmarks=None,
        sam_every: int = 1,
        describer=None,
    ):
        self.base, self.mode = s, "home"
        self.s, self.runner, self.finder, self.landmarks = s, runner, finder, landmarks
        self.sam_every, self.describer = sam_every, describer
        self._kfs: dict[int, _Stored] = {}
        self._sims: list[Sim3] = []
        self._seq_rel: list[Sim3 | None] = []  # each window's measured step from the one before
        self._seq_w: list[float] = []  # and how far to trust it
        self._last_cost: float | None = None
        self._loop_edges: list[tuple[int, int, Sim3]] = []
        self._obs: list[_Obs] = []
        self.loops: list[dict] = []  # every loop check that got as far as a match
        self.pins = PinStore()
        self.voxels = VoxelSet(s.voxel_m)
        self._tail: list[_Placed] = []
        self._align_rot: np.ndarray | None = None
        self._map_pts = np.empty((0, 3), np.float32)
        self._map_cols = np.empty((0, 3), np.uint8)
        self.trail: list[list[float]] = []
        self.last_pose: np.ndarray | None = None
        self.poses: dict[int, np.ndarray] = {}  # keyframe seq -> 4x4 map pose
        self._windows = 0
        self._prev_sim: tuple[float, np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ modes
    def set_mode(self, mode: str) -> None:
        """home keeps the whole house; outside keeps a short horizon. Starts a fresh map."""
        self.mode = mode
        self.s = replace(self.base, **self.base.outside) if mode == "outside" else self.base
        self.pins = PinStore()
        self.voxels = VoxelSet(self.s.voxel_m)
        self._tail, self._prev_sim, self._windows = [], None, 0
        self._map_pts = np.empty((0, 3), np.float32)
        self._map_cols = np.empty((0, 3), np.uint8)
        self.trail, self.last_pose, self.poses = [], None, {}
        self._kfs, self._sims, self._obs, self.loops = {}, [], [], []
        self._seq_rel, self._seq_w, self._loop_edges = [], [], []

    def _trim_to_horizon(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Outside only: forget map points farther than horizon_m from the newest pose."""
        if self.mode != "outside" or self.last_pose is None:
            return None
        d = np.linalg.norm(self._map_pts[:, [0, 2]] - self.last_pose[[0, 2], 3], axis=1)
        keep = d < self.s.horizon_m
        if keep.all():
            return None
        self._map_pts, self._map_cols = self._map_pts[keep], self._map_cols[keep]
        self.voxels = VoxelSet(self.s.voxel_m)
        self.voxels.add_new(self._map_pts)
        self.trail = self.trail[-60:]
        return self._map_pts, self._map_cols

    # ------------------------------------------------------------------ helpers
    def _gravity_rot(self, kf: Keyframe) -> np.ndarray:
        """Rotation world(first camera) -> map (y up). Level-camera default if no usable gravity."""
        default = np.diag([1.0, -1.0, -1.0])
        if kf.gravity is None:
            return default
        rot = gravity_alignment(kf.gravity)
        pitch = np.degrees(np.arcsin(np.clip((rot @ np.array([0.0, 0.0, 1.0]))[1], -1, 1)))
        return rot if abs(pitch) < 65 else default

    def _fit(self, new: list[dict], old: list[_Placed], first: Keyframe):
        """Similarity taking this window's local frame into the global map frame."""
        self._last_cost = None
        if not old:
            if self._prev_sim is not None:  # no overlap survived: carry the last transform
                return self._prev_sim
            rot = self._gravity_rot(first)
            return 1.0, rot, np.zeros(3)
        src, dst, cl, cg = [], [], [], []
        rng = np.random.default_rng(0)
        for j, placed in enumerate(old):
            n = new[j]  # the first len(old) frames of the window are the overlap
            ok = n["mask"] & placed.mask
            a, b = n["pts"][ok], placed.pts[ok]
            if len(a) > 4000:
                pick = rng.choice(len(a), 4000, replace=False)
                a, b = a[pick], b[pick]
            src.append(a)
            dst.append(b)
            cl.append(np.broadcast_to(n["pose"][:3, 3], a.shape))
            cg.append(np.broadcast_to(placed.pose[:3, 3], b.shape))
        src, dst = np.concatenate(src), np.concatenate(dst)
        cl, cg = np.concatenate(cl), np.concatenate(cg)
        cands = []
        fit = robust_similarity(src, dst)
        if fit is not None:
            cands.append(fit[:3])
        # From the overlap frames' own camera poses, which a flat wall cannot slide; the scale
        # is the ratio of the distances from each camera to the same pixels.
        rots = [p.pose[:3, :3] @ new[j]["pose"][:3, :3].T for j, p in enumerate(old)]
        u, _, vt = np.linalg.svd(np.sum(rots, axis=0))
        rot = u @ vt
        if np.linalg.det(rot) < 0:
            rot = u @ np.diag([1.0, 1.0, -1.0]) @ vt
        dl, dg = np.linalg.norm(src - cl, axis=1), np.linalg.norm(dst - cg, axis=1)
        good = dl > 1e-3
        sc = float(np.clip(np.median(dg[good] / dl[good]), 0.5, 2.0)) if good.any() else 1.0
        t = np.mean(
            [p.pose[:3, 3] - sc * rot @ new[j]["pose"][:3, 3] for j, p in enumerate(old)], 0
        )
        cands.append((sc, rot, t))
        costs = [self._stitch_cost(c, src, dst, new, old) for c in cands]
        self._last_cost = min(costs)
        return cands[int(np.argmin(costs))]

    @staticmethod
    def _stitch_cost(c, src, dst, new, old) -> float:
        """How badly a candidate stitch disagrees, as a share of the distance to the scene:
        median point gap plus the worst gap between where each overlap camera was and lands."""
        s_, rot, t = c
        depth = max(float(np.median(np.linalg.norm(dst - old[0].pose[:3, 3], axis=1))), 0.3)
        res = float(np.median(np.linalg.norm(apply_sim3(src, s_, rot, t) - dst, axis=1)))
        gap = max(
            float(np.linalg.norm(s_ * rot @ new[j]["pose"][:3, 3] + t - p.pose[:3, 3]))
            for j, p in enumerate(old)
        )
        return (res + gap) / depth

    def _locate(self, placed: _Placed, box: tuple[float, float, float, float]):
        """Median map position of the central half of a box (pixels of the keyframe)."""
        x0, y0, x1, y1 = box
        qx, qy = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
        rows = slice(max(0, int(y0 + qy)), int(y1 - qy) + 1)
        cols = slice(max(0, int(x0 + qx)), int(x1 - qx) + 1)
        pts = placed.pts[rows, cols][placed.mask[rows, cols]]
        return np.median(pts, axis=0) if len(pts) >= 6 else None

    def _observe(self, w, pos, label, kind, ts, score, thumb="", source="") -> None:
        """Pin a sighting, and keep it in its window's frame so a loop can move it later."""
        sc, rot, t = self._sims[w]
        local = rot.T @ (np.asarray(pos, float) - t) / sc
        self._obs.append(_Obs(w, local, label, kind, ts, score, thumb, source))
        pos_l = [float(v) for v in pos]
        near = self._near_landmarks(pos_l) if kind == "object" else None
        self.pins.observe(label, kind, pos_l, ts, score, near, thumb, source)

    def _pin_detections(self, placed: _Placed, run_finder: bool) -> int:
        kf = placed.kf
        ts = kf.ts
        w = self._kfs[kf.seq].w if kf.seq in self._kfs else len(self._sims) - 1
        count = 0
        if self.landmarks is not None:
            r = self.landmarks.predict(kf.bgr, imgsz=320, device="mps", verbose=False, conf=0.4)[0]
            for b, c, p in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                strict=True,
            ):
                name = LANDMARKS.get(r.names[int(c)])
                pos = self._locate(placed, tuple(b)) if name else None
                if pos is not None:
                    self._observe(w, pos, name, "landmark", ts, float(p), source="yolo26n")
        if run_finder:
            src = kf.hi if kf.hi is not None else kf.bgr
            sx, sy = kf.bgr.shape[1] / src.shape[1], kf.bgr.shape[0] / src.shape[0]
            for det in self.finder.find(src, self.s.watchlist):
                x0, y0, x1, y1 = det.box
                det = replace(det, box=(x0 * sx, y0 * sy, x1 * sx, y1 * sy))
                pos = self._locate(placed, det.box)
                if pos is None:
                    continue
                self._observe(
                    w,
                    pos,
                    det.label,
                    "object",
                    ts,
                    det.score,
                    _thumb(kf.bgr, det.box),
                    det.source or self.finder.name,
                )
                count += 1
        for label, nx, ny in kf.taps:
            h, wd = kf.bgr.shape[:2]
            half = 0.04
            box = ((nx - half) * wd, (ny - half) * h, (nx + half) * wd, (ny + half) * h)
            pos = self._locate(placed, box)
            if pos is not None:
                self._observe(w, pos, label, "object", ts, 1.0, _thumb(kf.bgr, box), "tap")
                count += 1
        return count

    def _near_landmarks(self, pos: list[float], radius: float = 1.6) -> list[str]:
        cands = []
        for p in self.pins.pins.values():
            if p.kind == "landmark":
                d = float(np.linalg.norm(np.array(p.pos) - np.array(pos)))
                if d <= radius:
                    cands.append((d, p.label))
        return [lb for _, lb in sorted(cands)[:2]]

    # ------------------------------------------------------------------ loops
    def _store(self, kf: Keyframe, pr: dict, w: int) -> None:
        ok, jpg = cv2.imencode(".jpg", kf.bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sl = (slice(None, None, STRIDE), slice(None, None, STRIDE))
        self._kfs[kf.seq] = _Stored(
            kf.seq,
            kf.ts,
            w,
            pr["pts"][sl].astype(np.float16),
            pr["mask"][sl],
            (pr["rgb"][sl] * 255).astype(np.uint8),
            pr["pose"],
            jpg.tobytes(),
            self.describer(kf.bgr) if self.describer is not None else None,
        )

    def _global(self, st: _Stored) -> np.ndarray:
        return apply_sim3(st.local.astype(np.float32).reshape(-1, 3), *self._sims[st.w]).reshape(
            st.local.shape
        )

    def _loop_proposal(self, newest: _Placed):
        """(old keyframe, this keyframe, correction, record) when `newest` sees an old place."""
        st = self._kfs.get(newest.kf.seq)
        if st is None or st.desc is None:
            return None
        here = newest.pose[:3, 3]
        cands = []
        for o in self._kfs.values():
            if o.w > st.w - 2 or st.ts - o.ts < GAP_S:
                continue
            sim = float(o.desc @ st.desc)
            o_pos = pose_through_sim3(o.pose, *self._sims[o.w])[:3, 3]
            if sim >= SIM_MIN and np.linalg.norm(o_pos - here) <= NEAR_M:
                cands.append((sim, o))
        if not cands:
            return None
        best, inliers = None, 0
        for _, o in sorted(cands, key=lambda c: -c[0])[:3]:
            old_bgr = cv2.imdecode(np.frombuffer(o.jpg, np.uint8), cv2.IMREAD_COLOR)
            n = shared_matches(old_bgr, newest.kf.bgr)
            if n > inliers:
                best, inliers, best_bgr = o, n, old_bgr
        rec = {"seq": st.seq, "to": best.seq if best else None, "inliers": inliers}
        self.loops.append(rec)
        if best is None or inliers < MIN_INLIERS:
            rec["result"] = "few matches"
            return None

        # MapAnything on the old and new frame together: the new frame's place next to the old.
        pr = self.runner.infer([best_bgr, newest.kf.bgr])
        sl = (slice(None, None, STRIDE), slice(None, None, STRIDE))
        m_old = pr[0]["mask"][sl] & best.mask
        a = robust_similarity(pr[0]["pts"][sl][m_old], self._global(best)[m_old])
        if a is None or a[3] > 0.05:
            rec["result"] = "old frame did not fit"
            return None
        m_new = pr[1]["mask"][sl] & st.mask
        target = apply_sim3(pr[1]["pts"][sl][m_new], *a[:3])
        c = robust_similarity(self._global(st)[m_new], target)
        if c is None:
            rec["result"] = "no fit"
            return None
        c = c[:3]
        shift = float(np.linalg.norm(c[0] * c[1] @ here + c[2] - here))
        turn = turn_deg(c[1])
        rec.update(shift_m=round(shift, 3), turn_deg=round(turn, 1), scale=round(c[0], 3))
        rec["result"] = "proposed"
        return best, st, c, rec, here

    def _close_loops(self, placed_new: list[_Placed]) -> bool:
        """Small corrections close on one keyframe's word; a large one (real walks drift by
        metres and tens of degrees) needs two keyframes of the window that agree on it."""
        props = [p for p in (self._loop_proposal(x) for x in reversed(placed_new)) if p]
        if not props:
            return False

        def size(p):
            return p[3]["shift_m"], p[3]["turn_deg"]

        def lands(p):
            c, here = p[2], placed_new[-1].pose[:3, 3]
            return c[0] * c[1] @ here + c[2], c[1]

        small = [
            p
            for p in props
            if MIN_SHIFT_M <= size(p)[0] <= MAX_SHIFT_M and size(p)[1] <= MAX_TURN_DEG
        ]
        big = [p for p in props if size(p)[0] > MAX_SHIFT_M or size(p)[1] > MAX_TURN_DEG]
        chosen = max(small, key=lambda p: p[3]["inliers"]) if small else None
        for p in sorted(big, key=lambda p: -p[3]["inliers"]):
            if chosen is not None:
                break
            pos, rot = lands(p)
            for q in big:
                q_pos, q_rot = lands(q)
                if (
                    q is not p
                    and np.linalg.norm(pos - q_pos) <= max(0.5, AGREE_SHARE * size(p)[0])
                    and turn_deg(rot.T @ q_rot) <= AGREE_TURN_DEG
                ):
                    chosen = p
                    break
        for p in props:
            p[3]["result"] = "closed" if p is chosen else "not used"
        if chosen is None:
            return False
        best, st, c, _, _ = chosen
        # The new window placed as the loop says, measured relative to the old one.
        self._loop_edges.append(
            (best.w, st.w, compose(inverse(self._sims[best.w]), compose(c, self._sims[st.w])))
        )
        before = list(self._sims)
        self._sims = optimize(self._sims, self._seq_rel, self._loop_edges, self._seq_w)
        self._prev_sim = self._sims[-1]
        for p in self._tail + [x for x in placed_new if x not in self._tail]:
            w = self._kfs[p.kf.seq].w if p.kf.seq in self._kfs else st.w
            cw = compose(self._sims[w], inverse(before[w]))
            p.pts = apply_sim3(p.pts.reshape(-1, 3), *cw).reshape(p.pts.shape)
            p.pose = pose_through_sim3(p.pose, *cw)
        return True

    def _rebuilt(self) -> list[dict]:
        """Map, trail and pins again from every keyframe's own geometry, after a loop."""
        self.voxels = VoxelSet(self.s.voxel_m)
        pts, cols = [], []
        self.trail, self.poses = [], {}
        for seq in sorted(self._kfs):
            st = self._kfs[seq]
            g = self._global(st)[st.mask]
            fresh = self.voxels.add_new(g)
            pts.append(g[fresh].astype(np.float32))
            cols.append(st.cols[st.mask][fresh])
            pose = pose_through_sim3(st.pose, *self._sims[st.w])
            self.poses[seq] = pose
            self.trail.append([*map(float, pose[:3, 3]), st.ts])
        self._map_pts, self._map_cols = np.concatenate(pts), np.concatenate(cols)
        self.last_pose = self.poses[max(self.poses)]
        self.pins = PinStore()
        for o in self._obs:
            pos = apply_sim3(o.local[None], *self._sims[o.w])[0]
            pos_l = [float(v) for v in pos]
            near = self._near_landmarks(pos_l) if o.kind == "object" else None
            self.pins.observe(o.label, o.kind, pos_l, o.ts, o.score, near, o.thumb, o.source)
        return [
            {
                "type": "map_replace",
                "n": len(self._map_pts),
                "points": self._map_pts.tobytes(),
                "colors": self._map_cols.tobytes(),
            },
            {"type": "trail_replace", "points": self.trail},
            {"type": "pins", "pins": [p.to_json() for p in self.pins.pins.values()]},
        ]

    # ------------------------------------------------------------------ main entry
    def process_window(self, frames: list[Keyframe]) -> list[dict]:
        """Run MapAnything on [overlap tail] + frames; return events for the clients."""
        t0 = time.perf_counter()
        # Blank walls and doors up close give MapAnything nothing to place, and a window
        # that stitches onto one slides off; leave them out.
        frames = [f for f in frames if texture(f.bgr) >= TEXTURE_MIN]
        if not frames:
            return []
        old = self._tail[-self.s.overlap :] if self._tail else []
        batch = [p.kf for p in old] + frames
        preds = self.runner.infer([k.bgr for k in batch])
        t_infer = time.perf_counter()
        sim = self._fit(preds, old, frames[0])
        self._prev_sim = sim
        s_, rot, trans = sim

        placed_new: list[_Placed] = []
        for j, (kf, pr) in enumerate(zip(batch, preds, strict=True)):
            if j < len(old):
                continue
            placed_new.append(
                _Placed(
                    kf,
                    apply_sim3(pr["pts"].reshape(-1, 3), s_, rot, trans).reshape(pr["pts"].shape),
                    pr["mask"],
                    pose_through_sim3(pr["pose"], s_, rot, trans),
                )
            )
        if self._sims:
            self._seq_rel.append(compose(inverse(self._sims[-1]), sim))
            # A join that fit badly is where drift came from, so a loop bends it the most.
            cost = self._last_cost
            self._seq_w.append(STITCH_FLOOR if cost is None else min(1.0, STITCH_OK / cost))
        else:
            self._seq_rel.append(None)
            self._seq_w.append(0.0)
        self._sims.append(sim)
        if self.mode == "home":
            for p, pr in zip(placed_new, preds[len(old) :], strict=True):
                self._store(p.kf, pr, len(self._sims) - 1)

        # Points -> map, only voxels not yet occupied.
        pts = np.concatenate([p.pts[p.mask] for p in placed_new])
        cols = np.concatenate(
            [
                (pr["rgb"][pr["mask"]] * 255).astype(np.uint8)
                for pr, k in zip(preds[len(old) :], frames, strict=True)
            ]
        )
        events: list[dict] = []
        if len(pts):
            fresh = self.voxels.add_new(pts)
            new_pts, new_cols = pts[fresh].astype(np.float32), cols[fresh]
            self._map_pts = np.concatenate([self._map_pts, new_pts])
            self._map_cols = np.concatenate([self._map_cols, new_cols])
            events.append(
                {
                    "type": "map_chunk",
                    "n": len(new_pts),
                    "points": new_pts.tobytes(),
                    "colors": new_cols.tobytes(),
                    "total": len(self._map_pts),
                }
            )

        # Camera trail and pose.
        for p in placed_new:
            pos = p.pose[:3, 3]
            self.trail.append([float(pos[0]), float(pos[1]), float(pos[2]), p.kf.ts])
            self.poses[p.kf.seq] = p.pose
        self.last_pose = placed_new[-1].pose
        events.append({"type": "trail", "points": self.trail[-len(placed_new) :]})

        # Objects and landmarks.
        found = 0
        for p in placed_new:
            found += self._pin_detections(
                p, run_finder=self.mode == "home" and p.kf.seq % self.sam_every == 0
            )
        events.append({"type": "pins", "pins": [p.to_json() for p in self.pins.pins.values()]})

        # A place seen before: bend the walk since then back onto it.
        if self._close_loops(placed_new):
            events += self._rebuilt()

        # Scale hint for the fast depth: map depth vs. the depth the safety loop produced.
        ratios = []
        for pr, kf in zip(preds[len(old) :], frames, strict=True):
            if kf.depth is None or pr["depth_z"].shape != kf.depth.shape:
                continue
            raw = kf.depth.astype(np.float32) / max(kf.depth_scale, 1e-3)
            ok = pr["mask"] & (raw > 0.2) & (pr["depth_z"] > 0.2)
            if ok.sum() > 2000:
                ratios.append(float(np.median(pr["depth_z"][ok] * s_ / raw[ok])))
        if ratios:
            events.append({"type": "scale_hint", "ratio": float(np.median(ratios))})

        trimmed = self._trim_to_horizon()
        if trimmed is not None:
            events.append(
                {
                    "type": "map_replace",
                    "n": len(trimmed[0]),
                    "points": trimmed[0].tobytes(),
                    "colors": trimmed[1].tobytes(),
                }
            )

        # Side and rear sectors from the map around the newest pose.
        near = self._map_pts[
            np.linalg.norm(self._map_pts[:, [0, 2]] - self.last_pose[[0, 2], 3], axis=1) < 4.0
        ]
        events.append(
            {
                "type": "pose",
                "pos": self.last_pose[:3, 3].tolist(),
                "fwd": self.last_pose[:3, 2].tolist(),
                "up": (-self.last_pose[:3, 1]).tolist(),
                "ts": frames[-1].ts,
                "map_sectors": [
                    None if not np.isfinite(d) else d
                    for d in map_sector_distances(near, self.last_pose)
                ],
            }
        )

        # The next window stitches onto the most textured of the latest few frames.
        recent = (self._tail + placed_new)[-2 * self.s.overlap :]
        keep = sorted(recent, key=lambda p: -texture(p.kf.bgr))[: self.s.overlap]
        self._tail = sorted(keep, key=lambda p: p.kf.seq)
        self._windows += 1
        events.append(
            {
                "type": "memory_stats",
                "window": self._windows,
                "views": len(batch),
                "infer_ms": 1000 * (t_infer - t0),
                "total_ms": 1000 * (time.perf_counter() - t0),
                "map_points": len(self._map_pts),
                "pins_found": found,
                "stitch_scale": float(s_),
            }
        )
        return events

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        np.savez_compressed(
            path,
            pts=self._map_pts,
            cols=self._map_cols,
            trail=np.array(self.trail, np.float64).reshape(-1, 4),
            pose=self.last_pose if self.last_pose is not None else np.eye(4),
        )
