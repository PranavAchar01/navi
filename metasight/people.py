"""People in view as anonymous tracks, and which of them is speaking.

Nobody is identified. A track is a small integer that lives as long as the person stays in view
and is forgotten with them: no face recognition, no names, no embeddings, and no image kept
beyond the previous 24 x 24 grey patch of the head, which dies with the track.

Tracking is ByteTrack's recipe on the person boxes the safety loop already makes: a constant
velocity prediction per track, a first association of confident boxes by IoU (Hungarian), a
second pass that lets weak boxes keep a track alive instead of starting a new one, and a short
memory (`MAX_LOST_S`) for people who walk behind something. One addition: matches whose measured
distances disagree by more than `DEPTH_RATIO` are refused, which stops two people who cross at
different depths from swapping.

Speaking is a fusion of three weak signals, each of which may be missing:

* VAD: somebody is talking (audio_doa.Vad). Without it nobody is marked as speaking.
* DoA: the bearing of the voice from two microphones (audio_doa.Doa), compared with each
  track's bearing from the camera.
* Lip activity: motion in the lower half of the head crop beyond motion in the upper half. This
  is a cheap frame-difference stand-in for a mouth landmark model, readable only when the head is
  at least `MIN_HEAD_PX` wide. See docs/people.md for how well each part works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HIGH = 0.5  # boxes at or above this start tracks and match first
LOW = 0.1  # boxes between LOW and HIGH only keep an existing track alive
MATCH_IOU = 0.2
MATCH_IOU_LOW = 0.4
CONFIRM_HITS = 3
MAX_LOST_S = 1.5
SHOW_LOST_S = 0.5  # a track missed for less than this is still reported (flagged lost)
DEPTH_RATIO = 1.8
DEPTH_WEIGHT = 0.8
BIG = 10.0
MIN_HEAD_PX = 24
PATCH = 24
LIP_FULL = 6.0  # lip activity (grey levels of frame difference) read as "clearly moving"
SPEAK_MIN = 0.35
DOA_SIGMA_DEG = 18.0
STICKY_S = 1.0  # the speaker is held this long after the evidence thins


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Det:
    box: tuple[float, float, float, float]  # normalised x0, y0, x1, y1
    score: float
    dist_m: float = math.inf


@dataclass
class Track:
    id: int
    box: np.ndarray  # x0, y0, x1, y1
    vel: np.ndarray  # box velocity per second, same layout
    dist_m: float
    t: float  # last time matched
    born: float
    hits: int = 1
    lost: bool = False
    lip: float | None = None  # smoothed lip activity, None when the head is too small to read
    patch: np.ndarray | None = field(default=None, repr=False)
    speak: float = 0.0

    def predict(self, t: float) -> np.ndarray:
        return self.box + self.vel * (t - self.t)


class PeopleTracker:
    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.next_id = 1

    def reset(self) -> None:
        self.tracks, self.next_id = [], 1

    def _match(self, tracks: list[Track], dets: list[Det], t: float, thresh: float):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost = np.full((len(tracks), len(dets)), BIG)
        for i, tr in enumerate(tracks):
            p = tr.predict(t)
            for j, d in enumerate(dets):
                o = iou(p, d.box)
                extra = 0.0
                if math.isfinite(tr.dist_m) and math.isfinite(d.dist_m):
                    r = max(tr.dist_m, d.dist_m) / max(1e-3, min(tr.dist_m, d.dist_m))
                    if r > DEPTH_RATIO:
                        continue
                    extra = DEPTH_WEIGHT * math.log(r)  # same box overlap: nearer depth wins
                if o >= thresh:
                    cost[i, j] = 1.0 - o + extra
        rows, cols = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(rows, cols, strict=True) if cost[i, j] < BIG]
        mi, mj = {i for i, _ in pairs}, {j for _, j in pairs}
        return (
            pairs,
            [i for i in range(len(tracks)) if i not in mi],
            [j for j in range(len(dets)) if j not in mj],
        )

    def _apply(self, tr: Track, d: Det, t: float) -> None:
        dt = max(1e-3, t - tr.t)
        new = np.array(d.box, float)
        tr.vel = 0.6 * tr.vel + 0.4 * (new - tr.box) / dt
        tr.box, tr.t, tr.hits, tr.lost = new, t, tr.hits + 1, False
        if math.isfinite(d.dist_m):
            tr.dist_m = (
                d.dist_m if not math.isfinite(tr.dist_m) else 0.6 * tr.dist_m + 0.4 * d.dist_m
            )

    def update(self, dets: list[Det], t: float) -> list[Track]:
        """Advance to time `t` with this frame's person boxes. Returns the tracks to report."""
        high = [d for d in dets if d.score >= HIGH]
        low = [d for d in dets if LOW <= d.score < HIGH]
        live = list(self.tracks)
        pairs, rest_t, rest_h = self._match(live, high, t, MATCH_IOU)
        for i, j in pairs:
            self._apply(live[i], high[j], t)
        left = [live[i] for i in rest_t]
        pairs2, rest_t2, _ = self._match(left, low, t, MATCH_IOU_LOW)
        for i, j in pairs2:
            self._apply(left[i], low[j], t)
        for i in rest_t2:
            left[i].lost = True
        for j in rest_h:
            d = high[j]
            self.tracks.append(
                Track(self.next_id, np.array(d.box, float), np.zeros(4), d.dist_m, t, t)
            )
            self.next_id += 1
        self.tracks = [tr for tr in self.tracks if t - tr.t <= MAX_LOST_S]
        return [tr for tr in self.tracks if tr.hits >= CONFIRM_HITS and t - tr.t <= SHOW_LOST_S]


def head_patch(gray: np.ndarray, box) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(lower half, upper half) of a 24 x 24 patch of the head, or None when too small.

    The head is taken as the top fifth of the person box, the middle 60% of its width. The lower
    half of that patch holds the mouth, the upper half the eyes and forehead."""
    h, w = gray.shape
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * w, (y1 - y0) * h
    hx0 = (x0 + 0.2 * (x1 - x0)) * w
    hx1 = (x1 - 0.2 * (x1 - x0)) * w
    hy0, hy1 = y0 * h, (y0 + 0.2 * (y1 - y0)) * h
    if hx1 - hx0 < MIN_HEAD_PX or bw <= 0 or bh <= 0:
        return None, None
    crop = gray[max(0, int(hy0)) : int(hy1) + 1, max(0, int(hx0)) : int(hx1) + 1]
    if crop.size == 0 or min(crop.shape) < 8:
        return None, None
    p = cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA).astype(np.float32)
    p -= p.mean()
    return p[PATCH // 2 :], p[: PATCH // 2]


def lip_activity(prev: np.ndarray, cur: np.ndarray) -> float:
    """Mouth-half change beyond eye-half change between two patches (see `head_patch`).

    Head turns and camera shake move both halves; talking moves mostly the lower one."""
    lo = float(np.mean(np.abs(cur[0] - prev[0])))
    up = float(np.mean(np.abs(cur[1] - prev[1])))
    return max(0.0, lo - 0.7 * up)


def pick_speaker(
    voice: bool,
    cands: list[tuple[int, float, float | None, float]],
    doa_deg: float | None,
) -> tuple[dict[int, float], str]:
    """Speaking score per track id and a name for the evidence used.

    `cands` is (id, bearing_deg, lip activity or None, distance). Scores are 0..1 and are not
    probabilities: they rank the people in view, and `SPEAK_MIN` is where one is called the
    speaker. Nothing is scored while the VAD hears no speech."""
    if not voice or not cands:
        return {c[0]: 0.0 for c in cands}, "none"
    scores: dict[int, float] = {}
    for tid, bearing, lip, _ in cands:
        lipn = None if lip is None else min(1.0, lip / LIP_FULL)
        doan = None if doa_deg is None else math.exp(-(((bearing - doa_deg) / DOA_SIGMA_DEG) ** 2))
        parts = [x for x in (lipn, doan) if x is not None]
        scores[tid] = sum(parts) / len(parts) if parts else 0.0
    ev = "+".join(
        n
        for n, v in (("lip", any(c[2] is not None for c in cands)), ("doa", doa_deg is not None))
        if v
    )
    if not ev:
        # Voice but nothing to tell people apart: only a person alone in view is called.
        if len(cands) == 1:
            scores[cands[0][0]] = 0.5
            return scores, "alone"
        return scores, "none"
    if len(cands) == 1 and max(scores.values()) < SPEAK_MIN and doa_deg is None:
        scores[cands[0][0]] = max(scores[cands[0][0]], 0.4)  # voice, one person, mouth unreadable
        ev = "alone+" + ev
    return scores, ev


def bearing_cue(bearing_deg: float, face_within_deg: float = 15.0) -> dict | None:
    """What the wearer should feel to turn toward `bearing_deg` (positive right), or None when
    already facing it. Motors come from pan.py so the belt and the beep agree on direction."""
    from .pan import pan_gains

    if abs(bearing_deg) <= face_within_deg:
        return None
    a, ga, b, gb = pan_gains(bearing_deg)
    return {
        "bearing": round(bearing_deg, 1),
        "turn": "right" if bearing_deg > 0 else "left",
        "motors": [[a, round(ga, 2)], [b, round(gb, 2)]],
    }


class Conversation:
    """Per-frame state: tracks, the speaker with hold, and the cue to face them."""

    def __init__(self, hfov_deg: float = 105.0) -> None:
        self.tracker = PeopleTracker()
        self.hfov_deg = hfov_deg
        self.speaker: int | None = None
        self.speaker_t = -math.inf
        self.cue_t = -math.inf
        self.cue_id: int | None = None
        self.offscreen_t = -math.inf

    def reset(self) -> None:
        self.tracker.reset()
        self.speaker = None

    def bearing(self, box) -> float:
        cx = 0.5 * (box[0] + box[2])
        return math.degrees(math.atan((2 * cx - 1) * math.tan(math.radians(self.hfov_deg) / 2)))

    def step(
        self,
        t: float,
        dets: list[Det],
        gray: np.ndarray | None,
        voice: bool,
        doa_deg: float | None,
        head_xz: tuple[float, float, float] | None = None,
    ) -> dict:
        """One frame. `head_xz` = (x, z, yaw radians) of the wearer on the map, if known."""
        tracks = self.tracker.update(dets, t)
        for tr in tracks:
            if tr.lost or gray is None:
                continue
            lo, up = head_patch(gray, tr.box)
            if lo is None:
                tr.lip, tr.patch = None, None
                continue
            cur = (lo, up)
            if tr.patch is not None:
                a = lip_activity(tr.patch, cur)
                tr.lip = a if tr.lip is None else 0.6 * tr.lip + 0.4 * a
            tr.patch = cur
        cands = [(tr.id, self.bearing(tr.box), tr.lip, tr.dist_m) for tr in tracks if not tr.lost]
        scores, evidence = pick_speaker(voice, cands, doa_deg)
        best = max(scores, key=scores.get) if scores else None
        if best is not None and scores[best] >= SPEAK_MIN:
            self.speaker, self.speaker_t = best, t
        elif self.speaker is not None and (
            t - self.speaker_t > STICKY_S or self.speaker not in {c[0] for c in cands}
        ):
            self.speaker = None
        out = []
        for tr in tracks:
            tr.speak = scores.get(tr.id, 0.0)
            b = self.bearing(tr.box)
            item = {
                "id": tr.id,
                "box": [round(float(v), 4) for v in tr.box],
                "bearing": round(b, 1),
                "dist": round(tr.dist_m, 2) if math.isfinite(tr.dist_m) else None,
                "speaking": round(tr.speak, 2),
                "lip": None if tr.lip is None else round(tr.lip, 2),
                "lost": tr.lost,
                "age": round(t - tr.born, 1),
                "map": None,
            }
            if head_xz and math.isfinite(tr.dist_m):
                x, z, yaw = head_xz
                ang = yaw + math.radians(b)
                item["map"] = [
                    round(x + tr.dist_m * math.sin(ang), 2),
                    round(z + tr.dist_m * math.cos(ang), 2),
                ]
            out.append(item)
        ev = {
            "type": "people",
            "t": round(t, 2),
            "tracks": out,
            "speaker": self.speaker,
            "voice": voice,
            "doa": None if doa_deg is None else round(doa_deg, 1),
            "evidence": evidence,
            "cue": None,
        }
        target = None
        if self.speaker is not None:
            tr = next((x for x in tracks if x.id == self.speaker), None)
            if tr is not None:
                target = (self.speaker, self.bearing(tr.box))
        elif voice and doa_deg is not None and not cands_near(cands, doa_deg):
            target = (None, doa_deg)  # a voice from where nobody is in view
            ev["offscreen"] = round(doa_deg, 1)
        if target is not None and t - self.cue_t >= 1.5:
            cue = bearing_cue(target[1])
            if cue is not None:
                cue["id"] = target[0]
                ev["cue"] = cue
                self.cue_t = t
        return ev


def cands_near(cands, doa_deg: float, within_deg: float = 25.0) -> bool:
    return any(abs(c[1] - doa_deg) <= within_deg for c in cands)
