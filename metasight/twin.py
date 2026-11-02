"""The room as a living set of objects: one track per physical thing, followed across walks.

Each walk gives observations (class, position, size, the mesh triangles they cover). A track is
one object over time. Walks are folded in order, and an observation joins the nearest track of
its class family within reach (SAME_M, or half the object's footprint for big things). Then:

- a track not re-seen although the camera looked at its last spot is "missing", and after
  MISSING_WALKS such walks "removed";
- an observation with no track in reach takes over a same-family movable track whose old spot
  was in view and empty ("moved", with from and to), and otherwise starts a new track ("added"
  when its spot was seen empty in an earlier walk, "first_seen" when it never was).

Size is the median over sightings, so it sharpens as views pile up; position is the last
sighting's, because furniture moves and the latest look is the truth.

Pure logic: callers say whether a spot was in view through a `visible(obs, walk)` callback.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .recon import PRIORS

SAME_M = 0.35
MOVE_MAX_M = 4.0
MISSING_WALKS = 2
FAMILIES = [{"chair", "office chair"}, {"bag", "backpack"}, {"computer monitor", "computer"}]


def family(cls: str) -> str:
    for f in FAMILIES:
        if cls in f:
            return min(f)
    return cls


def _dist(a: dict, b: dict) -> float:
    return float(np.linalg.norm(np.asarray(a["pos"]) - np.asarray(b["pos"])))


def reach(o: dict) -> float:
    """How far apart two sightings of this object can sit and still be it: a partial view of
    a bed puts its centre a long way from a full view's, a mug's barely moves."""
    return max(SAME_M, 0.5 * max(o["size"][0], o["size"][2]))


def movable(cls: str) -> bool:
    return PRIORS.get(cls, ("", 0, 0, True))[3]


def _event(kind: str, walk: str, ts: int, **kw) -> dict:
    return {"type": kind, "walk": walk, "ts_ms": ts, **kw}


def build_tracks(
    walks: list[tuple[str, int, list[dict]]],
    visible: Callable[[dict, str], bool],
) -> list[dict]:
    """Fold (walk, start ts ms, observations) in time order into object tracks with events."""
    tracks: list[dict] = []
    looked: dict[str, list[str]] = {}  # walk -> earlier walks, for "added" vs "first_seen"
    order = sorted(walks, key=lambda w: w[1])
    for wi, (walk, ts, obs) in enumerate(order):
        looked[walk] = [w for w, _, _ in order[:wi]]
        live = [t for t in tracks if t["state"] != "removed"]
        pairs = sorted(
            (_dist(o, t["last"]), i, j)
            for i, o in enumerate(obs)
            for j, t in enumerate(live)
            if family(o["class"]) == family(t["class"])
        )
        used_o: set[int] = set()
        used_t: set[int] = set()
        for d, i, j in pairs:
            if d > reach(live[j]["last"]) or i in used_o or j in used_t:
                continue
            used_o.add(i)
            used_t.add(j)
            _sight(live[j], obs[i], walk, ts)
        # Unexplained observations: a moved object, or a new one.
        for i, o in enumerate(obs):
            if i in used_o:
                continue
            gone = [
                (j, _dist(o, t["last"]))
                for j, t in enumerate(live)
                if j not in used_t
                and family(t["class"]) == family(o["class"])
                and movable(t["class"])
                and visible(t["last"], walk)
                and _dist(o, t["last"]) < MOVE_MAX_M
            ]
            if gone:
                j = min(gone, key=lambda g: g[1])[0]
                used_t.add(j)
                t = live[j]
                t["events"].append(
                    _event(
                        "moved", walk, o.get("ts_ms", ts), **{"from": t["last"]["pos"]}, to=o["pos"]
                    )
                )
                _sight(t, o, walk, ts)
                continue
            seen_before = [w for w in looked[walk] if visible(o, w)]
            t = {
                "class": o["class"],
                "sightings": [],
                "events": [
                    _event(
                        "added" if seen_before else "first_seen",
                        walk,
                        o.get("ts_ms", ts),
                        pos=o["pos"],
                        **({"empty_in": seen_before} if seen_before else {}),
                    )
                ],
                "state": "present",
                "missing_walks": 0,
            }
            _sight(t, o, walk, ts)
            tracks.append(t)
        # Tracks not re-seen: missing only when their spot was in view.
        for j, t in enumerate(live):
            if j in used_t or not visible(t["last"], walk):
                continue
            t["missing_walks"] += 1
            kind = "removed" if t["missing_walks"] >= MISSING_WALKS else "missing"
            if t["state"] != kind:
                t["events"].append(_event(kind, walk, ts, pos=t["last"]["pos"]))
            t["state"] = kind
    for t in tracks:
        _fuse(t)
    return tracks


def _sight(t: dict, o: dict, walk: str, ts: int) -> None:
    t["sightings"].append({**o, "walk": walk, "ts_ms": o.get("ts_ms", ts)})
    t["last"] = o
    t["missing_walks"] = 0
    t["state"] = "present"


def _fuse(t: dict) -> None:
    sizes = np.array([s["size"] for s in t["sightings"]], float)
    t["pos"] = list(map(float, t["last"]["pos"]))
    t["size_m"] = np.median(sizes, 0).round(3).tolist()
    t["size_sd_m"] = sizes.std(0).round(3).tolist() if len(sizes) > 1 else None
    t["walks"] = sorted({s["walk"] for s in t["sightings"]})
    t["first_ts_ms"] = min(s["ts_ms"] for s in t["sightings"])
    t["last_ts_ms"] = max(s["ts_ms"] for s in t["sightings"])


def where_is(tracks: list[dict], cls: str) -> list[dict]:
    """Last known places of every object of this class family, most recently seen first."""
    hits = [t for t in tracks if family(t["class"]) == family(cls) and t["state"] != "removed"]
    return sorted(hits, key=lambda t: -t["last_ts_ms"])
