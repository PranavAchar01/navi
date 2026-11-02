"""What the phone plays over a recorded take, as a stereo WAV to lay under the clip.

    PYTHONPATH=. .venv/bin/python scripts/render_take_sound.py bench/results/demo-1102/room.jsonl
    ... --out bench/results/room-sound.wav --pan --no-ticks
    ... --start 1793617303.48 --end 1793617344.14 --rate 48000 --find chair --find-from 1793617335

t = 0 is --start (wall clock), or the take's first message. The guide beacon, the arrival
notes and the hazard beeps come from the recorded safety messages, decided on the phone's
40 ms grid (metasight/spatial_sound.schedule_take).

A take recorded without a find (no guide in its messages) can still carry the beacon: --find
LABEL rebuilds the guide from that label's detections from --find-from on, the way the
server's guide does: each detection (box centre and depth) goes into the map through the
pose of its message, the item sits at the median of the last 2 s of them, and the bearing
and distance follow the head's pose between sightings. Arrival is within 0.8 m.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from metasight import spatial_sound as ss

HFOV_DEG = 44.8  # the glasses' camera, used when the take has no pose message saying it
KEEP_S = 2.0


def _frame(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Heading.to's horizontal forward and right for a map yaw."""
    fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    return fwd, np.cross(fwd, np.array([0.0, 1.0, 0.0]))


def add_guide(msgs: list[dict], label: str, since: float) -> int:
    """Write a guide (and found) into the safety messages from wall `since` on, from the
    take's detections of `label`. Returns how many messages got a guide."""
    hfov = next((m["hfov"] for m in reversed(msgs) if m.get("type") == "pose"), HFOV_DEG)
    tan_half = math.tan(math.radians(hfov / 2))
    seen: list[tuple[float, np.ndarray]] = []
    found, n = False, 0
    for m in msgs:
        if m.get("type") != "safety" or m["wall"] < since or m.get("pos") is None:
            continue
        pos = np.array(m["pos"], float)
        fwd, right = _frame(m["yaw"])
        for o in m.get("obstacles") or []:
            if o.get("label") == label and o.get("dist") is not None:
                b = math.atan((0.5 * (o["box"][0] + o["box"][2]) - 0.5) * 2 * tan_half)
                seen.append(
                    (m["wall"], pos + o["dist"] * (math.cos(b) * fwd + math.sin(b) * right))
                )
        if not seen:
            continue
        recent = [p for w, p in seen if w >= seen[-1][0] - KEEP_S]
        item = np.median(recent, axis=0)
        v = item - pos
        dist = float(math.hypot(v[0], v[2]))
        bearing = math.degrees(math.atan2(float(v @ right), float(v @ fwd)))
        found = found or dist < ss.ARRIVE_M
        m["guide"] = {"label": label, "bearing": bearing, "dist": dist}
        m["found"] = found
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("take", type=Path, help="a judge replay or live .jsonl")
    ap.add_argument("--out", type=Path, help="default: next to the take, <name>-sound.wav")
    ap.add_argument("--pan", action="store_true", help="stereo pan instead of the KEMAR HRTF")
    ap.add_argument("--no-ticks", action="store_true", help="the beacon only")
    ap.add_argument("--no-beacon", action="store_true", help="the hazard beeps only")
    ap.add_argument("--start", type=float, help="wall clock of t = 0")
    ap.add_argument("--end", type=float, help="wall clock where the WAV ends")
    ap.add_argument("--rate", type=int, default=ss.SAMPLE_RATE, help="48000 for video")
    ap.add_argument("--find", help="rebuild a guide to this label from its detections")
    ap.add_argument("--find-from", type=float, help="wall clock the find starts (with --find)")
    a = ap.parse_args()

    msgs = [json.loads(line) for line in a.take.read_text().splitlines() if line.strip()]
    if a.find:
        since = a.find_from if a.find_from is not None else min(m["wall"] for m in msgs)
        print(f"guide to {a.find}: {add_guide(msgs, a.find, since)} messages")
    beacon, ticks = not a.no_beacon, not a.no_ticks
    events = ss.schedule_take(msgs, beacon, ticks, a.start, a.end)
    x = ss.render_take(msgs, "pan" if a.pan else "hrtf", beacon, ticks, a.start, a.end)
    out = a.out or a.take.with_name(a.take.stem + "-sound.wav")
    ss.write_wav(out, x, a.rate)
    kinds = Counter(e.kind for e in events)
    print(f"{out}: {len(x) / ss.SAMPLE_RATE:.2f} s at {a.rate} Hz, peak {np.abs(x).max():.2f}")
    print(f"  {dict(kinds)}")
    for kind in ("beacon", "beacon_aligned", "arrive"):
        t = [e.t for e in events if e.kind == kind]
        if t:
            print(f"  {kind}: {len(t)}, first at {t[0]:.2f} s")


if __name__ == "__main__":
    main()
