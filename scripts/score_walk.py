"""One command to score a recorded walk: strips, ground-truth labels, schemes, priors on vs off.

    PYTHONPATH=. .venv/bin/python scripts/score_walk.py WALK [--events safety.jsonl]

1. depth strips for WALK are extracted into the cache when missing (runs the depth model);
2. make_sector_labels writes bench/labels/WALK-sectors.json (ARKit walks only);
3. score_labels scores the live schemes into bench/results/label-scores.json;
4. with a safety-event log (one server "safety" event per line: seq, cues, priors), the same
   frames are scored with the shadow priors off (the live `cues`) and on (`priors.after` when
   the priors would change the command, else `cues`), against the same labels, into
   bench/results/priors-score.json. belt_check reads that file.
Default event log: bench/events/WALK-safety.jsonl. Without one, step 4 is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_sector_labels  # noqa: E402
import replay_false_buzz  # noqa: E402
import score_labels  # noqa: E402
from score_labels import OFF_M, SLOW_M, _nearest, runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FELT = 26  # intensity floor of the skin model, as in score_labels
OUT = ROOT / "bench" / "results" / "priors-score.json"


def felt(cmd: list[dict]) -> bool:
    return any(c["intensity"] >= FELT for c in cmd)


def _after(e: dict) -> list[dict]:
    """The command with the shadow priors applied (the live cues when they change nothing)."""
    return (e.get("priors") or {}).get("after") or e["cues"]


def _row(on: np.ndarray, near: np.ndarray, minutes: float) -> dict:
    close, inside = near < OFF_M, near < SLOW_M
    return {
        "cue_on_pct": round(100 * float(on.mean()), 1),
        "false_buzz_events_per_min": round(runs(on & ~close) / minutes, 1),
        "missed_inside_2m_frames": int((~on & close).sum()),
        "missed_inside_1m_frames": int((~on & inside).sum()),
        "missed_inside_1m_events_per_min": round(runs(~on & inside) / minutes, 1),
    }


def priors_on_off(lab: dict, events: Path) -> dict:
    by_seq = {}
    for line in events.read_text().splitlines():
        e = json.loads(line)
        if "cues" in e and "seq" in e:
            by_seq[e["seq"]] = e
    frames = [f for f in lab["frames"] if f["tracked"] and f["seq"] in by_seq]
    if not frames:
        return {"ok": None, "source": str(events), "reason": "no event matches a tracked frame"}
    off = np.array([felt(by_seq[f["seq"]]["cues"]) for f in frames])
    on = np.array([felt(_after(by_seq[f["seq"]])) for f in frames])
    near = np.array([_nearest(f) for f in frames])
    ts = np.array([f["t"] for f in frames])
    minutes = float((ts[-1] - ts[0]) / 60) or 1e-9
    res = {
        "source": str(events.relative_to(ROOT)) if events.is_relative_to(ROOT) else str(events),
        "frames_matched": len(frames),
        "frames_tracked": lab["frames_tracked"],
        "minutes": round(minutes, 2),
        "changed_frames": int((off != on).sum()),
        "priors_off": _row(off, near, minutes),
        "priors_on": _row(on, near, minutes),
    }
    res["ok"] = (
        res["priors_on"]["missed_inside_1m_frames"] <= res["priors_off"]["missed_inside_1m_frames"]
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("--events", type=Path)
    a = ap.parse_args()
    d = np.load(replay_false_buzz.CACHE) if replay_false_buzz.CACHE.exists() else None
    if d is None or f"{a.walk}/ts" not in d.files:
        replay_false_buzz.extract([a.walk])
    make_sector_labels.main([a.walk])
    score_labels.main([a.walk])
    events = a.events or ROOT / "bench" / "events" / f"{a.walk}-safety.jsonl"
    if not events.exists():
        print("no safety-event log at", events, "- priors on/off not scored")
        return
    lab = json.loads((ROOT / "bench" / "labels" / f"{a.walk}-sectors.json").read_text())
    res = priors_on_off(lab, events)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out[a.walk] = res
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
