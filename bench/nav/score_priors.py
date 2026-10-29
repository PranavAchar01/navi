"""Score the belt with and without the map's priors against the walk's ground-truth labels.

    .venv/bin/python bench/nav/score_priors.py bench/results/nav/replay-*135040*.raw.json

Takes replay_walk.py raw runs of a walk that has labels (bench/labels/<walk>-sectors.json) and
scores them with scripts/score_labels.py's rules: false buzz = a motor on and no labelled obstacle
inside 2 m of the camera's view, missed = an obstacle inside 2 m (or 1 m) and no motor on, both as
events per minute, also with the labels scaled 0.84x and 1.16x. "Without" is rebuilt from the
command the server logged before the priors (replay_walk.belts). Writes
bench/results/nav/priors-vs-labels.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
from replay_walk import belts  # noqa: E402
from score_labels import OFF_M, SCALES, SLOW_M, _nearest, runs  # noqa: E402

FRONT = [7, 0, 1]


def main() -> None:
    out = {}
    for path in map(Path, sys.argv[1:]):
        walk = path.name.removeprefix("replay-")[:22]
        lab = json.loads((ROOT / "bench" / "labels" / f"{walk}-sectors.json").read_text())
        lt = np.array([f["t"] for f in lab["frames"]])
        tracked = np.array([f["tracked"] for f in lab["frames"]])
        near_l = np.array([_nearest(f) for f in lab["frames"]])
        ts, on_with, on_without = belts(json.loads(path.read_text()))
        i = np.clip(np.searchsorted(lt, ts), 0, len(lt) - 1)
        ok = tracked[i] & (np.abs(lt[i] - ts) < 0.3)
        near, minutes = near_l[i][ok], float(ts[ok][-1] - ts[ok][0]) / 60
        row: dict = {"frames": int(ok.sum()), "minutes": round(minutes, 2)}
        for name, on in (("with", on_with), ("without", on_without)):
            for scope, cols in (("front", FRONT), ("all", list(range(8)))):
                o = on[ok][:, cols].any(axis=1)
                r: dict = {"cue_on_pct": round(100 * float(o.mean()), 1)}
                for s in SCALES:
                    nm = near * s
                    r[f"x{s}"] = {
                        "false_per_min": round(runs(o & (nm >= OFF_M)) / minutes, 1),
                        "missed_2m_per_min": round(runs(~o & (nm < OFF_M)) / minutes, 1),
                        "missed_1m_per_min": round(runs(~o & (nm < SLOW_M)) / minutes, 1),
                    }
                row[f"{name}/{scope}"] = r
        out[path.name] = row
        print(path.name, row["frames"], "frames,", row["minutes"], "min")
        for k in ("with/all", "without/all"):
            print(f"  {k:12s} cue on {row[k]['cue_on_pct']}%  {row[k]['x1.0']}")
    (ROOT / "bench" / "results" / "nav" / "priors-vs-labels.json").write_text(
        json.dumps(out, indent=1)
    )


if __name__ == "__main__":
    main()
