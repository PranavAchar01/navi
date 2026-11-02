"""Where would a person hear each spatial cue? Scores every sound on both renderers.

SIMULATED LISTENER MODEL, NOT A HUMAN STUDY (metasight/spatial_listener.py).

    PYTHONPATH=. .venv/bin/python scripts/eval_localisation.py [--reps 20]

Each sound plays from every 15 degrees around the head at ear height, then again after a
10 degree head turn. Prints the mean lateral error, the mean azimuth error once the turn
has unfolded front and back, the worst azimuth, and the share heard on the wrong side;
writes bench/results/localisation.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metasight import spatial_listener as sl
from metasight import spatial_sound as ss

OUT = Path("bench/results/localisation.json")
SOUNDS = {
    "beacon": lambda: ss.beacon_sound(False),
    "beacon_aligned": lambda: ss.beacon_sound(True),
    "slow": lambda: ss.tick_sound("slow"),
    "fast": lambda: ss.tick_sound("fast"),
    "stop": lambda: ss.tick_sound("stop"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=20, help="internal-noise draws per azimuth")
    a = ap.parse_args()

    result: dict[str, dict] = {"label": "SIMULATED LISTENER MODEL, NOT A HUMAN STUDY"}
    print(result["label"])
    print(
        f"{'renderer':8s} {'sound':15s} {'lateral':>8s} {'azimuth':>8s} {'worst':>6s} {'wrong':>6s}"
    )
    for method in ("hrtf", "pan"):
        for name, make in SOUNDS.items():
            mono = make()
            rng = np.random.default_rng(0)
            rows = {az: sl.localise(mono, az, method, rng, a.reps) for az in range(-180, 180, 15)}
            worst = max(rows, key=lambda az: rows[az]["az_err"])
            s = {
                "lateral_err_deg": float(np.mean([r["lateral_err"] for r in rows.values()])),
                "az_err_deg": float(np.mean([r["az_err"] for r in rows.values()])),
                "worst_az": worst,
                "worst_err_deg": rows[worst]["az_err"],
                "wrong_side": float(np.mean([r["wrong_side"] for r in rows.values()])),
                "by_az": {str(k): v for k, v in rows.items()},
            }
            result[f"{method}/{name}"] = s
            print(
                f"{method:8s} {name:15s} {s['lateral_err_deg']:7.1f}d {s['az_err_deg']:7.1f}d"
                f" {s['worst_err_deg']:5.0f}d {s['wrong_side']:6.2f}"
            )
    OUT.write_text(json.dumps(result, indent=1))
    print(OUT)


if __name__ == "__main__":
    main()
