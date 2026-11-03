"""Put the replica render from the solved pose beside the real iPhone frame: replica/compare.png.

    .venv/bin/python scripts/replica_compare.py

Run after `blender_replica_render.py -- --shot match`. The real frame is the one the camera was
solved for (MATCH there), so the two should line up object for object.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "rooms" / "bedroom" / "replica"
REAL = ROOT / "data" / "walks" / "20261028-135040-iphone" / "frames" / "1793220651337-204.jpg"


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def main() -> None:
    real = cv2.imread(str(REAL))
    ren = cv2.imread(str(OUT / "match-render.png"))
    h = 1200
    real = cv2.resize(
        real, (round(real.shape[1] * h / real.shape[0]), h), interpolation=cv2.INTER_CUBIC
    )
    ren = cv2.resize(ren, (real.shape[1], h), interpolation=cv2.INTER_AREA)
    gap = np.full((h, 12, 3), 20, np.uint8)
    sheet = np.hstack([label(real, "his iPhone frame"), gap, label(ren, "replica, same pose")])
    cv2.imwrite(str(OUT / "compare.png"), sheet)


if __name__ == "__main__":
    main()
