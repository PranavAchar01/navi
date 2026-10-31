"""A walk saved on the Mac: every camera frame a phone sends, the phone's own tracked pose for each
one and the lens it reported, so the house map can be scored later against the phone's path.

Written under data/walks/<walk id>/ (data/ is gitignored: footage of someone's home never goes
into the repo):
  camera.json   the lens and stream size the phone last reported
  frames/       one JPEG per frame, named by its timestamp in ms
  frames.jsonl  {"file", "ts", "seq"} per frame, in arrival order
  poses.jsonl   the phone's messages verbatim: {"type": "arpose", "seq", "ts", "m": 16 floats
                column-major camera-to-world (ARKit: y up, metres), "track"}
  map_poses.jsonl  the map's pose for each keyframe, as memory placed it: {"seq", "ts", "pose":
                4x4 camera-to-map rows (OpenCV axes), "ar_align": 4x4 rows or null}. ar_align
                takes the phone's pose into the map, map = ar_align @ server.arkit_pose(m), so
                a replay joins the two exactly
  motion.jsonl  the device's motion batches verbatim: {"type": "motion", "s": [[t, ax, ay, az,
                (gx, gy, gz)], ...]}; from the glasses this is the only record of which way is
                down, since they have no ARKit pose
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ROOT

WALKS = Path(os.environ.get("METASIGHT_WALKS") or ROOT / "data" / "walks")
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
log = logging.getLogger(__name__)


class WalkRecorder:
    """Appends to one walk folder. Writes happen on its own thread so a slow disk never holds up
    the frame loop; a phone that reconnects with the same walk id keeps appending."""

    def __init__(self, walk_id: str, camera: dict, root: Path = WALKS):
        if not _ID.fullmatch(walk_id):
            raise ValueError(f"bad walk id {walk_id!r}")
        self.id = walk_id
        self.dir = root / walk_id
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="walk")
        self._frames = open(self.dir / "frames.jsonl", "a")
        self._poses = open(self.dir / "poses.jsonl", "a")
        self._map_poses = open(self.dir / "map_poses.jsonl", "a")
        self._motion = open(self.dir / "motion.jsonl", "a")
        self.n = 0
        self.camera(camera)

    def camera(self, info: dict) -> None:
        text = json.dumps({k: v for k, v in info.items() if k != "type"}, indent=1)
        self._pool.submit((self.dir / "camera.json").write_text, text)

    def frame(self, ts: float, seq: int, jpeg: bytes) -> None:
        self.n += 1
        self._pool.submit(self._write_frame, ts, seq, jpeg)

    def pose(self, msg: dict) -> None:
        self._pool.submit(self._write_line, self._poses, msg)

    def motion(self, msg: dict) -> None:
        self._pool.submit(self._write_line, self._motion, msg)

    def map_pose(self, seq: int | None, ts: float | None, pose, ar_align) -> None:
        line = {
            "seq": seq,
            "ts": ts,
            "pose": pose.tolist(),
            "ar_align": None if ar_align is None else ar_align.tolist(),
        }
        self._pool.submit(self._write_line, self._map_poses, line)

    def close(self) -> None:
        self._pool.submit(self._frames.close)
        self._pool.submit(self._poses.close)
        self._pool.submit(self._map_poses.close)
        self._pool.submit(self._motion.close)
        self._pool.shutdown(wait=True)
        log.info("walk %s: %d frames this connection in %s", self.id, self.n, self.dir)

    def _write_frame(self, ts: float, seq: int, jpeg: bytes) -> None:
        name = f"{int(ts)}-{seq}.jpg"
        (self.dir / "frames" / name).write_bytes(jpeg)
        self._write_line(self._frames, {"file": name, "ts": ts, "seq": seq})

    @staticmethod
    def _write_line(f, msg: dict) -> None:
        f.write(json.dumps(msg, separators=(",", ":")) + "\n")
        f.flush()
