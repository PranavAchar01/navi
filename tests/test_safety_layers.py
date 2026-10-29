"""The stop layer never waits for the detector, and the label layer adds the same boxes and
distances the single pass gave."""

import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from metasight.config import Settings
from metasight.safety import SafetyLoop


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy, self.cls, self.conf = (
            torch.tensor(v, dtype=torch.float32) for v in (xyxy, cls, conf)
        )


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def predict(self, bgr, **_):
        self.calls += 1
        h, w = bgr.shape[:2]
        boxes = FakeBoxes([[0.4 * w, 0.3 * h, 0.6 * w, 0.9 * h]], [0], [0.8])
        return [SimpleNamespace(boxes=boxes, names={0: "chair"})]


def make_loop():
    loop = SafetyLoop.__new__(SafetyLoop)
    loop.s, loop._seg, loop._det = Settings(), None, FakeDetector()
    loop.scale, loop.plane, loop.plane_t, loop.floor_t = 1.0, None, 0.0, 0.0
    loop.path_hist, loop.drop_hist = deque(), deque(maxlen=4)
    # A floor that falls away below the horizon and a wall 2 m ahead above it.
    h, w = 294, 518
    depth = np.full((h, w), 2.0, np.float32)
    rows = np.arange(h // 2, h)
    depth[h // 2 :] = np.minimum(2.0, 1.37 * 300 / (rows - h / 2 + 1))[:, None]
    loop.estimate_depth = lambda bgr: depth.copy()
    return loop


def test_reflex_runs_no_detector_and_carries_the_stop_inputs():
    loop = make_loop()
    res = loop.reflex(np.zeros((294, 518, 3), np.uint8))
    assert loop._det.calls == 0
    assert res.obstacles == [] and res.objects == []
    assert len(res.sectors) == 3 and res.scene is not None
    assert set(res.timings_ms) == {"depth"}


def test_label_after_reflex_matches_one_pass():
    bgr = np.zeros((294, 518, 3), np.uint8)
    one = make_loop().process(bgr)
    loop = make_loop()
    two = loop.label(bgr, loop.reflex(bgr))
    assert loop._det.calls == 1
    assert one.sectors == two.sectors
    assert [(o.label, o.box) for o in one.obstacles] == [(o.label, o.box) for o in two.obstacles]
    # The central half of the box, 25th percentile of full-precision depth (as before the
    # split), not the float16 copy sent to the map.
    d = loop.estimate_depth(bgr) * loop.scale
    want = float(
        np.percentile(
            d[int(0.45 * 294) : int(0.75 * 294) + 1, int(0.45 * 518) : int(0.55 * 518) + 1], 25
        )
    )
    assert math.isclose(two.obstacles[0].dist_m, want, rel_tol=1e-6)
    assert math.isclose(one.obstacles[0].dist_m, two.obstacles[0].dist_m)
    assert {"depth", "detect", "post", "total"} <= set(two.timings_ms)
