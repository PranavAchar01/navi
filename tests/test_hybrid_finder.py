import numpy as np

from metasight.providers.finder import Detection, HybridFinder

FRAME = np.zeros((10, 10, 3), np.uint8)


class Fake:
    def __init__(self, dets):
        self.dets, self.calls = dets, []

    def find(self, bgr, labels):
        self.calls.append(list(labels))
        return [d for d in self.dets if d.label in labels]


def hybrid(local, meta, interval=1e9):
    h = HybridFinder.__new__(HybridFinder)
    h.local, h.meta = Fake(local), Fake(meta)
    h.items, h.interval, h.last_meta, h.meta_images = {"water bottle", "keys"}, interval, 0.0, 0
    return h


def test_a_local_glimpse_calls_sam_at_once_and_sam_wins():
    h = hybrid(
        [Detection("bed", 0.9, (0, 0, 5, 5)), Detection("water bottle", 0.2, (1, 1, 2, 2))],
        [Detection("water bottle", 0.8, (1, 1, 3, 3), "sam-3.1")],
    )
    h.last_meta = 1e18  # the timed sweep is not due
    out = h.find(FRAME, ["bed", "water bottle", "keys"])
    assert h.meta.calls == [["water bottle"]]
    assert [(d.label, d.source) for d in out] == [("bed", ""), ("water bottle", "sam-3.1")]


def test_a_sure_local_glimpse_stands_when_sam_misses_a_blurred_frame():
    h = hybrid(
        [Detection("water bottle", 0.5, (1, 1, 2, 2)), Detection("keys", 0.15, (0, 0, 1, 1))], []
    )
    h.last_meta = 1e18
    out = h.find(FRAME, ["water bottle", "keys"])
    assert [d.label for d in out] == ["water bottle"]


def test_nothing_glimpsed_and_no_sweep_due_costs_nothing():
    h = hybrid([Detection("bed", 0.9, (0, 0, 5, 5))], [])
    h.last_meta = 1e18
    h.find(FRAME, ["bed", "water bottle"])
    assert h.meta.calls == [] and h.meta_images == 0
