import numpy as np

from metasight.providers.finder import Detection, HybridFinder

FRAME = np.zeros((10, 10, 3), np.uint8)


class Fake:
    def __init__(self, dets):
        self.dets, self.calls, self.shapes = dets, [], []

    def find(self, bgr, labels):
        self.calls.append(list(labels))
        self.shapes.append(bgr.shape[:2])
        return [d for d in self.dets if d.label in labels]

    def find_jobs(self, jobs):
        return [self.find(img, labels) for img, labels in jobs]


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


def test_a_cup_on_the_desk_makes_sam_look_for_the_water_bottle():
    h = hybrid(
        [Detection("cup", 0.17, (1, 1, 2, 2))],
        [Detection("water bottle", 0.7, (1, 1, 2, 3), "sam-3.1")],
    )
    h.last_meta = 1e18
    out = h.find(FRAME, ["cup", "water bottle", "keys"])
    assert h.meta.calls == [["water bottle"]]
    assert {d.label for d in out} == {"cup", "water bottle"}


def test_sam_gets_a_crop_for_a_glimpse_and_a_shrunk_frame_for_the_sweep():
    """Privacy: a glimpsed item leaves the Mac as a crop around it, the timed sweep as a frame
    shrunk to 640 px; boxes come back in full-frame pixels."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    h = hybrid(
        [Detection("water bottle", 0.2, (1000, 500, 1040, 580))],
        [
            Detection("water bottle", 0.8, (10, 20, 50, 100), "sam-3.1"),
            Detection("keys", 0.8, (64, 36, 96, 72), "sam-3.1"),
        ],
        interval=0.0,
    )
    out = h.find(frame, ["water bottle", "keys"])  # the sweep is due: keys go in the shrunk frame
    assert h.meta.calls == [["water bottle"], ["keys"]]
    crop, small = h.meta.shapes
    assert crop[0] < 1080 and crop[1] < 1920 and min(crop) >= 240
    assert max(small) == 640
    by = {d.label: d.box for d in out}
    assert by["keys"] == (192.0, 108.0, 288.0, 216.0)
    x0, y0 = by["water bottle"][:2]
    assert 1000 - 300 < x0 < 1000 and 400 < y0 < 600


def test_hybrid_without_a_key_falls_back_to_the_local_finder(monkeypatch):
    import metasight.providers.finder as fm
    from metasight.config import Settings

    monkeypatch.setattr(fm, "secret", lambda *names: None)
    monkeypatch.setattr(fm.YoloeFinder, "__init__", lambda self, s: None)
    s = Settings()
    s.finder = "hybrid"
    assert isinstance(fm.make_finder(s), fm.YoloeFinder)
