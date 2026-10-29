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
    h.wanted = {}
    for item in h.items:
        h.want(item)  # asked for: the cases below are about how SAM 3.1 looks, not whether
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


def test_sam_looks_only_for_what_was_asked_for():
    """The house memory runs on the local model: SAM 3.1 bills only while an ask is open."""
    h = hybrid([Detection("water bottle", 0.2, (1, 1, 2, 2))], [], interval=0.0)
    h.wanted = {}
    h.find(FRAME, ["water bottle", "keys"])  # glimpsed and the sweep due, but nobody asked
    assert h.meta.calls == [] and h.meta_images == 0

    h.want("keys")
    h.want("chair")  # big things stay with the local model
    h.find(FRAME, ["water bottle", "keys", "chair"])
    assert h.meta.calls == [["keys"]]

    h.wanted["keys"] = 0.0  # the ask ran out
    h.find(FRAME, ["water bottle", "keys"])
    assert h.meta.calls == [["keys"]]


def test_a_never_seen_thing_is_swept_and_a_word_nobody_can_name_pins_nothing():
    """A new thing becomes a personal item for the ask: SAM 3.1 sweeps it, and a weak local hit
    on it stands only once SAM confirms it."""
    h = hybrid([Detection("umbrella", 0.15, (1, 1, 2, 2))], [], interval=0.0)
    h.wanted = {}
    h.want("umbrella", new=True)
    assert h.find(FRAME, ["bed", "umbrella"]) == []  # SAM saw none: the weak local hit is dropped
    assert h.meta.calls == [["umbrella"]]

    h.want("zorblax", new=True)  # neither model knows the word: no crash, no detection
    assert h.find(FRAME, ["zorblax"]) == []


# SAM 3.1's "keys" on walk 20261028-135040 at 18 s (640x360 frame): a box per keycap.
KEYCAPS = [
    (206, 576, 227, 592), (226, 585, 248, 602), (121, 569, 143, 589), (223, 551, 242, 568),
    (208, 556, 227, 573), (184, 620, 213, 640), (192, 562, 212, 580), (131, 581, 159, 604),
    (290, 570, 332, 605), (189, 581, 210, 598), (208, 592, 230, 609), (138, 595, 174, 621),
    (176, 552, 194, 568), (148, 607, 193, 638), (158, 574, 178, 593), (141, 563, 160, 580),
    (171, 587, 193, 606), (161, 631, 186, 640), (223, 569, 241, 586), (191, 598, 213, 619),
    (231, 590, 284, 628), (207, 611, 235, 640), (175, 568, 196, 585),
]  # fmt: skip


def sam_returning(per_label):
    """A Sam31Finder whose API answers come from `per_label`, so no request leaves the Mac."""
    from metasight.providers.finder import Sam31Finder

    sam = Sam31Finder.__new__(Sam31Finder)
    sam._api_key, sam.meta_images, sam.meta_ms = "test", 0, 0.0

    async def one(client, url, label):
        return [Detection(label, 0.55, box, "sam-3.1") for box in per_label.get(label, [])]

    sam._one = one
    return sam


def test_a_keycap_grid_is_a_texture_and_finds_no_keys():
    """23 small boxes for "keys" are the keyboard: dropped, so no pin; the bottle in the same
    sweep, and one or two real-size boxes for keys, still come through."""
    frame = np.zeros((640, 360, 3), np.uint8)
    sam = sam_returning({"keys": KEYCAPS, "water bottle": [(10, 10, 60, 120)]})
    assert [d.label for d in sam.find(frame, ["keys", "water bottle"])] == ["water bottle"]

    for boxes in ([(100, 300, 160, 340)], [(100, 300, 160, 340), (20, 20, 50, 45)]):
        sam = sam_returning({"keys": boxes})
        assert len(sam.find(frame, ["keys"])) == len(boxes)


def test_sam_boxes_are_marked_backed_only_with_a_sure_local_hit():
    sam_box = [Detection("keys", 0.8, (1, 1, 3, 3), "sam-3.1")]
    for local_score, backed in ((0.2, False), (0.35, True)):
        h = hybrid([Detection("keys", local_score, (1, 1, 2, 2))], sam_box)
        out = h.find(FRAME, ["keys"])
        assert [(d.source, d.backed) for d in out] == [("sam-3.1", backed)]
