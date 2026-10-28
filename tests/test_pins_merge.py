from metasight.pins import PinStore


def test_merge_folds_drifted_duplicates_and_answers_with_the_most_seen():
    s = PinStore()
    for i in range(5):
        s.observe("chair", "object", [1.0, 0.5, 0.0], 10.0 + i)
    s.observe("chair", "object", [1.8, 0.5, 0.0], 99.0)  # a stray, newer, seen once
    s.observe("cup", "object", [1.0, 0.9, 0.0], 50.0)
    assert len(s.pins) == 3
    s.merge(1.0)
    assert sorted(p.label for p in s.pins.values()) == ["chair", "cup"]
    chair = s.latest("chair")
    assert chair is not None and chair.seen == 6 and chair.ts == 99.0


def test_latest_prefers_the_better_seen_pin_over_the_newer():
    s = PinStore()
    for i in range(4):
        s.observe("chair", "object", [0.0, 0.0, 0.0], float(i))
    s.observe("chair", "object", [5.0, 0.0, 0.0], 100.0)
    s.observe("chair", "object", [5.0, 0.0, 0.0], 101.0)
    assert s.latest("chair").seen == 4
