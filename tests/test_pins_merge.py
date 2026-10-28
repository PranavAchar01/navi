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


def test_moved_keys_answer_with_the_new_place():
    s = PinStore()
    for i in range(6):  # a long time on the table
        s.observe("keys", "object", [0.0, 0.8, 0.0], 10.0 + i, score=0.3)
    s.observe("keys", "object", [4.0, 0.9, 2.0], 500.0, score=0.3)
    s.observe("keys", "object", [4.0, 0.9, 2.0], 501.0, score=0.3)  # two looks at the counter
    assert s.latest("keys").pos[0] == 4.0


def test_a_lone_unsure_sighting_does_not_beat_the_firm_pin():
    s = PinStore()
    for i in range(3):
        s.observe("keys", "object", [0.0, 0.8, 0.0], 10.0 + i, score=0.3)
    s.observe("keys", "object", [4.0, 0.9, 2.0], 500.0, score=0.2)  # one weak look
    assert s.latest("keys").pos[0] == 0.0
    s.observe("keys", "object", [8.0, 0.9, 2.0], 600.0, score=0.6)  # one look, crop check firm
    assert s.latest("keys").pos[0] == 8.0


def test_with_no_firm_pin_the_most_seen_stands_in():
    s = PinStore()
    s.observe("keys", "object", [0.0, 0.8, 0.0], 10.0, score=0.2)
    s.observe("keys", "object", [4.0, 0.9, 2.0], 500.0, score=0.2)
    s.pins[1].seen = 1
    assert s.latest("keys").pos[0] == 4.0
