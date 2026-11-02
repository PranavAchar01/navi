from metasight.twin import build_tracks, family, where_is


def ob(cls, x, z=0.0, size=(1.0, 0.5, 1.0), ts=0):
    return {"class": cls, "pos": [x, 0.5, z], "size": list(size), "ts_ms": ts}


def always(_o, _w):
    return True


def never(_o, _w):
    return False


def test_same_object_across_walks_is_one_track_and_size_is_median():
    walks = [
        ("a", 0, [ob("bed", 0.0, size=(1.9, 0.5, 1.6))]),
        ("b", 10, [ob("bed", 0.1, size=(2.1, 0.5, 1.7))]),
        ("c", 20, [ob("bed", 0.05, size=(2.0, 0.5, 1.8))]),
    ]
    (t,) = build_tracks(walks, always)
    assert t["walks"] == ["a", "b", "c"]
    assert t["size_m"] == [2.0, 0.5, 1.7]
    assert t["state"] == "present"
    assert [e["type"] for e in t["events"]] == ["first_seen"]


def test_moved_when_old_spot_is_empty():
    walks = [("a", 0, [ob("chair", 0.0)]), ("b", 10, [ob("office chair", 1.5, ts=12)])]
    (t,) = build_tracks(walks, always)
    assert [e["type"] for e in t["events"]] == ["first_seen", "moved"]
    assert t["events"][1]["from"][0] == 0.0 and t["events"][1]["to"][0] == 1.5
    assert t["events"][1]["ts_ms"] == 12
    assert t["pos"][0] == 1.5


def test_new_instance_when_old_spot_was_not_in_view():
    walks = [("a", 0, [ob("chair", 0.0)]), ("b", 10, [ob("chair", 1.5)])]
    tracks = build_tracks(walks, never)
    assert len(tracks) == 2
    assert tracks[1]["events"][0]["type"] == "first_seen"


def test_added_and_removed():
    def vis(o, w):
        return not (w == "a" and o["pos"][0] > 1)  # walk a never looked at x > 1

    walks = [
        ("a", 0, [ob("box", 0.0)]),
        ("b", 10, [ob("shoe", 2.0)]),
        ("c", 20, [ob("shoe", 2.0)]),
    ]
    box, shoe = build_tracks(walks, vis)
    assert [e["type"] for e in box["events"]] == ["first_seen", "missing", "removed"]
    assert box["state"] == "removed"
    assert shoe["events"][0]["type"] == "first_seen"

    walks.insert(1, ("a2", 5, [ob("box", 0.0)]))
    tracks = build_tracks(walks, always)
    shoe = next(t for t in tracks if t["class"] == "shoe")
    assert shoe["events"][0]["type"] == "added"
    assert shoe["events"][0]["empty_in"] == ["a", "a2"]


def test_where_is_skips_removed_and_ranks_recent():
    walks = [("a", 0, [ob("bag", 0.0, ts=1), ob("backpack", 3.0, ts=2)])]
    tracks = build_tracks(walks, always)
    assert family("backpack") == family("bag")
    assert [t["pos"][0] for t in where_is(tracks, "backpack")] == [3.0, 0.0]
