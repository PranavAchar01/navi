import json

import pytest

from metasight.walk import WalkRecorder


def test_walk_writes_frames_poses_and_lens(tmp_path):
    w = WalkRecorder("20261028-1200-iphone", {"type": "camera", "hfov": 41.2}, tmp_path)
    w.frame(1761672000123.0, 7, b"\xff\xd8jpeg")
    w.pose({"type": "arpose", "seq": 7, "ts": 1761672000123.0, "m": [1.0] * 16})
    w.close()
    d = tmp_path / "20261028-1200-iphone"
    assert json.loads((d / "camera.json").read_text()) == {"hfov": 41.2}
    row = json.loads((d / "frames.jsonl").read_text())
    assert row == {"file": "1761672000123-7.jpg", "ts": 1761672000123.0, "seq": 7}
    assert (d / "frames" / row["file"]).read_bytes() == b"\xff\xd8jpeg"
    assert json.loads((d / "poses.jsonl").read_text())["seq"] == 7


def test_walk_keeps_motion_batches(tmp_path):
    w = WalkRecorder("walk", {}, tmp_path)
    w.motion({"type": "motion", "s": [[1.0, 0.0, -9.8, 0.1]]})
    w.close()
    assert json.loads((tmp_path / "walk" / "motion.jsonl").read_text())["s"][0][2] == -9.8


def test_reconnect_appends_to_the_same_walk(tmp_path):
    for seq in (1, 2):
        w = WalkRecorder("walk", {}, tmp_path)
        w.frame(1000.0 + seq, seq, b"x")
        w.close()
    assert len((tmp_path / "walk" / "frames.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("bad", ["../home", "a/b", "", "x" * 65, "walk.1"])
def test_walk_id_cannot_leave_the_walks_folder(tmp_path, bad):
    with pytest.raises(ValueError):
        WalkRecorder(bad, {}, tmp_path)


def test_walk_keeps_each_map_pose_and_the_arkit_tie(tmp_path):
    import numpy as np

    w = WalkRecorder("walk", {}, tmp_path)
    pose, align = np.eye(4), np.eye(4)
    pose[:3, 3], align[:3, 3] = [1.0, 0.0, 2.0], [5.0, 0.0, -2.0]
    w.map_pose(7, 1793220641.5, pose, None)  # before the first tie
    w.map_pose(12, 1793220642.5, pose, align)
    w.close()
    rows = [json.loads(x) for x in (tmp_path / "walk" / "map_poses.jsonl").read_text().splitlines()]
    assert [r["seq"] for r in rows] == [7, 12]
    assert rows[0]["ar_align"] is None
    assert np.allclose(rows[1]["pose"], pose) and np.allclose(rows[1]["ar_align"], align)
