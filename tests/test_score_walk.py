import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_walk import priors_on_off  # noqa: E402

ON = [{"mode": 3, "intensity": 255}] * 8
OFF = [{"mode": 0, "intensity": 0}] * 8


def _frame(seq, t, thirds):
    return {"seq": seq, "t": t, "tracked": True, "fov_thirds": thirds}


def _lab():
    # frame 2 has an obstacle at 0.8 m: inside 1 m, so a silent command misses it
    return {
        "frames_tracked": 3,
        "frames": [_frame(0, 0.0, [None] * 3), _frame(2, 1.0, [0.8, None, None]),
                   _frame(4, 2.0, [None] * 3)],
    }  # fmt: skip


def _events(tmp_path, rows):
    p = tmp_path / "ev.jsonl"
    p.write_text("\n".join(json.dumps({"type": "safety", **r}) for r in rows))
    return p


def test_priors_that_cover_a_miss_pass(tmp_path):
    ev = _events(tmp_path, [
        {"seq": 0, "cues": OFF},
        {"seq": 2, "cues": OFF, "priors": {"after": ON}},
        {"seq": 4, "cues": OFF},
    ])  # fmt: skip
    r = priors_on_off(_lab(), ev)
    assert r["frames_matched"] == 3 and r["changed_frames"] == 1
    assert r["priors_off"]["missed_inside_1m_frames"] == 1
    assert r["priors_on"]["missed_inside_1m_frames"] == 0
    assert r["ok"] is True


def test_priors_that_silence_a_cue_fail(tmp_path):
    ev = _events(tmp_path, [
        {"seq": 0, "cues": OFF},
        {"seq": 2, "cues": ON, "priors": {"after": OFF}},
        {"seq": 4, "cues": OFF},
    ])  # fmt: skip
    r = priors_on_off(_lab(), ev)
    assert (
        r["priors_on"]["missed_inside_1m_frames"] == 1 > r["priors_off"]["missed_inside_1m_frames"]
    )
    assert r["ok"] is False


def test_events_matching_no_frame_are_not_scored(tmp_path):
    r = priors_on_off(_lab(), _events(tmp_path, [{"seq": 99, "cues": OFF}]))
    assert r["ok"] is None
