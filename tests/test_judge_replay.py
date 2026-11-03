"""judge_replay: a glasses walk's recorded map poses reach the server, and exit always cleans up."""

import importlib.util
import json
import os
import signal
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("judge_replay", ROOT / "scripts/judge_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_map_poses_become_arkit_messages_the_server_reads_back(tmp_path):
    from metasight.server import arkit_pose

    jr = _script()
    a, b = np.eye(4), np.eye(4)
    b[:3, :3] = Rotation.from_euler("y", 40, degrees=True).as_matrix()
    b[:3, 3] = [1.0, 0.0, 2.0]
    rows = [{"seq": 10, "ts": 1.0, "pose": a.tolist()}, {"seq": 20, "ts": 2.0, "pose": b.tolist()}]
    (tmp_path / "map_poses.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = jr.walk_poses(tmp_path, [5, 10, 15, 20, 25])
    assert set(out) == {5, 10, 15, 20, 25}
    assert all(m["type"] == "arpose" and m["track"] == "tracking" for m in out.values())
    assert np.allclose(arkit_pose(out[20]["m"]), b)
    assert np.allclose(arkit_pose(out[5]["m"]), a)  # before the first keyframe: held
    mid = arkit_pose(out[15]["m"])
    assert np.allclose(mid[:3, 3], [0.5, 0.0, 1.0])
    yaw = Rotation.from_matrix(mid[:3, :3]).as_euler("yxz", degrees=True)[0]
    assert abs(yaw - 20) < 1e-6


def test_a_phone_walk_keeps_its_own_arkit_poses(tmp_path):
    jr = _script()
    msg = {"type": "arpose", "seq": 3, "m": list(range(16)), "track": "tracking"}
    (tmp_path / "poses.jsonl").write_text(json.dumps(msg) + "\n")
    (tmp_path / "map_poses.jsonl").write_text("")
    assert jr.walk_poses(tmp_path, [3]) == {3: msg}


def test_exit_frees_the_slot_when_the_group_is_already_gone(tmp_path, monkeypatch):
    """GAPS #13: os.killpg raised PermissionError at exit and the heavy-lock slot stayed."""
    jr = _script()
    slot, home = tmp_path / ".heavy-lock-9", tmp_path / "home"
    slot.mkdir()
    home.mkdir()
    (slot / "owner").write_text("1 test\n")

    def gone(pid, sig):
        raise PermissionError(1, "Operation not permitted")

    class Done:
        pid = 999999

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(os, "killpg", gone)
    old = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    try:
        jr.stop_server(Done(), slot, home)
    finally:
        signal.signal(signal.SIGINT, old[0])
        signal.signal(signal.SIGTERM, old[1])
    assert not slot.exists() and not home.exists()


def test_a_server_that_ignores_sigterm_is_killed(tmp_path, monkeypatch):
    jr = _script()
    sent = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: sent.append(sig))

    class Stuck:
        pid = 999999
        calls = 0

        def wait(self, timeout=None):
            Stuck.calls += 1
            if Stuck.calls == 1:
                raise subprocess.TimeoutExpired("server", timeout)
            return 0

    old = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    try:
        jr.stop_server(Stuck(), tmp_path / "slot", tmp_path / "home")
    finally:
        signal.signal(signal.SIGINT, old[0])
        signal.signal(signal.SIGTERM, old[1])
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_a_walk_without_a_lens_gets_the_glasses_hfov():
    jr = _script()
    assert jr.walk_hfov({"width": 504, "height": 896}) == 44.8
    assert jr.walk_hfov({"width": 640, "height": 480, "hfov": 62.0}) == 62.0
    assert abs(jr.walk_hfov({"width": 640, "height": 480, "fx": 320.0}) - 90.0) < 1e-9
