import math
import socket
import struct
import subprocess
import time

from metasight.belt import EMU_BIN, FLAG_DROP, MODES, ensure_emulator, plan

INF = math.inf


def test_plan_unknown_sectors_never_buzz():
    cmd, _ = plan([0.3] * 8, [True] * 3 + [False] * 5, INF, INF, INF)
    assert cmd[3][0] == MODES["unknown"] and cmd[3][1] == 0
    assert cmd[0][0] == MODES["solid"]


def test_plan_closing_fast_raises_front():
    still, _ = plan([1.5] + [None] * 7, [True] * 8, INF, INF, INF)
    closing, _ = plan([1.5] + [None] * 7, [True] * 8, INF, 0.8, INF)
    assert still[0][0] == MODES["slow"] and closing[0][0] == MODES["fast"]


def test_plan_front_uses_walking_path_and_drop_flag():
    cmd, flags = plan([INF] * 8, [True] * 8, 0.4, INF, 1.0)
    assert cmd[0][0] == MODES["solid"] and flags & FLAG_DROP


def _tel(sock):
    data, _ = sock.recvfrom(64)
    lost, flags, seq, _e, _a = struct.unpack("<BBHII", data[2:14])
    return lost, list(data[14:22])


def test_emulator_runs_firmware_and_watchdog():
    ensure_emulator()
    port = 4300 + int(time.time()) % 500
    proc = subprocess.Popen([str(EMU_BIN), str(port)], stderr=subprocess.DEVNULL)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        time.sleep(0.2)
        body = struct.pack("<BB", MODES["solid"], 200) + b"\0\0" * 7
        for seq in range(1, 6):
            s.sendto(struct.pack("<2sBBHI", b"MS", 1, 0, seq, 0) + body, ("127.0.0.1", port))
            time.sleep(0.02)
        lost, duty = _tel(s)
        assert not lost and duty[0] == 200 and duty[1] == 0
        time.sleep(0.4)  # stop sending: the belt must fall back to its "lost" taps by itself
        seen = [_tel(s) for _ in range(80)]
        assert all(t[0] == 1 for t in seen[-10:])
        assert any(max(t[1]) > 0 for t in seen) and any(max(t[1]) == 0 for t in seen)
    finally:
        proc.terminate()
