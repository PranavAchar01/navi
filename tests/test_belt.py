import math
import socket
import struct
import subprocess
import time

from metasight.belt import EMU_BIN, FLAG_DROP, FLAG_FOUND, MODES, ensure_emulator, guide, plan

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


def test_guide_points_the_facing_motor_and_yields_to_obstacles():
    cmd, _ = plan([None] * 8, [True] * 8, INF, INF, INF)
    guide(cmd, 95.0, 3.0)  # on the right
    assert cmd[2][0] == MODES["guide"] and sum(m == MODES["guide"] for m, _ in cmd) == 1
    near, _ = plan([None] * 8, [True] * 8, INF, INF, INF)
    guide(near, 95.0, 1.0)
    assert near[2][1] > cmd[2][1]  # closer is stronger
    blocked, _ = plan([0.3, 1.5] + [None] * 6, [True] * 8, INF, INF, INF)
    guide(blocked, 0.0, 2.0)
    guide(blocked, 45.0, 2.0)
    assert blocked[0][0] == MODES["solid"] and blocked[1][0] == MODES["guide"]


def _tel(sock):
    data, _ = sock.recvfrom(64)
    lost, flags, seq, _e, _a = struct.unpack("<BBHII", data[2:14])
    return lost, list(data[14:22])


def test_emulator_runs_firmware_and_watchdog():
    ensure_emulator()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # free now; avoids clashes with other test runs
    proc = subprocess.Popen([str(EMU_BIN), str(port)], stderr=subprocess.DEVNULL)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        body = struct.pack("<BB", MODES["solid"], 200) + b"\0\0" * 7
        seq = 0
        deadline = time.time() + 15.0  # a loaded machine can be slow to start the emulator
        while True:  # send until it answers: poll for readiness, not a fixed sleep
            seq += 1
            s.sendto(struct.pack("<2sBBHI", b"MS", 1, 0, seq, 0) + body, ("127.0.0.1", port))
            try:
                lost, duty = _tel(s)
            except TimeoutError:
                assert time.time() < deadline, "emulator never answered"
                continue
            if not lost:
                break
        s.settimeout(1.0)
        assert duty[0] == 200 and duty[1] == 0
        time.sleep(0.4)  # stop sending: the belt must fall back to its "lost" taps by itself
        seen = [_tel(s) for _ in range(80)]
        assert all(t[0] == 1 for t in seen[-10:])
        assert any(max(t[1]) > 0 for t in seen) and any(max(t[1]) == 0 for t in seen)
    finally:
        proc.terminate()


def _run(s, port, cmd, flags, seconds, seq0):
    body = b"".join(struct.pack("<BB", m, i) for m, i in cmd)
    seen, seq, end = [], seq0, time.time() + seconds
    while time.time() < end:
        seq += 1
        s.sendto(struct.pack("<2sBBHI", b"MS", 1, flags, seq, 0) + body, ("127.0.0.1", port))
        seen.append(_tel(s)[1])
    return seen


def test_emulator_guide_taps_and_found_sweep():
    ensure_emulator()
    port = 4800 + int(time.time()) % 500
    proc = subprocess.Popen([str(EMU_BIN), str(port)], stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.2)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        cmd = [(0, 0)] * 8
        cmd[2] = (MODES["guide"], 180)
        seen = _run(s, port, cmd, 0, 1.1, 0)
        assert {d[2] for d in seen} == {0, 180} and all(d[0] == 0 for d in seen)
        swept = _run(s, port, [(0, 0)] * 8, FLAG_FOUND, 1.1, 1000)
        lit = {i for d in swept for i in range(8) if d[i]}
        assert lit == set(range(8)) and all(sum(1 for v in d if v) <= 1 for d in swept)
    finally:
        proc.terminate()


def test_packet_is_the_26_byte_command_for_udp_and_ble():
    from metasight.belt import packet

    cmd = [(m, 10 * m) for m in range(6)] + [(0, 0)] * 2
    pkt = packet(cmd, FLAG_DROP, 513, 0x01020304)
    assert len(pkt) == 26 and pkt[:3] == b"MS\x01" and pkt[3] == FLAG_DROP
    assert (
        struct.unpack("<H", pkt[4:6])[0] == 513 and struct.unpack("<I", pkt[6:10])[0] == 0x01020304
    )
    assert pkt[10:14] == bytes([0, 0, 1, 10])


def test_app_kind_relays_each_packet_and_reports_the_command():
    import asyncio
    import base64

    from metasight.belt import BeltLink

    relayed, states = [], []

    async def go():
        link = BeltLink("app", "AA", states.append, relay=relayed.append)
        await link.start()
        cmd, flags = plan([0.3] + [None] * 7, [True] * 8, INF, INF, INF)
        link.send(cmd, flags, time.monotonic())
        return link

    asyncio.run(go())
    assert relayed[0]["type"] == "beltcmd"
    assert len(base64.b64decode(relayed[0]["pkt"])) == 26
    assert states[0]["modes"][0] == "solid" and states[0]["lost"] is False
