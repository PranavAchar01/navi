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


def test_serial_kind_writes_packets_and_reads_telemetry_from_the_stream():
    import asyncio
    import os
    import tty

    from metasight.belt import BeltLink, packet

    board, host = os.openpty()  # a pty stands in for the USB cable
    tty.setraw(board)
    states: list[dict] = []

    async def go():
        link = BeltLink("serial", os.ttyname(host), states.append)
        await link.start()
        cmd = [(MODES["solid"], 200)] + [(MODES["off"], 0)] * 7
        link.send(cmd, 0, time.monotonic())
        await asyncio.sleep(0.05)
        got = os.read(board, 64)
        tel = b"MT" + struct.pack("<BBHII", 0, 0, 1, 0, 12) + bytes([200] + [0] * 7)
        os.write(board, b"\x00junk" + tel)  # a stray byte before the frame
        await asyncio.sleep(0.1)
        link.stop()
        return got, packet(cmd, 0, 1, 0)

    got, want = asyncio.run(go())
    assert got[:6] == want[:6] and got[10:] == want[10:]  # all but the timestamp
    assert states and states[-1]["device"] == "ESP32 (USB)"
    assert states[-1]["duty"][0] == 200 and states[-1]["age_ms"] == 12


def test_ble_pump_repeats_the_latest_command_inside_the_watchdog_then_lets_it_lapse():
    """One command, sent once: the pump resends it with fresh seqs well inside the belt's 250 ms
    watchdog for BLE_HOLD_S, then stops so a stalled pipeline still reads as a lost link."""
    import asyncio
    import sys
    import types

    from metasight import belt as beltmod
    from metasight.belt import BeltLink

    writes: list[tuple[float, bytes]] = []

    class Client:
        def __init__(self, target):
            self.is_connected = False

        async def connect(self):
            self.is_connected = True

        async def write_gatt_char(self, uuid, pkt, response):
            writes.append((time.monotonic(), bytes(pkt)))

    class Scanner:
        @staticmethod
        async def find_device_by_name(name, timeout):
            return "AA:BB"

    fake = types.ModuleType("bleak")
    fake.BleakClient, fake.BleakScanner = Client, Scanner
    old, sys.modules["bleak"] = sys.modules.get("bleak"), fake
    old_hold, beltmod.BLE_HOLD_S = beltmod.BLE_HOLD_S, 0.4

    async def go():
        link = BeltLink("ble", "", lambda s: None)
        link.send([(MODES["solid"], 200)] + [(MODES["off"], 0)] * 7, 0, time.monotonic())
        await asyncio.sleep(0.8)
        return link

    try:
        link = asyncio.run(go())
    finally:
        beltmod.BLE_HOLD_S = old_hold
        if old is None:
            sys.modules.pop("bleak")
        else:
            sys.modules["bleak"] = old
    assert 6 <= len(writes) <= 10  # about 20 Hz for 0.4 s, then silence
    gaps = [b[0] - a[0] for a, b in zip(writes, writes[1:], strict=False)]
    assert max(gaps) < 0.15  # far inside the 250 ms watchdog
    seqs = [struct.unpack_from("<H", p, 4)[0] for _, p in writes]
    assert seqs == sorted(set(seqs))  # each resend is a new packet
    assert all(p[10:] == writes[0][1][10:] for _, p in writes)  # the same command
    assert link.ble_latest is None


def test_strong_puts_every_cue_at_full_strength_on_the_diagonal_motors():
    from metasight.belt import MODES, strong

    cmd = [(MODES["off"], 0)] * 8
    cmd[0] = (MODES["fast"], 110)  # straight ahead: both front motors
    cmd[3] = (MODES["slow"], 40)
    cmd[4] = (MODES["guide"], 90)  # behind: takes the far slow cue beside it, as with_guide does
    cmd[5] = (MODES["solid"], 255)  # but never a near one
    cmd[6] = (MODES["unknown"], 0)
    out = strong(cmd)
    assert out[1] == out[7] == (MODES["fast"], 255)
    assert out[3] == (MODES["guide"], 255) and out[5] == (MODES["solid"], 255)
    assert all(out[c][0] in (MODES["off"], MODES["unknown"]) for c in (0, 2, 4, 6))


def test_strong_puts_the_front_cue_on_the_side_its_obstacle_is_on():
    from metasight.belt import CENTRE_DEG, MODES, strong

    off, fast = (MODES["off"], 0), (MODES["fast"], 255)
    cmd = [off] * 8
    cmd[0] = (MODES["fast"], 110)
    left = strong(cmd, -15.0)
    assert left[7] == fast and left[1] == off  # a chair left of centre: front-left only
    right = strong(cmd, 12.0)
    assert right[1] == fast and right[7] == off
    ahead = strong(cmd, CENTRE_DEG / 2)
    assert ahead[1] == ahead[7] == fast  # dead ahead: both
    assert strong(cmd)[1] == strong(cmd)[7] == fast  # side unknown: both, as before
    cmd[7] = (MODES["slow"], 30)  # the same object spilling onto the far motor: cleared
    assert strong(cmd, 15.0)[7] == off
    cmd[7] = (MODES["solid"], 255)  # a nearer cue of its own on the far motor stays
    assert strong(cmd, 15.0)[7] == (MODES["solid"], 255)


def test_scene_front_bearing_points_at_the_near_obstacle():
    import numpy as np

    from metasight.safety import GRID, analyse

    w, h = GRID[::-1]
    depth = np.full((h, w), 9.0, np.float32)
    depth[int(0.35 * h) : int(0.75 * h), int(0.08 * w) : int(0.3 * w)] = 1.2  # a box, left
    sc = analyse(depth, 1.0, 1.0, 44.8, 1.45, None)[0]  # no floor in view: plain depth
    assert sc.front_bearing_deg < -6


def test_rhythm_hold_escalates_at_once_and_steps_down_only_after_the_hold():
    from metasight.belt import MODES, RhythmHold

    fast, solid, off = (MODES["fast"], 255), (MODES["solid"], 255), (MODES["off"], 0)
    h = RhythmHold(0.5)
    assert h([fast] * 4, 0.0)[0] == fast
    assert h([solid] * 4, 0.1)[0] == solid  # nearer: at once
    assert h([fast] * 4, 0.2)[0] == solid  # one flicker back is not felt
    assert h([solid] * 4, 0.3)[0] == solid
    assert h([fast] * 4, 0.5)[0] == solid
    assert h([fast] * 4, 0.81)[0] == fast  # 0.5 s with no solid asked for
    assert h([off] * 4, 0.9)[0] == fast
    assert h([off] * 4, 1.32)[0] == off
