import socket
import struct
import subprocess
import time

from metasight import belt_modes as bm
from metasight import modes
from metasight.belt import EMU_BIN, MODES, ensure_emulator


def test_each_mode_has_its_own_signature_and_it_ends():
    seen = set()
    for m in modes.ALL:
        frames = [tuple(bm.signature_duty(m, t)) for t in range(0, 1200, 20)]
        assert any(any(f) for f in frames) and not any(bm.signature_duty(m, 1200))
        seen.add(tuple(frames))
    assert len(seen) == 4


def test_find_and_talk_cues_are_panned_and_use_their_own_modes():
    f = bm.find_cmd(30.0, 1.0)
    assert [m for m, i in f if i] == [MODES["find"]] * 2
    assert bm.find_cmd(30.0, 0.5)[1][1] > bm.find_cmd(30.0, 3.0)[1][1]
    t = bm.talk_cmd(-60.0)
    assert sum(1 for _, i in t if i) == 2 and all(m in (0, MODES["talk"]) for m, _ in t)


def test_emulated_belt_plays_the_signature_on_a_mode_change_only():
    ensure_emulator()
    port = 5400 + int(time.time()) % 500
    proc = subprocess.Popen([str(EMU_BIN), str(port)], stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.2)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        body = b"\0\0" * 8

        def send(seq, mode_idx):
            flags = mode_idx << modes.FLAG_MODE_SHIFT
            s.sendto(struct.pack("<2sBBHI", b"MS", 1, flags, seq, 0) + body, ("127.0.0.1", port))
            return list(s.recvfrom(64)[0][14:22])

        quiet = [send(i, 0) for i in range(1, 6)]
        assert not any(any(d) for d in quiet)  # the first packet is not a change
        seen, seq, end = [], 10, time.time() + 1.1
        while time.time() < end:
            seq += 1
            seen.append(send(seq, 2))  # guide -> talk
            time.sleep(0.01)
        lit = {i for d in seen for i in range(8) if d[i]}
        assert lit == {2, 6}  # talk: left and right only
        later = [send(seq + 1 + k, 2) for k in range(5)]
        time.sleep(0.2)
        assert not any(any(d) for d in later[-3:])
    finally:
        proc.terminate()
