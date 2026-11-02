"""Hardware in the loop: simulated walks drive the real belt over the USB cable.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Each held-out walk is first run in hapsim with the
current best belt (panned, four motors, constant-power fold), recording every command it sends
(25 per second). The commands are then replayed in real time, at the same moments, to the board
over USB serial and to the host emulator (firmware/emulator, the same belt_core.h) over UDP, side
by side. Both report back what their eight cells are driving at 50 Hz. Measured:
  - applied: share of commands the board ran (by sequence number in its telemetry)
  - round trip: write to the board's telemetry showing that command (USB, both directions)
  - duty error: per command, the board's mean duty per cell against the emulator's, /255
  - lost: telemetry frames where the board's watchdog fired (no command for 250 ms)
Writes bench/results/hil.json.

    PYTHONPATH=. .venv/bin/python scripts/run_hil.py [--walks 3] [--port /dev/cu.usbserial-0001]
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import serial

from metasight import hapsim as h
from metasight import pan
from metasight.belt import EMU_BIN, ensure_emulator, packet

ROOT = Path(__file__).resolve().parent.parent
EMU_PORT = 4299


def record(seed: int) -> tuple[list[tuple[float, list, int]], h.Result]:
    cmds: list[tuple[float, list, int]] = []
    belt = h.Belt(mapping=pan.DEFAULT, persist=2, scheme="panned", layout="diag4", fold="cp")
    r = h.run(h.make_room(seed), belt, h.DEFAULT_SKIN, seed=0, on_cmd=lambda *c: cmds.append(c))
    return cmds, r


def frames(buf: bytes) -> tuple[list[bytes], bytes]:
    out = []
    while (i := buf.find(b"MT")) >= 0 and len(buf) - i >= 22:
        out.append(buf[i : i + 22])
        buf = buf[i + 22 :]
    return out, buf[-64:]


def replay(cmds, ser: serial.Serial, udp: socket.socket, seq: int) -> tuple[dict, int]:
    """Plays one walk's commands; `seq` carries on from the last walk so old telemetry (frames
    still naming an earlier command) never counts for this one."""
    sent: dict[int, float] = {}
    tel: dict[str, list[tuple[float, int, int, list[int]]]] = {"board": [], "emu": []}
    buf, t0 = b"", time.monotonic()

    def drain():
        nonlocal buf
        now = time.monotonic()
        buf += ser.read(4096)
        got, buf = frames(buf)
        for f in got:
            lost, _, s, _, _ = struct.unpack("<BBHII", f[2:14])
            tel["board"].append((now, s, lost, list(f[14:22])))
        while True:
            try:
                f = udp.recv(64)
            except BlockingIOError:
                break
            if len(f) == 22 and f[:2] == b"MT":
                lost, _, s, _, _ = struct.unpack("<BBHII", f[2:14])
                tel["emu"].append((now, s, lost, list(f[14:22])))

    for t, cmd, flags in cmds:
        while time.monotonic() - t0 < t:
            drain()
            time.sleep(0.002)
        seq = (seq + 1) & 0xFFFF
        pkt = packet(cmd, flags, seq, int(time.monotonic() * 1000))
        ser.write(pkt)
        udp.sendto(pkt, ("127.0.0.1", EMU_PORT))
        sent[seq] = time.monotonic()
    last_send = time.monotonic()
    end = last_send + 0.3
    while time.monotonic() < end:
        drain()
        time.sleep(0.002)

    for k in tel:  # only frames from this walk: the board names one of its commands
        tel[k] = [f for f in tel[k] if f[1] in sent]
    first: dict[int, float] = {}
    for now, s, _, _ in tel["board"]:
        first.setdefault(s, now)
    rtt = [1000 * (first[s] - sent[s]) for s in sent if s in first]
    lost_at = [round(now - t0, 2) for now, _, lost, _ in tel["board"] if lost and now <= last_send]

    def mean_duty(rows):
        by: dict[int, list[list[int]]] = {}
        for _, s, _, d in rows:
            by.setdefault(s, []).append(d)
        return {s: np.mean(v, axis=0) for s, v in by.items()}

    b, e = mean_duty(tel["board"]), mean_duty(tel["emu"])
    both = [s for s in b if s in e]
    err = [float(np.abs(b[s] - e[s]).mean() / 255) for s in both]
    return {
        "commands": len(sent),
        "applied": len(first) / max(1, len(sent)),
        "rtt_ms_median": statistics.median(rtt) if rtt else None,
        "rtt_ms_p95": float(np.percentile(rtt, 95)) if rtt else None,
        "duty_err_mean": statistics.mean(err) if err else None,
        "duty_err_max": max(err) if err else None,
        "lost_frames": len(lost_at),
        "lost_span_s": [lost_at[0], lost_at[-1]] if lost_at else None,
        "board_frames": len(tel["board"]),
    }, seq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", type=int, default=3)
    ap.add_argument("--port", default="/dev/cu.usbserial-0001")
    a = ap.parse_args()
    ensure_emulator()
    emu = subprocess.Popen([str(EMU_BIN), str(EMU_PORT)])
    ser = serial.Serial(a.port, 115200, timeout=0)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.setblocking(False)
    out, seq = {}, 0
    try:
        time.sleep(0.5)
        ser.reset_input_buffer()
        for seed in range(1000, 1000 + a.walks):  # the held-out rooms
            cmds, r = record(seed)
            m, seq = replay(cmds, ser, udp, seq)
            m.update(sim_success=r.success, sim_time_s=round(r.time_s, 1))
            out[str(seed)] = m
            print(
                f"walk {seed}: {m['commands']} commands over {r.time_s:.0f} s, "
                f"applied {m['applied']:.3f}, round trip {m['rtt_ms_median']:.1f} ms "
                f"(p95 {m['rtt_ms_p95']:.1f}), duty error {m['duty_err_mean']:.3f} "
                f"(max {m['duty_err_max']:.3f}), lost frames {m['lost_frames']}",
                flush=True,
            )
    finally:
        ser.close()  # the board's watchdog takes over: lost-link taps
        emu.terminate()
    (ROOT / "bench" / "results" / "hil.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
