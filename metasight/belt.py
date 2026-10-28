"""The haptic belt link: cues out over UDP to the ESP32 (or its emulator), telemetry back.

The wire format is in firmware/belt/belt_core.h. The belt owns pulse timing and its own
watchdog, so this side only says, per sector, which pattern and how strong; if these packets
stop, the belt falls back to its "lost" taps by itself within 250 ms.
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import struct
import subprocess
import time
from collections import deque
from collections.abc import Callable

from .config import ROOT
from .geometry import cue_for_distance

log = logging.getLogger(__name__)

FIRMWARE = ROOT / "firmware"
EMU_SRC = FIRMWARE / "emulator" / "emu.c"
EMU_BIN = FIRMWARE / "emulator" / "belt-emu"
MODES = {"off": 0, "slow": 1, "fast": 2, "solid": 3, "unknown": 4, "guide": 5}
NAMES = {v: k for k, v in MODES.items()}
FLAG_DROP = 1
FLAG_FOUND = 2


def plan(
    sectors: list[float | None],
    known: list[bool],
    path_m: float,
    ttc_s: float,
    drop_m: float,
) -> tuple[list[tuple[int, int]], int]:
    """Per-sector (mode, intensity 0..255) plus flags, from distances in metres.

    Front uses the nearer of its wedge and the walking path. Closing fast (time to contact under
    1.2 s) raises the front cue one step, so walking at something warns earlier than standing
    beside it. A sector with no trusted data is "unknown": no buzz, and the UI says so.
    """
    out = []
    for i, (d, ok) in enumerate(zip(sectors, known, strict=True)):
        if not ok:
            out.append((MODES["unknown"], 0))
            continue
        d = math.inf if d is None else d
        if i == 0:
            d = min(d, path_m)
        cue = cue_for_distance(d)
        mode = MODES[cue.mode]
        inten = cue.intensity
        if i == 0 and ttc_s < 1.2 and d < 2.5:
            mode = min(MODES["solid"], max(mode, MODES["slow"]) + 1)
            inten = max(inten, 0.8)
        out.append((mode, int(round(255 * min(1.0, inten)))))
    flags = FLAG_DROP if drop_m < 1.5 else 0
    return out, flags


def guide(cmd: list[tuple[int, int]], bearing_deg: float, dist_m: float) -> None:
    """Point the belt at a remembered item: the motor facing it plays the guide taps, from soft
    at 4 m to strong up close. It takes over the "slow" cue (something 1 to 2 m away), which
    in a small room is on most of the time; anything nearer than 1 m still wins."""
    i = round(bearing_deg / 45) % 8
    if cmd[i][0] in (MODES["off"], MODES["slow"]):
        cmd[i] = (MODES["guide"], int(round(110 + 145 * max(0.0, min(1.0, 1 - dist_m / 4)))))


def ensure_emulator() -> None:
    """Build the emulator from the firmware sources when it is missing or older than them."""
    srcs = [EMU_SRC, FIRMWARE / "belt" / "belt_core.h"]
    if EMU_BIN.exists() and EMU_BIN.stat().st_mtime >= max(p.stat().st_mtime for p in srcs):
        return
    subprocess.run(["cc", "-O2", "-Wall", "-o", str(EMU_BIN), str(EMU_SRC)], check=True)


class _Proto(asyncio.DatagramProtocol):
    def __init__(self, on_packet: Callable[[bytes], None]):
        self.on_packet = on_packet

    def datagram_received(self, data: bytes, addr) -> None:
        self.on_packet(data)


class BeltLink:
    def __init__(self, kind: str, addr: str, on_state: Callable[[dict], None]):
        self.kind = kind
        host, port = addr.rsplit(":", 1)
        self.addr = (host, int(port))
        self.on_state = on_state
        self.proc: subprocess.Popen | None = None
        self.transport: asyncio.DatagramTransport | None = None
        self.seq = 0
        self.arrived: dict[int, float] = {}  # belt seq -> when its camera frame reached the server
        self.lat: deque[float] = deque(maxlen=60)
        self.applied_seq = -1
        self.last_state = 0.0
        self.last_cmd: list[tuple[int, int]] = [(0, 0)] * 8

    async def start(self) -> None:
        if self.kind == "emulator":
            ensure_emulator()
            self.proc = subprocess.Popen([str(EMU_BIN), str(self.addr[1])])
        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: _Proto(self._telemetry), local_addr=("127.0.0.1", 0)
        )

    def stop(self) -> None:
        if self.transport:
            self.transport.close()
        if self.proc:
            self.proc.terminate()

    def send(self, cmd: list[tuple[int, int]], flags: int, frame_arrived: float) -> None:
        if not self.transport:
            return
        self.seq = (self.seq + 1) & 0xFFFF
        self.arrived[self.seq] = frame_arrived
        self.arrived.pop((self.seq - 200) & 0xFFFF, None)
        self.last_cmd = cmd
        body = b"".join(struct.pack("<BB", m, i) for m, i in cmd)
        t_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
        pkt = struct.pack("<2sBBHI", b"MS", 1, flags, self.seq, t_ms) + body
        self.transport.sendto(pkt, self.addr)

    def _telemetry(self, data: bytes) -> None:
        if len(data) != 22 or data[:2] != b"MT":
            return
        lost, flags, seq, _echo, age = struct.unpack("<BBHII", data[2:14])
        duty = list(data[14:22])
        now = time.monotonic()
        if seq != self.applied_seq and seq in self.arrived:
            # Camera frame at the server -> the belt's motors running that command.
            self.lat.append(1000 * (now - self.arrived[seq]))
            self.applied_seq = seq
        if now - self.last_state < 1 / 30:
            return
        self.last_state = now
        self.on_state(
            {
                "type": "belt",
                "device": "ESP32 emulator" if self.kind == "emulator" else "ESP32",
                "lost": bool(lost),
                "drop": bool(flags & FLAG_DROP),
                "found": bool(flags & FLAG_FOUND),
                "duty": duty,
                "modes": [NAMES[m] for m, _ in self.last_cmd],
                "age_ms": None if age == 0xFFFFFFFF else age,
                "latency_ms": round(statistics.median(self.lat), 1) if self.lat else None,
            }
        )
