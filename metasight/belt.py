"""The haptic belt link: cues out over UDP to the ESP32 (or its emulator), telemetry back.

The wire format is in firmware/belt/belt_core.h. The belt owns pulse timing and its own
watchdog, so this side only says, per sector, which pattern and how strong; if these packets
stop, the belt falls back to its "lost" taps by itself within 250 ms.
"""

from __future__ import annotations

import asyncio
import base64
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
MODES = {"off": 0, "slow": 1, "fast": 2, "solid": 3, "unknown": 4, "guide": 5, "find": 6, "talk": 7}
NAMES = {v: k for k, v in MODES.items()}
FLAG_DROP = 1
FLAG_FOUND = 2
# The belt's BLE service (firmware/belt/belt.ino): one characteristic, write without response,
# taking the same 26-byte packet as UDP 4210.
BELT_SERVICE_UUID = "6d657461-7369-6768-742d-62656c740000"
BELT_CHAR_UUID = "6d657461-7369-6768-742d-62656c740001"
BELT_BLE_NAME = "metasight-belt"


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


def packet(cmd: list[tuple[int, int]], flags: int, seq: int, t_ms: int) -> bytes:
    """The 26-byte command packet, same for UDP and BLE."""
    body = b"".join(struct.pack("<BB", m, i) for m, i in cmd)
    return struct.pack("<2sBBHI", b"MS", 1, flags, seq, t_ms & 0xFFFFFFFF) + body


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
    """kind: emulator and udp talk to the belt over UDP; ble writes to it over Bluetooth with
    bleak (desk testing; `addr` is a BLE address or empty to find "metasight-belt"); app hands
    each packet to the phone app, which relays it over Bluetooth. ble and app get no telemetry
    back, so the state shown is the command, not the motors' measured duty."""

    def __init__(
        self,
        kind: str,
        addr: str,
        on_state: Callable[[dict], None],
        relay: Callable[[dict], None] | None = None,
    ):
        self.kind = kind
        self.addr = (addr.rsplit(":", 1)[0], int(addr.rsplit(":", 1)[1])) if self._udp else ("", 0)
        self.ble_addr = addr
        self.relay = relay
        self.on_state = on_state
        self.ble_busy = False
        self.ble_client = None
        self.proc: subprocess.Popen | None = None
        self.transport: asyncio.DatagramTransport | None = None
        self.seq = 0
        self.arrived: dict[int, float] = {}  # belt seq -> when its camera frame reached the server
        self.lat: deque[float] = deque(maxlen=60)
        self.applied_seq = -1
        self.last_state = 0.0
        self.last_cmd: list[tuple[int, int]] = [(0, 0)] * 8

    @property
    def _udp(self) -> bool:
        return self.kind in ("emulator", "udp")

    async def start(self) -> None:
        if not self._udp:
            return
        if self.kind == "emulator":
            ensure_emulator()
            self.proc = subprocess.Popen([str(EMU_BIN), str(self.addr[1])])
        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: _Proto(self._telemetry), local_addr=("127.0.0.1", 0)
        )

    def stop(self) -> None:
        if self.ble_client is not None:
            asyncio.ensure_future(self.ble_client.disconnect())
        if self.transport:
            self.transport.close()
        if self.proc:
            self.proc.terminate()

    def send(self, cmd: list[tuple[int, int]], flags: int, frame_arrived: float) -> None:
        if self._udp and not self.transport:
            return
        self.seq = (self.seq + 1) & 0xFFFF
        self.arrived[self.seq] = frame_arrived
        self.arrived.pop((self.seq - 200) & 0xFFFF, None)
        self.last_cmd = cmd
        pkt = packet(cmd, flags, self.seq, int(time.monotonic() * 1000))
        if self.transport:
            self.transport.sendto(pkt, self.addr)
        elif self.kind == "ble":
            if not self.ble_busy:  # a slow write drops frames rather than queueing them
                self.ble_busy = True
                asyncio.ensure_future(self._ble_write(pkt))
        elif self.kind == "app" and self.relay:
            self.relay({"type": "beltcmd", "pkt": base64.b64encode(pkt).decode()})
        if not self._udp:
            self._command_state(flags)

    async def _ble_write(self, pkt: bytes) -> None:
        try:
            if self.ble_client is None or not self.ble_client.is_connected:
                from bleak import BleakClient, BleakScanner  # optional: only desk testing needs it

                target = self.ble_addr
                if not target or target == "scan":
                    target = await BleakScanner.find_device_by_name(BELT_BLE_NAME, timeout=8.0)
                    if target is None:
                        log.warning("belt: no %s found over Bluetooth", BELT_BLE_NAME)
                        return
                self.ble_client = BleakClient(target)
                await self.ble_client.connect()
            await self.ble_client.write_gatt_char(BELT_CHAR_UUID, pkt, response=False)
        except Exception as e:
            log.warning("belt: Bluetooth write failed: %s", e)
            self.ble_client = None
        finally:
            self.ble_busy = False

    def _command_state(self, flags: int) -> None:
        """No telemetry over Bluetooth: report the command itself, at most 30 times a second."""
        now = time.monotonic()
        if now - self.last_state < 1 / 30:
            return
        self.last_state = now
        self.on_state(
            {
                "type": "belt",
                "device": "ESP32 via phone" if self.kind == "app" else "ESP32 (Bluetooth)",
                "lost": False,
                "drop": bool(flags & FLAG_DROP),
                "found": bool(flags & FLAG_FOUND),
                "duty": [i if m in (1, 2, 3, 5) else 0 for m, i in self.last_cmd],
                "modes": [NAMES[m] for m, _ in self.last_cmd],
                "age_ms": 0,
                "latency_ms": None,
            }
        )

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
