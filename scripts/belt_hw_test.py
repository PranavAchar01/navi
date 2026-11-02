"""Hardware test for the real belt over Bluetooth from this Mac (about 40 s, belt firmware flashed).

    .venv/bin/python scripts/belt_hw_test.py

Finds "metasight-belt", then: each of the eight directions solid for 1.2 s, front then clockwise
(four motors on the diagonals stand in for the eight); every mode on the front; all at once;
then it goes silent without disconnecting so the 250 ms watchdog
plays the lost-link taps. Commands go out at 20 Hz, like the server. The board sends nothing
back over Bluetooth, so the motors themselves are the result: each step is printed as it plays.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from bleak import BleakClient, BleakScanner  # noqa: E402

from metasight.belt import BELT_BLE_NAME, BELT_CHAR_UUID, MODES, packet  # noqa: E402

POSITIONS = [
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
]
# which of the four diagonal motors each direction should buzz (firmware/belt/belt.ino)
FELT = [
    "front-left + front-right (D4 + D26)",
    "front-right (D26)",
    "front-right + back-right (D26 + D14)",
    "back-right (D14)",
    "back-right + back-left (D14 + D13)",
    "back-left (D13)",
    "back-left + front-left (D13 + D4)",
    "front-left (D4)",
]


async def hold(
    client: BleakClient, cmd: list[tuple[int, int]], secs: float, seq: list[int]
) -> None:
    end = time.monotonic() + secs
    while time.monotonic() < end:
        seq[0] = (seq[0] + 1) & 0xFFFF
        pkt = packet(cmd, 0, seq[0], int(time.monotonic() * 1000))
        await client.write_gatt_char(BELT_CHAR_UUID, pkt, response=False)
        await asyncio.sleep(0.05)


async def main() -> int:
    print(f"scanning for {BELT_BLE_NAME} ...")
    dev = await BleakScanner.find_device_by_name(BELT_BLE_NAME, timeout=10.0)
    if dev is None:
        print(
            f"FAIL no {BELT_BLE_NAME} advertising: is the belt firmware flashed and the board on?"
        )
        return 1
    async with BleakClient(dev) as client:
        print(f"connected {dev.address}")
        seq = [0]
        off = [(MODES["off"], 0)] * 8
        await hold(client, off, 1.0, seq)
        for i, name in enumerate(POSITIONS):
            print(f"{name}: expect {FELT[i]}")
            cmd = list(off)
            cmd[i] = (MODES["solid"], 255)
            await hold(client, cmd, 1.2, seq)
            await hold(client, off, 1.0, seq)
        for mode in ("slow", "fast", "solid", "unknown", "guide", "find", "talk"):
            print(f"front motor: {mode}")
            cmd = list(off)
            cmd[0] = (MODES[mode], 255)
            await hold(client, cmd, 2.0, seq)
            await hold(client, off, 0.6, seq)
        print("all directions solid")
        await hold(client, [(MODES["solid"], 255)] * 8, 1.0, seq)
        await hold(client, off, 0.5, seq)
        print("going silent (still connected): the watchdog should play the lost-link taps")
        await asyncio.sleep(3.0)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
