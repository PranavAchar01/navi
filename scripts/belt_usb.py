"""Drive the real belt over the USB cable (belt firmware flashed), with its telemetry as proof.

    .venv/bin/python scripts/belt_usb.py walk              eight directions, front then clockwise
    .venv/bin/python scripts/belt_usb.py hold front-right  one direction solid until Ctrl-C
    .venv/bin/python scripts/belt_usb.py hold right --secs 300  both right motors, 5 minutes
    .venv/bin/python scripts/belt_usb.py hold all          every direction at once

Commands go out at 20 Hz like the server. For each step it prints what the board reports it is
driving (its 8-cell duty, then the four motor outputs after the constant-power fold). The
telemetry proves the board took the command; whether a motor buzzes is still for the hand.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

import serial  # noqa: E402

from metasight.belt import MODES, packet  # noqa: E402

DIRS = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
MOTORS = ["D26 front-right", "D14 back-right", "D13 back-left", "D4 front-left"]


def fold(duty: list[int]) -> list[int]:
    """firmware/belt/belt.ino: each motor plays its cell and 1/sqrt(2) of the two beside it."""
    out = []
    for m in range(4):
        c = 2 * m + 1
        a, lt, rt = duty[c], duty[c - 1], duty[(c + 1) % 8]
        out.append(min(255, round(math.sqrt(a * a + 0.5 * (lt * lt + rt * rt)))))
    return out


class Belt:
    def __init__(self, port: str):
        self.s = serial.Serial(port, 115200, timeout=0)
        self.seq, self.buf, self.duty = 0, b"", [0] * 8
        time.sleep(0.3)  # opening the port can reset the board on some cables
        self.s.reset_input_buffer()

    def send(self, on: list[int]) -> None:
        self.seq = (self.seq + 1) & 0xFFFF
        cmd = [(MODES["solid"], 255) if c in on else (MODES["off"], 0) for c in range(8)]
        self.s.write(packet(cmd, 0, self.seq, int(time.monotonic() * 1000)))
        self.buf += self.s.read(4096)
        while (i := self.buf.find(b"MT")) >= 0 and len(self.buf) - i >= 22:
            f = self.buf[i : i + 22]
            self.duty = list(f[14:22])
            self.buf = self.buf[i + 22 :]

    def play(self, on: list[int], secs: float) -> None:
        end = time.monotonic() + secs
        while time.monotonic() < end:
            self.send(on)
            time.sleep(0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["walk", "hold"])
    ap.add_argument("direction", nargs="?", default="front", choices=DIRS + ["all"])
    ap.add_argument("--port", default="/dev/cu.usbserial-0001")
    ap.add_argument("--secs", type=float, default=math.inf, help="hold: stop after this long")
    a = ap.parse_args()
    b = Belt(a.port)
    if a.what == "hold":
        on = list(range(8)) if a.direction == "all" else [DIRS.index(a.direction)]
        print(f"holding {a.direction}; Ctrl-C to stop")
        end = time.monotonic() + a.secs
        try:
            while time.monotonic() < end:
                b.play(on, 1.0)
                print(
                    f"  board duty {b.duty}  motors {dict(zip(MOTORS, fold(b.duty), strict=True))}"
                )
        except KeyboardInterrupt:
            pass
        b.play([], 0.3)
        return
    for d, name in enumerate(DIRS):
        b.play([d], 1.5)
        live = [m for m, v in zip(MOTORS, fold(b.duty), strict=True) if v >= 26]
        print(f"{name:12s} board duty {b.duty}  motors on: {', '.join(live) or 'none'}")
        b.play([], 1.0)


if __name__ == "__main__":
    main()
