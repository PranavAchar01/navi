"""Drive the real belt over the USB cable (belt firmware flashed), with its telemetry as proof.

    .venv/bin/python scripts/belt_usb.py walk              eight directions, front then clockwise
    .venv/bin/python scripts/belt_usb.py hold front-right  one direction solid until Ctrl-C
    .venv/bin/python scripts/belt_usb.py hold right --secs 300  both right motors, 5 minutes
    .venv/bin/python scripts/belt_usb.py hold all          every direction at once
    .venv/bin/python scripts/belt_usb.py quiz --n 24       blind direction test (below)

quiz: a random direction buzzes for 1.2 s; type where you felt it on the number pad layout
(8 front, 9 front-right, 6 right, 3 back-right, 2 back, 1 back-left, 4 left, 7 front-left).
Every direction comes up equally often. Prints exact and within-one-step accuracy and writes
bench/results/felt-quiz-<time>.json. A human number, unlike the sim: one person, one sitting.

Commands go out at 20 Hz like the server. For each step it prints what the board reports it is
driving (its 8-cell duty, then the four motor outputs after the constant-power fold). The
telemetry proves the board took the command; whether a motor buzzes is still for the hand.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run from anywhere

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


KEYS = {"8": 0, "9": 1, "6": 2, "3": 3, "2": 4, "1": 5, "4": 6, "7": 7}


def quiz(b: Belt, n: int) -> None:
    on: list[int] = []
    stop = threading.Event()

    def keep():  # off packets while he answers, or the watchdog would play the lost taps
        while not stop.is_set():
            b.send(on)
            time.sleep(0.05)

    threading.Thread(target=keep, daemon=True).start()
    order = [d for _ in range(math.ceil(n / 8)) for d in range(8)][:n]
    random.shuffle(order)
    rows = []
    try:
        for k, d in enumerate(order, 1):
            input(f"[{k}/{n}] Enter to buzz ")
            on[:] = [d]
            time.sleep(1.2)
            on[:] = []
            while (ans := input("  where? (number pad) ").strip()) not in KEYS:
                print("  8 front, 9 front-right, 6 right, 3 back-right, 2 back, 1 back-left, ...")
            rows.append({"played": DIRS[d], "felt": DIRS[KEYS[ans]]})
    finally:
        stop.set()
    off = [(DIRS.index(r["felt"]) - DIRS.index(r["played"])) % 8 for r in rows]
    exact = sum(o == 0 for o in off) / len(rows)
    near = sum(o in (0, 1, 7) for o in off) / len(rows)
    print(f"exact {exact:.2f}, within one step {near:.2f} over {len(rows)} trials")
    for name in DIRS:
        got = [r["felt"] for r in rows if r["played"] == name]
        print(f"  {name:12s} felt as {', '.join(got)}")
    out = ROOT / "bench" / "results" / f"felt-quiz-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"exact": exact, "within_one": near, "trials": rows}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["walk", "hold", "quiz"])
    ap.add_argument("direction", nargs="?", default="front", choices=DIRS + ["all"])
    ap.add_argument("--port", default="/dev/cu.usbserial-0001")
    ap.add_argument("--secs", type=float, default=math.inf, help="hold: stop after this long")
    ap.add_argument("--n", type=int, default=24, help="quiz: trials")
    a = ap.parse_args()
    b = Belt(a.port)
    if a.what == "quiz":
        quiz(b, a.n)
        return
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
