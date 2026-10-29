"""Haptic Braille output: turn an answer into a timed cell schedule and play it on a backend.

Backends (all prototypes, none verified on hardware yet):
  EmulatorBraille  cell events as JSON on the server stream, so a screen can light 6 dots
  WristBraille     6 motors, one per dot, driven by firmware/wrist over BLE
  BeltBraille      fallback: each dot is a short pulse on one belt motor, played one after the
                   other. Lower fidelity: a cell takes about 3x longer and the dot positions are
                   serial in time instead of laid out on skin.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from . import braille

log = logging.getLogger(__name__)

# Classic ESP32 38-pin, dot 1..6. Clear of the belt's 13,14,27,26,25,33,32,23, of the strapping
# pins (0,2,5,12,15), flash (6-11), UART0 (1,3) and the input-only pins (34-39). 16 and 17 are
# the PSRAM lines on WROVER modules, free on the WROOM-32 DevKitC the belt uses.
WRIST_PINS = (4, 16, 17, 18, 19, 21)
# Belt motor (sector index, front = 0, clockwise) that plays each dot in the fallback. Dots 1-3
# are the left column of the cell, 4-6 the right, so left motors carry them going down.
BELT_DOT_MOTOR = {1: 0, 2: 7, 3: 6, 4: 1, 5: 2, 6: 3}
BELT_PULSE_MS = 90
BELT_PULSE_GAP_MS = 70
BELT_KEEPALIVE_MS = 100  # the belt falls back to its lost taps if silent for 250 ms
WRIST_MAX_MS = 2000


@dataclass(frozen=True)
class Timing:
    """ERM motors need 50-100 ms to spin up, so a cell is on for a quarter of a second."""

    dot_ms: int = 250
    cell_gap_ms: int = 150
    word_gap_ms: int = 500


DEFAULT_TIMING = Timing()


@dataclass(frozen=True)
class CellEvent:
    i: int
    word: int
    mask: int
    t_ms: int
    dur_ms: int
    rest_ms: int  # silence after this cell, before the next one

    def to_json(self) -> dict:
        return {
            "i": self.i, "word": self.word, "mask": self.mask, "dots": braille.dots_of(self.mask),
            "t_ms": self.t_ms, "dur_ms": self.dur_ms, "rest_ms": self.rest_ms,
        }  # fmt: skip


def build_timeline(
    words: list[list[int]],
    timing: Timing = DEFAULT_TIMING,
    cell_ms: Callable[[int], int] | None = None,
) -> list[CellEvent]:
    """Flatten words of cells into events. `cell_ms(mask)` lets a backend stretch a cell."""
    events: list[CellEvent] = []
    t = 0
    for w, cells in enumerate(words):
        for k, m in enumerate(cells):
            dur = cell_ms(m) if cell_ms else timing.dot_ms
            last_in_word = k == len(cells) - 1
            last = last_in_word and w == len(words) - 1
            rest = 0 if last else (timing.word_gap_ms if last_in_word else timing.cell_gap_ms)
            events.append(CellEvent(len(events), w, m, t, dur, rest))
            t += dur + rest
    return events


def total_ms(events: list[CellEvent]) -> int:
    return events[-1].t_ms + events[-1].dur_ms if events else 0


Sleep = Callable[[float], Awaitable[None]]


class HapticDriver:
    name = "base"
    sleep: Sleep = staticmethod(asyncio.sleep)  # type: ignore[assignment]

    def cell_ms(self, m: int, timing: Timing) -> int:
        return timing.dot_ms

    async def begin(self, text: str, cues: str, events: list[CellEvent]) -> None: ...

    async def cell(self, ev: CellEvent) -> None:
        raise NotImplementedError

    async def rest(self, ms: int) -> None:
        await self.sleep(ms / 1000)

    async def end(self) -> None: ...


async def play(driver: HapticDriver, text: str, cues: str, timing: Timing = DEFAULT_TIMING) -> list:
    """Play `cues` (already terse) start to finish; `text` is the full answer, for display."""
    events = build_timeline(braille.encode(cues), timing, lambda m: driver.cell_ms(m, timing))
    await driver.begin(text, cues, events)
    try:
        for ev in events:
            await driver.cell(ev)
            if ev.rest_ms:
                await driver.rest(ev.rest_ms)
    finally:
        await driver.end()
    return events


class EmulatorBraille(HapticDriver):
    """Events for a screen: one `start` with the whole timeline, a `cell` as each one begins, an
    `end`. Shape documented in docs/braille.md."""

    name = "emulator"

    def __init__(self, emit: Callable[[dict], Awaitable[None]], timing: Timing = DEFAULT_TIMING):
        self.emit = emit
        self.timing = timing

    async def begin(self, text, cues, events) -> None:
        await self.emit(
            {
                "type": "braille", "phase": "start", "device": "emulated wrist band",
                "text": text, "cues": cues, "total_ms": total_ms(events),
                "timing": {"dot_ms": self.timing.dot_ms, "cell_gap_ms": self.timing.cell_gap_ms,
                           "word_gap_ms": self.timing.word_gap_ms},
                "cells": [e.to_json() for e in events],
            }
        )  # fmt: skip

    async def cell(self, ev: CellEvent) -> None:
        await self.emit({"type": "braille", "phase": "cell", **ev.to_json()})
        await self.sleep(ev.dur_ms / 1000)

    async def end(self) -> None:
        await self.emit({"type": "braille", "phase": "end"})


def wrist_packet(m: int, dur_ms: int) -> bytes:
    """The BLE write the wrist firmware takes: 6-bit dot mask, then on-time in ms (u16 LE)."""
    return struct.pack("<BH", m & 0x3F, max(0, min(WRIST_MAX_MS, dur_ms)))


class WristBraille(HapticDriver):
    """The firmware times the buzz itself; we only wait it out so the next cell lines up."""

    name = "wrist"

    def __init__(self, write: Callable[[bytes], Awaitable[None]]):
        self.write = write

    async def cell(self, ev: CellEvent) -> None:
        await self.write(wrist_packet(ev.mask, ev.dur_ms))
        await self.sleep(ev.dur_ms / 1000)

    async def end(self) -> None:
        await self.write(wrist_packet(0, 0))  # all motors off


class BeltBraille(HapticDriver):
    """Dots played one after another as pulses on belt motors. Lower fidelity than the wrist."""

    name = "belt"

    def __init__(
        self, send: Callable[[list[tuple[int, int]], int], None], solid: int, inten: int = 255
    ):
        self.send = send  # (8 x (mode, intensity), flags), same shape as BeltLink.send
        self.solid = solid
        self.inten = inten

    def cell_ms(self, m: int, timing: Timing) -> int:
        n = max(1, len(braille.dots_of(m)))
        return n * (BELT_PULSE_MS + BELT_PULSE_GAP_MS)

    def _frame(self, dot: int | None) -> list[tuple[int, int]]:
        cmd = [(0, 0)] * 8
        if dot is not None:
            cmd[BELT_DOT_MOTOR[dot]] = (self.solid, self.inten)
        return cmd

    async def _hold(self, dot: int | None, ms: int) -> None:
        left = ms
        while left > 0:
            step = min(BELT_KEEPALIVE_MS, left)
            self.send(self._frame(dot), 0)
            await self.sleep(step / 1000)
            left -= step

    async def cell(self, ev: CellEvent) -> None:
        for d in braille.dots_of(ev.mask) or [None]:
            await self._hold(d, BELT_PULSE_MS)
            await self._hold(None, BELT_PULSE_GAP_MS)

    async def rest(self, ms: int) -> None:
        await self._hold(None, ms)

    async def end(self) -> None:
        self.send(self._frame(None), 0)


@dataclass
class Fanout(HapticDriver):
    """Drive several backends in lockstep (for example the wrist band and its on-screen twin)."""

    drivers: list[HapticDriver] = field(default_factory=list)
    name = "fanout"

    def cell_ms(self, m, timing) -> int:
        return max(d.cell_ms(m, timing) for d in self.drivers)

    async def begin(self, text, cues, events) -> None:
        await asyncio.gather(*(d.begin(text, cues, events) for d in self.drivers))

    async def cell(self, ev) -> None:
        await asyncio.gather(*(d.cell(ev) for d in self.drivers))

    async def rest(self, ms) -> None:
        await asyncio.gather(*(d.rest(ms) for d in self.drivers))

    async def end(self) -> None:
        await asyncio.gather(*(d.end() for d in self.drivers))


def ble_writer(address: str) -> Callable[[bytes], Awaitable[None]]:
    """Write to the wrist firmware's Braille characteristic with bleak (optional dependency).
    Prototype: untested against the real board."""
    from bleak import BleakClient  # imported late: only the wrist backend needs it

    client = BleakClient(address)

    async def write(data: bytes) -> None:
        if not client.is_connected:
            await client.connect()
        await client.write_gatt_char(WRIST_CHAR_UUID, data, response=False)

    return write


WRIST_SERVICE_UUID = "6d657461-7369-6768-742d-627261696c6c"
WRIST_CHAR_UUID = "6d657461-7369-6768-742d-627261696c01"


class BrailleOutput:
    """What the server holds: compress an answer, cancel whatever was playing, play the new cue."""

    def __init__(self, driver: HapticDriver, timing: Timing = DEFAULT_TIMING, max_cells: int = 24):
        self.driver = driver
        self.timing = timing
        self.max_cells = max_cells
        self.task: asyncio.Task | None = None

    @property
    def owns_belt(self) -> bool:
        """True while a belt-fallback cue is playing: the server holds back obstacle cues then."""
        return self.task is not None and not self.task.done() and self.driver_uses_belt

    @property
    def driver_uses_belt(self) -> bool:
        d = self.driver
        return isinstance(d, BeltBraille) or (
            isinstance(d, Fanout) and any(isinstance(x, BeltBraille) for x in d.drivers)
        )

    def say(self, text: str) -> str:
        """Start playing `text` as a terse cue and return the cue. Needs a running loop."""
        cues = braille.compress(text, self.max_cells)
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = asyncio.ensure_future(self._run(text, cues))
        return cues

    async def _run(self, text: str, cues: str) -> None:
        try:
            await play(self.driver, text, cues, self.timing)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("braille playback failed")


def make_output(settings, emit: Callable[[dict], Awaitable[None]], belt) -> BrailleOutput | None:
    """Build the output named by settings.braille: off, emulator, wrist or belt. The emulator
    always rides along with the wrist and belt backends so a screen can show the same cells."""
    kind = settings.braille
    if kind == "off":
        return None
    timing = Timing(
        settings.braille_dot_ms, settings.braille_cell_gap_ms, settings.braille_word_gap_ms
    )
    drivers: list[HapticDriver] = [EmulatorBraille(emit, timing)]
    if kind == "wrist":
        drivers.append(WristBraille(ble_writer(settings.braille_addr)))
    elif kind == "belt":
        from .belt import MODES

        drivers.append(
            BeltBraille(lambda cmd, flags: belt.send(cmd, flags, time.monotonic()), MODES["solid"])
        )
    elif kind != "emulator":
        raise ValueError(f"braille must be off, emulator, wrist or belt, not {kind!r}")
    out = drivers[0] if len(drivers) == 1 else Fanout(drivers)
    return BrailleOutput(out, timing, settings.braille_max_cells)
