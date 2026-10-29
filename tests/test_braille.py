import asyncio
import struct

import pytest

from metasight import braille, haptics
from metasight.haptics import (
    BeltBraille,
    BrailleOutput,
    EmulatorBraille,
    Fanout,
    Timing,
    WristBraille,
    build_timeline,
    play,
    total_ms,
    wrist_packet,
)


def pat(word: str) -> list[list[int]]:
    return [braille.dots_of(m) for m in braille.encode_word(word)]


def test_known_letters():
    assert pat("cab") == [[1, 4], [1], [1, 2]]
    assert pat("wxyz") == [[2, 4, 5, 6], [1, 3, 4, 6], [1, 3, 4, 5, 6], [1, 3, 5, 6]]


def test_digits_take_a_number_sign_once_per_run():
    assert pat("42") == [[3, 4, 5, 6], [1, 4, 5], [1, 2]]
    assert pat("3.5") == [
        [3, 4, 5, 6],
        [1, 4],
        [2, 5, 6],
        [1, 5],
    ]  # decimal point keeps the number open
    # a-j after digits would read as digits: letter sign first
    assert pat("4m") == [[3, 4, 5, 6], [1, 4, 5], [1, 3, 4]]
    assert pat("4a") == [[3, 4, 5, 6], [1, 4, 5], [5, 6], [1]]


def test_capitals():
    assert pat("Ab") == [[6], [1], [1, 2]]
    assert pat("AB") == [[6], [6], [1], [1, 2]]  # capital word: two signs, once


def test_punctuation():
    assert pat("hi!") == [[1, 2, 5], [2, 4], [2, 3, 5]]
    assert pat("a.") == [[1], [2, 5, 6]]


def test_every_symbol_has_a_unique_cell_and_roundtrips():
    cells = [*braille.LETTER_MASK.values(), *braille.PUNCT_MASK.values()]
    assert len(set(cells)) == len(cells)
    for text in ["keys 4m ahead", "Room 12, left.", "NASA ok?", "3.5 ft"]:
        assert braille.decode(braille.encode(text)) == text


def test_words_are_kept_apart_and_unknown_characters_dropped():
    assert len(braille.encode("keys  table")) == 2
    assert braille.encode("☃ ab") == [braille.encode_word("ab")]


@pytest.mark.parametrize(
    ("text", "cue"),
    [
        ("Your keys are on the table, 4 meters ahead.", "keys table 4m ahead"),
        ("The door is on your left.", "door left"),
        ("Stairs 2 o'clock, about 3 meters.", "stairs 2oc ~ 3m"),
        ("I don't know.", "?"),
        ("Which object do you mean?", "which?"),
    ],
)
def test_compress(text, cue):
    assert braille.compress(text) == cue


def test_compress_is_shorter_and_capped():
    long = "The wallet is on the kitchen counter next to the laptop, about 2 meters ahead of you"
    cue = braille.compress(long)
    assert len(cue) < len(long) / 2
    capped = braille.compress(long, max_cells=10)
    assert sum(len(c) + 1 for c in braille.encode(capped)) <= 11 and capped.startswith("wallet")


def test_timeline_gaps_and_total():
    t = Timing(dot_ms=250, cell_gap_ms=100, word_gap_ms=400)
    ev = build_timeline(braille.encode("ab c"), t)
    assert [e.t_ms for e in ev] == [0, 350, 1000]
    assert [e.rest_ms for e in ev] == [100, 400, 0]
    assert total_ms(ev) == 1250 and ev[1].word == 0 and ev[2].word == 1


def test_default_cell_time_covers_erm_spinup():
    assert haptics.DEFAULT_TIMING.dot_ms >= 250 > 100


class Clock:
    def __init__(self):
        self.slept: list[float] = []

    async def sleep(self, s):
        self.slept.append(s)


def run(coro):
    return asyncio.run(coro)


def test_emulator_events_and_real_time_total():
    out, clock = [], Clock()

    async def emit(msg):
        out.append(msg)

    drv = EmulatorBraille(emit)
    drv.sleep = clock.sleep
    ev = run(play(drv, "Keys on the table", "ab c"))
    assert [m["phase"] for m in out] == ["start", "cell", "cell", "cell", "end"]
    assert out[0]["total_ms"] == total_ms(ev) and len(out[0]["cells"]) == 3
    assert out[1]["dots"] == [1] and out[1]["type"] == "braille"
    assert sum(clock.slept) * 1000 == pytest.approx(total_ms(ev))


def test_wrist_packet_and_playback():
    assert wrist_packet(0b100101, 250) == struct.pack("<BH", 0b100101, 250)
    assert wrist_packet(0xFF, 99999) == struct.pack("<BH", 0x3F, haptics.WRIST_MAX_MS)
    sent, clock = [], Clock()

    async def write(b):
        sent.append(b)

    drv = WristBraille(write)
    drv.sleep = clock.sleep
    run(play(drv, "", "ab"))
    assert sent == [wrist_packet(1, 250), wrist_packet(3, 250), wrist_packet(0, 0)]
    assert sum(clock.slept) == pytest.approx(0.25 + 0.15 + 0.25)


def test_wrist_pins_do_not_collide_and_are_safe():
    belt = {13, 14, 27, 26, 25, 33, 32, 23}
    bad = {0, 2, 5, 12, 15, 1, 3, *range(6, 12), *range(34, 40)}
    assert len(set(haptics.WRIST_PINS)) == 6
    assert not set(haptics.WRIST_PINS) & (belt | bad)


def test_belt_fallback_pulses_each_dot_and_keeps_the_link_alive():
    frames, clock = [], Clock()
    drv = BeltBraille(lambda cmd, flags: frames.append(list(cmd)), solid=3)
    drv.sleep = clock.sleep
    run(play(drv, "", "b c"))  # b = dots 1,2; c = dots 1,4
    hot = [[i for i, (m, _) in enumerate(f) if m] for f in frames]
    pulses = [h for h in hot if h]
    assert pulses[0] == [haptics.BELT_DOT_MOTOR[1]]
    assert {tuple(p) for p in pulses} == {(haptics.BELT_DOT_MOTOR[d],) for d in (1, 2, 4)}
    assert all(len(p) == 1 for p in pulses)  # one motor at a time: serial, lower fidelity
    assert max(clock.slept) <= haptics.BELT_KEEPALIVE_MS / 1000  # never silent past the watchdog
    assert hot[-1] == []


def test_belt_cells_take_longer_than_wrist_cells():
    t = Timing()
    assert BeltBraille(lambda *_: None, 3).cell_ms(0b111111, t) > 3 * t.dot_ms


def test_fanout_runs_all_backends():
    a, b, clock = [], [], Clock()

    async def ea(m):
        a.append(m)

    async def wb(x):
        b.append(x)

    em, wr = EmulatorBraille(ea), WristBraille(wb)
    em.sleep = wr.sleep = clock.sleep
    run(play(Fanout([em, wr]), "t", "a"))
    assert a[1]["dots"] == [1] and b[0] == wrist_packet(1, 250)


def test_output_cancels_the_previous_cue_and_compresses():
    async def go():
        seen = []

        async def emit(m):
            seen.append(m)

        out = BrailleOutput(EmulatorBraille(emit))
        cue = out.say("Your keys are on the table, 4 meters ahead.")
        await asyncio.sleep(0.01)
        out.say("The door is on your left.")
        await out.task
        return cue, seen

    cue, seen = run(go())
    assert cue == "keys table 4m ahead"
    starts = [m for m in seen if m["phase"] == "start"]
    assert [s["cues"] for s in starts] == ["keys table 4m ahead", "door left"]
    assert seen[-1]["phase"] == "end"


def test_wrist_firmware_core_matches_the_python_packet(tmp_path):
    import subprocess
    from pathlib import Path

    src = tmp_path / "t.c"
    src.write_text(
        '#include <stdio.h>\n#include "braille_core.h"\n'
        "int main(void){ wrist_t w; wrist_init(&w); uint8_t p[3];\n"
        "  int ok = 0; unsigned a, b, c, d, e;\n"
        "  p[0]=0x25; p[1]=250; p[2]=0; ok += wrist_rx(&w,p,3,1000);\n"
        "  a = wrist_tick(&w,1100); b = wrist_tick(&w,1249); c = wrist_tick(&w,1250);\n"
        "  p[0]=0xFF; p[1]=0xFF; p[2]=0xFF; wrist_rx(&w,p,3,5000);\n"
        "  d = wrist_tick(&w,6999); e = wrist_tick(&w,7000);\n"
        '  printf("%d %u %u %u %u %u %d\\n", ok, a, b, c, d, e, wrist_rx(&w,p,2,0)); return 0; }\n'
    )
    exe = tmp_path / "t"
    inc = Path(__file__).parent.parent / "firmware" / "wrist"
    subprocess.run(["cc", "-Wall", "-I", str(inc), "-o", str(exe), str(src)], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
    # on for the packet's 250 ms, then off; 0xFFFF ms is capped at 2000 ms, the mask at 6 bits
    assert out == ["1", "37", "37", "0", "63", "0", "0"]


def test_make_output_picks_backends():
    from metasight.config import Settings

    async def emit(m):
        pass

    s = Settings()
    assert haptics.make_output(s, emit, None) is None
    s.braille = "emulator"
    assert isinstance(haptics.make_output(s, emit, None).driver, EmulatorBraille)
    s.braille = "belt"
    out = haptics.make_output(s, emit, object())
    assert out.driver.drivers[0].device == "belt fallback"
    assert out.driver_uses_belt and not out.owns_belt
    s.braille = "audio"
    out = haptics.make_output(s, emit, object())
    assert out.driver.drivers[0].device == "belt audio stand-in"
    assert not out.driver_uses_belt  # sound only: the belt keeps its obstacle cues
    s.braille = "bogus"
    with pytest.raises(ValueError):
        haptics.make_output(s, emit, None)
