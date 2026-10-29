"""Belt feature rows in one run, no server needed (about 10 s): the reflex cue reaches the belt
emulator, Persist(2) is wired and filters a one-frame glitch, each mode's signature plays in the
emulated firmware, and Stop silences Guide.

    .venv/bin/python scripts/belt_check.py

Prints one PASS/FAIL line per check and exits 1 on any FAIL. Importable: `run() -> list[tuple]`
of (key, ok, evidence).
"""

from __future__ import annotations

import math
import random
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from metasight import belt_modes as bm  # noqa: E402
from metasight import hapsim, modes, pan  # noqa: E402
from metasight.belt import EMU_BIN, MODES, ensure_emulator, plan  # noqa: E402
from metasight.server import Server  # noqa: E402


def _emulator():
    ensure_emulator()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    proc = subprocess.Popen([str(EMU_BIN), str(port)], stderr=subprocess.DEVNULL)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.3)
    return proc, s, port


def _send(s, port, seq, cmd, flags=0):
    body = b"".join(struct.pack("<BB", m, i) for m, i in cmd)
    s.sendto(struct.pack("<2sBBHI", b"MS", 1, flags, seq, 0) + body, ("127.0.0.1", port))
    return list(s.recvfrom(64)[0][14:22])


def _ready(s, port, seq0=1):
    """Send until the emulator answers; returns the next seq."""
    quiet = [(0, 0)] * 8
    for k in range(60):
        try:
            _send(s, port, seq0 + k, quiet)
            return seq0 + k + 1
        except TimeoutError:
            continue
    raise RuntimeError("emulator never answered")


def reflex_reaches_belt():
    proc, s, port = _emulator()
    try:
        seq = _ready(s, port)
        cmd, _ = plan([0.3] + [math.inf] * 7, [True] * 8, math.inf, math.inf, math.inf)
        duty = []
        for k in range(8):
            duty = _send(s, port, seq + k, cmd)
            time.sleep(0.02)
        ok = duty[0] > 0 and not any(duty[1:])
        return ok, f"obstacle at 0.3 m ahead -> emulated motor duty {duty}"
    finally:
        proc.terminate()


def persist_on():
    wired = "self.persist = Persist(2)" in open(Server.__init__.__code__.co_filename).read() and (
        "live = self.persist(res.sectors)" in open(Server.__init__.__code__.co_filename).read()
    )
    f = pan.Persist(2)
    far, near = [math.inf] * 8, [0.4] * 8
    f(far)
    glitch = f(near)  # one near frame after far: not yet
    real = f(near)  # second near frame: now it counts
    ok = wired and glitch == far and real == near
    return (
        ok,
        f"server wiring {wired}; one-frame glitch passes {glitch != far}; "
        f"2 frames pass {real == near}",
    )


def signatures_play():
    proc, s, port = _emulator()
    try:
        seq = _ready(s, port)
        cur, bad, notes = 0, [], []
        for target in (2, 1, 3, 0):  # guide->talk->find->explore->guide
            for _ in range(5):
                seq += 1
                _send(s, port, seq, [(0, 0)] * 8, cur << modes.FLAG_MODE_SHIFT)
            lit, end = set(), time.time() + 1.15
            while time.time() < end:
                seq += 1
                d = _send(s, port, seq, [(0, 0)] * 8, target << modes.FLAG_MODE_SHIFT)
                lit |= {i for i in range(8) if d[i]}
                time.sleep(0.01)
            want = {
                i
                for t in range(0, 1200, 10)
                for i, v in enumerate(bm.signature_duty(modes.ALL[target], t))
                if v
            }
            notes.append(f"{modes.ALL[target]} motors {sorted(lit)}")
            if lit != want:
                bad.append(
                    f"{modes.ALL[target]}: emulator {sorted(lit)} vs expected {sorted(want)}"
                )
            for _ in range(5):  # let it settle, then confirm silence
                seq += 1
                _send(s, port, seq, [(0, 0)] * 8, target << modes.FLAG_MODE_SHIFT)
            time.sleep(0.2)
            cur = target
        return not bad, "; ".join(bad or notes)
    finally:
        proc.terminate()


def stop_silences_guide():
    rng = random.Random(1)
    bad = 0
    for _ in range(2000):
        cmd = [(rng.choice([0, 1, 2, 3, 4, 5]), rng.randint(0, 255)) for _ in range(8)]
        if not any(m == MODES["solid"] for m, _ in cmd):
            cmd[rng.randrange(8)] = (MODES["solid"], 255)
        out, shown = pan.with_guide(cmd, rng.uniform(-180, 180))
        bad += out != cmd or shown
    belt = hapsim.Belt(scheme="sector", guide="on", persist=2, planner=True)
    r = hapsim.run(hapsim.make_room(1003), belt, hapsim.Skin(nav_drift_deg=12.0), seed=0)
    ok = bad == 0 and r.stop_violations == 0 and r.stop_overridden == r.stop_frames
    return ok, (
        f"2000 random commands with a solid motor: {bad} changed; sim walk to goal "
        f"{'reached' if r.success else 'not reached'}, {r.stop_frames} stop frames, "
        f"{r.stop_overridden} silenced Guide, {r.stop_violations} violations"
    )


def priors_do_not_add_misses():
    """Skipped until metasight-compare's map priors land: then this scores the ARKit walk with and
    without priors (scripts/score_labels.py) and fails if misses inside 1 m get worse."""
    return True, "SKIP, blocked on metasight-compare's priors commit (no priors scheme yet)"


CHECKS = [
    ("belt_reflex", "Reflex cue reaches the belt emulator", reflex_reaches_belt),
    ("belt_persist", "Persist(2) wired and filtering", persist_on),
    ("belt_signatures", "Mode signature plays for each mode", signatures_play),
    ("belt_stop_guide", "Stop silences Guide", stop_silences_guide),
    ("belt_priors_labels", "Priors do not add misses inside 1 m", priors_do_not_add_misses),
]


def run() -> list[tuple[str, bool, str]]:
    out = []
    for key, name, fn in CHECKS:
        try:
            ok, ev = fn()
        except Exception as e:  # a crashed check is a FAIL with the reason
            ok, ev = False, f"{type(e).__name__}: {e}"
        out.append((key, ok, f"{name}: {ev}"))
    return out


if __name__ == "__main__":
    res = run()
    for key, ok, ev in res:
        print("PASS" if ok else "FAIL", key, ev)
    sys.exit(0 if all(ok for _, ok, _ in res) else 1)
