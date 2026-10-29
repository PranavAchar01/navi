"""What the belt feels like in each activity mode (modes.py: Guide, Find, Talk, Explore).

Standing cues, one per mode, none of them like an obstacle cue (obstacles are continuous or
125 to 200 ms pulses; these are taps):
  Guide    "guide": two soft taps a second on the two motors nearest the next waypoint
           (pan.guide_cmd). Faint on the front motor alone when on track; firmer round a turn.
           Stop (solid) silences it everywhere (pan.with_guide).
  Find     "find": 40 ms ticks, 2.5 a second, on the motor(s) toward the item; intensity rises
           with closeness (110 at 4 m up to 255), so nearer means stronger, as the phone's Find
           haptic gets quicker. In view or not, the bearing comes from the map.
  Talk     "talk": a burst of three soft taps every 1.5 s on the two motors toward whoever is
           speaking (replaces the slow pulse the speaker cue used, which sounded like a far
           obstacle). Obstacle cues at a third less strength (modes.PROFILES).
  Explore  no standing cue: the description is spoken. Obstacle cues at 85%.

Mode-change signature: when the mode bits in the packet flags change the belt itself plays a
1.2 s pattern (firmware belt_signature) so a mode switch is felt before it is heard:
  Guide    two taps, then a firm hold, on the front motor                 "ahead"
  Find     a sweep left across the front, to the right, and back          "scanning"
  Talk     left, right, left, right short taps                            "two voices"
  Explore  opposite pairs in turn: front/back, front-right/back-left, ... "all around"
`signature` mirrors the firmware for tests and for rendering to sound.
"""

from __future__ import annotations

from .belt import MODES
from .modes import ALL
from .pan import N, pan_gains

SIGNATURE_S = 1.2
SIGNATURE_DUTY = 180


def signature_duty(mode: str, t_ms: int) -> list[int]:
    """Motor duty (0..255) t_ms after a switch into `mode`; the same as belt_core.h."""
    d = [0] * N
    k = ALL.index(mode)
    if t_ms >= 1200:
        return d
    if k == 0:
        if t_ms < 80 or 160 <= t_ms < 240 or 400 <= t_ms < 800:
            d[0] = SIGNATURE_DUTY
    elif k == 1:
        d[(6, 7, 0, 1, 2, 2, 1, 0, 7, 6)[t_ms // 120]] = SIGNATURE_DUTY
    elif k == 2:
        if t_ms < 100 or 400 <= t_ms < 500:
            d[6] = SIGNATURE_DUTY
        if 200 <= t_ms < 300 or 600 <= t_ms < 700:
            d[2] = SIGNATURE_DUTY
    else:
        step = t_ms // 300
        if step < 4 and t_ms % 300 < 200:
            d[step] = d[step + 4] = SIGNATURE_DUTY
    return d


def find_cmd(bearing_deg: float, dist_m: float) -> list[tuple[int, int]]:
    """The Find cue: ticks on the two motors toward the item, panned, stronger when closer."""
    cmd = [(MODES["off"], 0)] * N
    amp = 110 + 145 * max(0.0, min(1.0, 1 - dist_m / 4))
    a, ga, b, gb = pan_gains(bearing_deg, "linear")
    for m, g in ((a, ga), (b, gb)):
        if g > 0.05:
            cmd[m] = (MODES["find"], int(round(amp * g)))
    return cmd


def talk_cmd(bearing_deg: float, intensity: int = 150) -> list[tuple[int, int]]:
    """The Talk cue toward a speaker at `bearing_deg` (positive right)."""
    cmd = [(MODES["off"], 0)] * N
    a, ga, b, gb = pan_gains(bearing_deg, "linear")
    for m, g in ((a, ga), (b, gb)):
        if g > 0.05:
            cmd[m] = (MODES["talk"], int(round(intensity * g)))
    return cmd
