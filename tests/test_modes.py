from metasight.config import Settings
from metasight.memory import MemoryEngine


def test_outside_mode_uses_smaller_faster_windows_and_a_horizon():
    eng = MemoryEngine(Settings(), runner=None, finder=None)
    assert eng.mode == "home" and eng.s.window == 8 and eng.s.horizon_m == 0.0
    eng.set_mode("outside")
    assert eng.s.window == 4 and eng.s.overlap == 1 and eng.s.horizon_m == 25.0
    assert eng.s.keyframe_hz > eng.base.keyframe_hz
    eng.set_mode("home")
    assert eng.s.window == 8
