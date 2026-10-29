import subprocess
import sys

import pytest

from metasight import band

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS scheduling bands")

PROBE = """
import ctypes, os, threading
from metasight import band
libc = ctypes.CDLL(None)
libc.pthread_self.restype = ctypes.c_void_p
libc.pthread_get_qos_class_np.argtypes = [ctypes.c_void_p] * 3

def qos():
    c, r = ctypes.c_uint(), ctypes.c_int()
    libc.pthread_get_qos_class_np(libc.pthread_self(), ctypes.byref(c), ctypes.byref(r))
    return hex(c.value)

out = [band.split(), os.getpriority(os.PRIO_DARWIN_PROCESS, 0)]
def stop_path():
    band.stop_path_thread()
    out.append(band.low(qos)())
    out.append(qos())
t = threading.Thread(target=stop_path); t.start(); t.join()
print(out)
"""


def test_a_background_launch_keeps_only_the_stop_path_out_of_the_band():
    run = subprocess.run(
        ["taskpolicy", "-b", sys.executable, "-c", PROBE], capture_output=True, text=True
    )
    # Out of the band; the label pass runs in the background class; the thread gets UI back.
    assert run.stdout.strip() == "[True, 0, '0x9', '0x21']", run.stderr


def test_a_normal_launch_is_left_as_it_is():
    assert band.split() is False
    assert band.low(lambda x: x + 1)(1) == 2
