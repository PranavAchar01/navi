"""The stop path keeps its priority when the server is started in the background band.

The morning check and the replays start the server under `nice -n 19 taskpolicy -b` (the
heavy-job rule). That band slows plain CPU work 5 to 13 times (docs/perf.md), and the reflex
with it. `split()` takes the process out of the band and puts back everything but the stop path:
the map worker and its solver, the model and review threads, and the label pass after the buzz.
A server started at normal priority (the live service) keeps its map worker as it is: there the
worker in the band slowed map windows from 8.9 to 14.2 s for little gain on the buzz.

Threads use the QoS classes (sys/qos.h): once a thread has been put in the band with
setpriority, macOS refuses to give it a QoS class again, so only processes use setpriority.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable
from functools import wraps

_on = False
_DARWIN = hasattr(os, "PRIO_DARWIN_BG")
_UI, _BG = 0x21, 0x09  # QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_BACKGROUND
_libc = ctypes.CDLL(None) if _DARWIN else None


def split() -> bool:
    """If started in the background band, leave it and keep it for the rest (macOS only)."""
    global _on
    if not _DARWIN or os.getpriority(os.PRIO_DARWIN_PROCESS, 0) == 0:
        return False
    os.setpriority(os.PRIO_DARWIN_PROCESS, 0, 0)
    _on = True
    return True


def stop_path_thread() -> None:
    """The calling thread is on the stop path: the user-interactive class, in any band."""
    if _libc:
        _libc.pthread_set_qos_class_self_np(_UI, 0)


def low_process(pid: int) -> None:
    """Back into the band: a child process (it passes the band on to its own children)."""
    if _on:
        os.setpriority(os.PRIO_DARWIN_PROCESS, pid, os.PRIO_DARWIN_BG)


def low_thread() -> None:
    """Back into the band: the calling thread, for good (a pool's initializer)."""
    if _on:
        _libc.pthread_set_qos_class_self_np(_BG, 0)


def low[T](fn: Callable[..., T]) -> Callable[..., T]:
    """`fn` runs in the band, on the stop path's thread, which gets its class back after."""

    @wraps(fn)
    def run(*a, **k) -> T:
        if not _on:
            return fn(*a, **k)
        _libc.pthread_set_qos_class_self_np(_BG, 0)
        try:
            return fn(*a, **k)
        finally:
            _libc.pthread_set_qos_class_self_np(_UI, 0)

    return run
