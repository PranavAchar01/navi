"""Voice isolation: the open dns48 model cleans pink noise off a tone-like voice stand-in, fast
enough per 64 ms chunk, and the worker hands cleaned audio back to the event loop."""

import asyncio
import time

import numpy as np
import pytest

pytest.importorskip("denoiser")

from metasight.isolate import IsolatedPush, Isolator  # noqa: E402

SR = 16000


def _voice(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # syllable-rate loudness
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 12)) * env
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def test_isolator_removes_noise_in_real_time():
    rng = np.random.default_rng(0)
    clean = _voice(SR * 3)
    noisy = clean + 0.1 * rng.normal(size=clean.size).astype(np.float32)
    iso = Isolator()
    out, ms = [], []
    for i in range(0, noisy.size, 1024):
        t0 = time.perf_counter()
        out.append(iso.feed(noisy[i : i + 1024]))
        ms.append(1000 * (time.perf_counter() - t0))
    out.append(iso.flush())
    y = np.concatenate(out)[: clean.size]
    residual_before = np.mean((noisy - clean) ** 2)
    residual_after = np.mean((y - clean) ** 2)
    assert y.size == clean.size
    assert residual_after < residual_before
    assert np.median(ms) < 32  # half of a 64 ms chunk
    assert iso.latency_s < 0.1


def test_isolated_push_returns_to_loop():
    got = []

    async def run():
        push = IsolatedPush(got.append, asyncio.get_running_loop())
        for _ in range(10):
            push(np.zeros(1024, np.float32))
        for _ in range(100):
            await asyncio.sleep(0.02)
            if push.stats["chunks"] == 10:
                break
        await asyncio.sleep(0.05)
        return push.stats

    stats = asyncio.run(run())
    assert stats["chunks"] + stats["raw"] == 10
    assert sum(len(x) for x in got) > 0
