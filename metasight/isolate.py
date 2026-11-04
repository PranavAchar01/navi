"""Voice isolation: clean the wearer's voice out of room noise before Transcribe hears it.

The model is Meta's open-weight speech enhancer (facebookresearch/denoiser, the causal Demucs
"dns48", 18.9M parameters, trained on the DNS challenge set). It runs on this Mac's CPU in
streaming mode, 16 kHz in and out, which is the rate the microphone chunks already arrive at.

It sits on the talk path only. Chunks are handed to one worker thread, so the event loop that
sends belt cues never waits on it; the cleaned audio is passed back to the loop for Transcribe.
If the worker falls more than MAX_BACKLOG chunks behind, chunks go to Transcribe raw instead of
queueing: a late question is worse than a slightly noisier one.

Install: `uv pip install --no-deps denoiser julius` (the package's own pins are for its training
scripts; inference needs only torch, torchaudio and julius). Weights download once, 72 MB.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

import numpy as np

MAX_BACKLOG = 8  # 64 ms chunks: half a second behind and isolation steps aside
DRY = 0.05  # a little of the raw signal back in keeps consonants that the model over-cuts


class Isolator:
    """Streaming dns48. `feed` takes and returns 16 kHz mono float32; output trails input by
    `latency_s` (the model's look-ahead), and may be empty for the first chunks."""

    def __init__(self, dry: float = DRY):
        import torch
        from denoiser import pretrained
        from denoiser.demucs import DemucsStreamer

        self._torch = torch
        model = pretrained.dns48().eval()
        self.streamer = DemucsStreamer(model, dry=dry, num_frames=1)
        self.latency_s = self.streamer.total_length / model.sample_rate

    def feed(self, x: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            out = self.streamer.feed(self._torch.from_numpy(x.astype(np.float32))[None])
        return out[0].numpy()

    def flush(self) -> np.ndarray:
        with self._torch.no_grad():
            return self.streamer.flush()[0].numpy()


class IsolatedPush:
    """Wraps a `push(mono16k)` sink: chunks go through the isolator on a worker thread, then to
    `push` on the caller's event loop via `call_soon_threadsafe`."""

    def __init__(self, push: Callable[[np.ndarray], None], loop, isolator: Isolator | None = None):
        self.push, self.loop = push, loop
        self.iso = isolator or Isolator()
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.stats = {"chunks": 0, "raw": 0, "ms_total": 0.0, "ms_max": 0.0}
        threading.Thread(target=self._work, name="voice-isolate", daemon=True).start()

    def __call__(self, mono16k: np.ndarray) -> None:
        if self.q.qsize() >= MAX_BACKLOG:
            self.stats["raw"] += 1
            self.push(mono16k)
            return
        self.q.put(mono16k)

    def _work(self) -> None:
        while True:
            x = self.q.get()
            t0 = time.perf_counter()
            y = self.iso.feed(x)
            ms = 1000 * (time.perf_counter() - t0)
            self.stats["chunks"] += 1
            self.stats["ms_total"] += ms
            self.stats["ms_max"] = max(self.stats["ms_max"], ms)
            if len(y):
                self.loop.call_soon_threadsafe(self.push, y)
