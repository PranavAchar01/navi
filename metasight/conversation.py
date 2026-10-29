"""Conversations: the glue between the camera, the microphones and the cues.

`Talk` owns the VAD, the direction finder and the people tracker for one source. The server hands
it each analysed frame and each audio chunk and gets back events to broadcast (docs/people.md has
their schema) and, when the wearer should turn toward a speaker, a short belt overlay.
"""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np

from .audio_doa import SR, Doa, Vad, to_mono16k
from .belt import MODES
from .belt_modes import talk_cmd
from .people import Conversation, Det

AUDIO_STALE_S = 1.5  # no audio for this long: nobody is heard
CUE_PULSE_S = 0.7  # how long the belt overlay for a turn cue lasts
CUE_INTENSITY = 190


class Talk:
    def __init__(self, hfov_deg: float):
        self.conv = Conversation(hfov_deg)
        self.vad: Vad | None = None  # built on the first audio chunk (loads the model)
        self.doa: Doa | None = None
        self.mics = 0
        self.audio_t = -math.inf
        self.cue: dict | None = None
        self.cue_until = 0.0
        self.cues_enabled = True  # the mode decides: only Talk turns the wearer toward a voice

    def reset(self) -> None:
        self.conv.reset()
        self.cue = None
        if self.vad is not None:
            self.vad.reset()
        if self.doa is not None:
            self.doa.reset()

    def audio(self, m: dict, now: float) -> None:
        """{"type": "audio", "sr": 16000, "ch": 2, "spacing": 0.14, "pcm": base64 int16 LE,
        interleaved}. `spacing` (metres between the left and right microphone) is only needed
        for ch >= 2; with one channel the speaker is found from the VAD and the lips alone."""
        pcm = np.frombuffer(base64.b64decode(m["pcm"]), dtype="<i2").astype(np.float32) / 32768
        ch, sr = max(1, int(m.get("ch", 1))), int(m.get("sr", SR))
        x = pcm[: len(pcm) // ch * ch].reshape(-1, ch)
        if self.vad is None:
            self.vad = Vad()
        self.mics = ch
        spacing = float(m.get("spacing") or 0)
        if ch >= 2 and spacing > 0 and (self.doa is None or self.doa.spacing_m != spacing):
            self.doa = Doa(spacing)
        mono = to_mono16k(x.mean(axis=1), sr)
        self.vad.push(mono)
        self.audio_t = now
        if ch >= 2 and self.doa is not None:
            two = np.stack([to_mono16k(x[:, 0], sr), to_mono16k(x[:, 1], sr)], axis=1)
            self.doa.push(two, self.vad.active())

    def frame(self, now: float, obstacles, bgr: np.ndarray, heading) -> dict:
        """People event for one analysed frame; `obstacles` are the safety loop's detections."""
        heard = self.vad is not None and now - self.audio_t < AUDIO_STALE_S
        voice = bool(heard and self.vad.active())
        doa = self.doa.bearing() if heard and self.doa is not None and voice else None
        dets = [Det(o.box, o.score, o.dist_m) for o in obstacles if o.label == "person"]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        xz = None
        if heading.pos is not None and heading.yaw is not None:
            xz = (float(heading.pos[0]), float(heading.pos[2]), float(heading.yaw))
        self.conv.hfov_deg = heading.hfov_deg
        ev = self.conv.step(now, dets, gray, voice, doa, xz)
        ev["mics"] = self.mics if heard else 0
        ev["vad"] = self.vad.kind if self.vad is not None else None
        if ev["cue"] and not self.cues_enabled:
            ev["cue"] = None
        if ev["cue"]:
            self.cue, self.cue_until = ev["cue"], now + CUE_PULSE_S
        return ev

    def overlay(self, cmd: list[tuple[int, int]], now: float) -> list[tuple[int, int]]:
        """The turn cue on the belt: the Talk taps on the two motors toward the speaker, unless
        a proximity cue already holds them (walking into something outranks conversation)."""
        if self.cue is None or now > self.cue_until:
            return cmd
        out = list(cmd)
        for motor, (mode, amp) in enumerate(talk_cmd(self.cue["bearing"], CUE_INTENSITY)):
            if mode != MODES["off"] and out[motor][0] == MODES["off"]:
                out[motor] = (mode, amp)
        return out
