"""Settings: metasight.toml, with environment overrides. Keys are never stored in files."""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "sessions"


def secret(*names: str) -> str | None:
    """First of: environment variable, macOS Keychain generic password with that service name."""
    for name in names:
        if value := os.environ.get(name):
            return value
    for name in names:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", name, "-w"],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return None


# Meta Model API: OpenAI-compatible, so the OpenAI SDK is the documented client.
META_BASE_URL = "https://api.meta.ai/v1"
META_TEAM_ID = "1009485265518211"
META_PROJECT_ID = "1741493303586631"


@dataclass
class Settings:
    language: str = "openai"
    finder: str = "yoloe"
    asr: str = "meta"  # spoken questions: "meta" (Muse Voice Transcribe) | "off" (client STT)
    offline: str = ""  # "glimmer": answers fall back to Muse Glimmer here (scripts/glimmer.sh)
    glimmer_url: str = "http://127.0.0.1:8091"
    review: str = "wonder"  # deep map review: "wonder" | "glimmer" | "off"
    review_every_min: float = 10.0
    openai_model: str = "gpt-4.1-mini"
    meta_model: str = "muse-spark-1.3"
    sam_model: str = "sam-3.1"
    depth_width: int = 518
    depth_height: int = 294
    detector_imgsz: int = 320
    # Starting live-depth scale. The glasses read about 2.5x too far: against the room mesh on 554
    # frames of 7 walks, 0.40 misses 0 of 513 near cues (1.0 missed 121). bench/mesh_reflex.py
    depth_scale: float = 0.40
    eye_height_m: float = 1.37  # camera height above the floor; the live depth's ruler
    hfov_deg: float = 105.0  # horizontal field of view of the camera feeding the safety loop
    map_intrinsics: bool = False  # tell MapAnything the lens (hfov_deg) instead of letting it guess
    # emulator: spawn firmware/emulator/belt-emu; udp: a real belt on Wi-Fi; app: the phone app
    # relays over Bluetooth; ble: this machine writes over Bluetooth (belt_addr = BLE address,
    # empty to find "metasight-belt")
    belt: str = "emulator"
    belt_addr: str = "127.0.0.1:4210"
    braille: str = "off"  # deafblind mode: off | emulator | wrist | belt | audio (docs/braille.md)
    braille_addr: str = ""  # BLE address of the wrist band, for braille = "wrist"
    braille_dot_ms: int = 250  # a cell is on this long (ERM motors need 50-100 ms to spin up)
    braille_cell_gap_ms: int = 150
    braille_word_gap_ms: int = 500
    braille_max_cells: int = 24  # cues are cut to about this many cells
    segment: bool = False  # HUD: open-vocabulary masks over the watch-list vocabulary
    keyframe_hz: float = 1.5
    window: int = 8
    overlap: int = 2
    voxel_m: float = 0.05
    horizon_m: float = 0.0  # outside mode: map points farther than this are forgotten
    # Small personal items that go to SAM 3.1 when finder = "hybrid", and how often.
    meta_items: list[str] = field(
        default_factory=lambda: [
            "water bottle",
            "keys",
            "wallet",
            "phone",
            "glasses case",
            "remote control",
        ]
    )
    meta_interval_s: float = 2.0
    watchlist: list[str] = field(
        default_factory=lambda: ["keys", "wallet", "phone", "glasses case", "remote control"]
    )
    weights_dir: Path = ROOT
    mode: str = "home"
    # Per-mode overrides of the memory settings; the safety loop is identical in both modes.
    outside: dict = field(
        default_factory=lambda: {
            "keyframe_hz": 2.5,
            "window": 4,
            "overlap": 1,
            "voxel_m": 0.08,
            "horizon_m": 25.0,
        }
    )

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        raw = {}
        p = path or Path(os.environ.get("METASIGHT_CONFIG", ROOT / "metasight.toml"))
        if p.exists():
            raw = tomllib.loads(p.read_text())
        s = cls()
        prov, mod = raw.get("providers", {}), raw.get("models", {})
        saf, mem = raw.get("safety", {}), raw.get("memory", {})
        s.language = os.environ.get("METASIGHT_LANGUAGE", prov.get("language", s.language))
        s.finder = os.environ.get("METASIGHT_FINDER", prov.get("finder", s.finder))
        s.asr = os.environ.get("METASIGHT_ASR", prov.get("asr", s.asr))
        s.offline = os.environ.get("METASIGHT_OFFLINE", prov.get("offline", s.offline))
        s.glimmer_url = prov.get("glimmer_url", s.glimmer_url)
        s.review = os.environ.get("METASIGHT_REVIEW", prov.get("review", s.review))
        s.review_every_min = float(prov.get("review_every_min", s.review_every_min))
        for k in ("openai_model", "meta_model", "sam_model"):
            setattr(s, k, mod.get(k, getattr(s, k)))
        for k in (
            "depth_width",
            "depth_height",
            "detector_imgsz",
            "segment",
            "depth_scale",
            "eye_height_m",
            "hfov_deg",
            "map_intrinsics",
            "belt",
            "belt_addr",
            "braille",
            "braille_addr",
            "braille_dot_ms",
            "braille_cell_gap_ms",
            "braille_word_gap_ms",
            "braille_max_cells",
        ):
            setattr(s, k, saf.get(k, getattr(s, k)))
        for k in (
            "keyframe_hz",
            "window",
            "overlap",
            "voxel_m",
            "watchlist",
            "meta_items",
            "meta_interval_s",
        ):
            setattr(s, k, mem.get(k, getattr(s, k)))
        s.mode = os.environ.get("METASIGHT_MODE", raw.get("mode", {}).get("start", s.mode))
        s.outside = {**s.outside, **raw.get("mode", {}).get("outside", {})}
        return s
