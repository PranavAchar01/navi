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
    # voice isolation before Transcribe: "dns48" (metasight/isolate.py) | "off"
    isolate: str = "off"
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
    # How far the camera may pitch down and still find the floor. A phone is held near upright;
    # the glasses look down about 40 degrees while walking (gravity in tonight's take).
    floor_pitch_deg: float = 41.0
    hfov_deg: float = 105.0  # horizontal field of view of the camera feeding the safety loop
    map_intrinsics: bool = False  # tell MapAnything the lens (hfov_deg) instead of letting it guess
    # emulator: spawn firmware/emulator/belt-emu; udp: a real belt on Wi-Fi; app: the phone app
    # relays over Bluetooth; ble: this machine writes over Bluetooth (belt_addr = BLE address,
    # empty to find "metasight-belt"); serial: the USB cable (belt_addr = /dev/cu.usbserial-...)
    belt: str = "emulator"
    belt_addr: str = "127.0.0.1:4210"
    # Obstacle cues: "panned" (an object's bearing split across the two nearest directions) or
    # "sector" (one direction per sector). The built belt has four motors, and on four the sector
    # cues lose the bearing: 58% vs 73% of simulated walks reach the goal
    # (docs/belt-sim-experiments.md L1, L2).
    belt_cues: str = "panned"
    # Side and back sectors from the map. Off, only the three camera sectors can buzz: in a
    # small room the map's walls and furniture sit inside 1 m behind the wearer most of the time.
    map_sides: bool = True
    # "graded": intensity follows distance. "strong": every cue at full strength on the four
    # diagonal motors, distance told by rhythm alone (belt.strong), for faint motors.
    belt_strength: str = "graded"
    # Depth strips and the path ahead through a 3-frame median (pan.Median) instead of the
    # 2-frame near filter (pan.Persist) and the raw path.
    cue_median: bool = False
    cue_hold_s: float = 0.0  # a motor's rhythm steps down only after this long (belt.RhythmHold)
    # Panned cue ranges (pan.Params): silent beyond cue_off_m, slow pulse down to cue_fast_m, fast
    # down to cue_solid_m, solid inside. The defaults are the street ones; a bedroom wants shorter.
    cue_off_m: float = 1.5
    cue_fast_m: float = 1.6
    cue_solid_m: float = 0.7
    found_m: float = 0.8  # the guide says "found" when the item is this close
    sound: bool = True  # spatial sound on the phone: find beacon, hazard beeps (spatial_sound)
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
        s.isolate = os.environ.get("METASIGHT_ISOLATE", prov.get("isolate", s.isolate))
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
            "floor_pitch_deg",
            "hfov_deg",
            "map_intrinsics",
            "belt",
            "belt_addr",
            "belt_cues",
            "map_sides",
            "belt_strength",
            "cue_hold_s",
            "cue_median",
            "cue_off_m",
            "cue_fast_m",
            "cue_solid_m",
            "found_m",
            "sound",
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
