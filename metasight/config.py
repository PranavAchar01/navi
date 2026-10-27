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


@dataclass
class Settings:
    language: str = "openai"
    finder: str = "yoloe"
    openai_model: str = "gpt-4.1-mini"
    meta_model: str = "muse-spark-1.3"
    sam_model: str = "sam-3.1"
    depth_width: int = 518
    depth_height: int = 294
    detector_imgsz: int = 320
    keyframe_hz: float = 1.5
    window: int = 8
    overlap: int = 2
    voxel_m: float = 0.05
    watchlist: list[str] = field(
        default_factory=lambda: ["keys", "wallet", "phone", "glasses case", "remote control"]
    )
    weights_dir: Path = ROOT

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        raw = {}
        p = path or ROOT / "metasight.toml"
        if p.exists():
            raw = tomllib.loads(p.read_text())
        s = cls()
        prov, mod = raw.get("providers", {}), raw.get("models", {})
        saf, mem = raw.get("safety", {}), raw.get("memory", {})
        s.language = os.environ.get("METASIGHT_LANGUAGE", prov.get("language", s.language))
        s.finder = os.environ.get("METASIGHT_FINDER", prov.get("finder", s.finder))
        for k in ("openai_model", "meta_model", "sam_model"):
            setattr(s, k, mod.get(k, getattr(s, k)))
        for k in ("depth_width", "depth_height", "detector_imgsz"):
            setattr(s, k, saf.get(k, getattr(s, k)))
        for k in ("keyframe_hz", "window", "overlap", "voxel_m", "watchlist"):
            setattr(s, k, mem.get(k, getattr(s, k)))
        return s
