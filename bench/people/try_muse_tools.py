"""Ask the real Muse Spark questions that need tool calls, over a fixed fake state."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metasight.config import Settings  # noqa: E402
from metasight.pins import Pin, PinStore  # noqa: E402
from metasight.providers.language import make_language  # noqa: E402
from metasight.providers.tools import Tools, agent_answer  # noqa: E402

s = PinStore()
s.pins[1] = Pin(1, "keys", "object", [0, 0, 3.0], ts=900, first_ts=100, seen=3, near=["table"])
s.pins[2] = Pin(2, "keys", "object", [0, 0, 1.0], ts=990, first_ts=980, seen=1, near=["sofa"])
ev = {"voice": True, "speaker": None, "doa": -65.0, "tracks": [], "evidence": "none"}
head = NS(to=lambda pos: (abs(pos[2]), 20.0))
t = Tools(s, head, lambda: ev, now=lambda: 1000)
lang = make_language(Settings(language="meta", meta_model="muse-spark-1.3"))
for q in ("Did anyone move my keys in the last 5 minutes?", "Where is the person talking?",
          "Where are my keys?"):
    print(q, "->", agent_answer(lang, q, t))
