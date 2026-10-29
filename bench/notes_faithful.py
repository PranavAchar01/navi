"""Do the spoken answers about saved notes stay inside the transcript?

Five made-up visits, each saved as a note, each asked a question the notes answer plus one they
do not. Live Muse Spark with the same tool path the server uses. An answer is faithful when
`notes.unsupported` finds nothing it cannot back; correct when the expected fact is in it.
    PYTHONPATH=. python bench/notes_faithful.py
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from metasight.config import ROOT, Settings
from metasight.notes import Notes, unsupported
from metasight.pins import PinStore
from metasight.providers.language import make_language
from metasight.providers.tools import Tools, agent_answer

VISITS = [
    (
        [
            "Hi, how is the knee?",
            "Better, but stairs still hurt.",
            "Take 400 milligrams of ibuprofen twice a day with food.",
            "Come back on December 3rd and we will look at an X-ray.",
        ],
        [
            ("what did the doctor say about the ibuprofen dose", "400"),
            ("when is my next visit", "december 3"),
        ],
        "what did the doctor say about my blood pressure",
    ),
    (
        [
            "Your cholesterol came back at 210.",
            "Is that bad?",
            "A little high. Cut fried food and walk thirty minutes a day.",
            "We will recheck in three months.",
        ],
        [("what was my cholesterol number", "210"), ("how long should I walk", "30")],
        "what did they say about my sleep",
    ),
    (
        [
            "The rash should clear in a week.",
            "Use the cream once at night, a thin layer.",
            "If it spreads past the wrist, call the office right away.",
        ],
        [("how often do I use the cream", "once"), ("when should I call the office", "wrist")],
        "did the doctor mention an allergy test",
    ),
    (
        [
            "Your prescription is 5 milligrams of the blue pill every morning.",
            "Don't take it with grapefruit juice.",
            "Refill is at the pharmacy on Maple Street after Friday.",
        ],
        [
            ("what did the doctor say about the blue pill dose", "5"),
            ("what should I avoid with it", "grapefruit"),
        ],
        "what did they say about my diet",
    ),
    (
        [
            "The blood test shows low iron.",
            "Take one iron tablet with orange juice.",
            "Test again on January 14th and bring the results to the clinic.",
        ],
        [("what did the doctor say about iron", "tablet"), ("when is the next test", "january 14")],
        "what did they say about my thyroid",
    ),
]


def main() -> None:
    lang = make_language(Settings.load(ROOT / "metasight.toml"))
    rows = []
    for i, (lines, asks, trap) in enumerate(VISITS):
        n = Notes(Path(tempfile.mkdtemp()))
        n.start()
        for line in lines:
            n.hear(line)
        _, note = n.stop()
        n.summarize(note, lang)
        tools = Tools(PinStore(), None, lambda: None, notes=n)
        for q, expect in [*asks, (trap, None)]:
            t = time.perf_counter()
            said = agent_answer(lang, q, tools)
            ms = round(1000 * (time.perf_counter() - t))
            bad = unsupported(said, note.text())
            rows.append(
                {
                    "visit": i + 1,
                    "q": q,
                    "said": said,
                    "ms": ms,
                    "unsupported": bad,
                    "expected": expect,
                    "correct": None if expect is None else expect in said.lower(),
                }
            )
            print(rows[-1])
    asked = [r for r in rows if r["expected"]]
    out = {
        "model": lang.name,
        "answers": len(rows),
        "faithful": sum(not r["unsupported"] for r in rows),
        "correct": f"{sum(r['correct'] for r in asked)}/{len(asked)}",
        "rows": rows,
    }
    (ROOT / "bench/results/notes-faithful.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
