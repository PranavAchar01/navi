"""The deep model on two offline jobs: reading the belt training rounds, and tidying the map.

python bench/deep_reviews.py [--model wonder|glimmer] [--url http://127.0.0.1:8445/api/state]

1. Training rounds: `metasight.sim_review.review` on bench/results/sim-walk.json. Each verdict is
   graded against the lab rule (keep only when the paired interval is below zero), and any
   number the model wrote that is not in its input is counted.
2. Map tidy: `metasight.review.tidy` on the live house pins. Proposals are checked before they
   count; the result shows how many pins the checked proposals would merge or drop (on a copy,
   the live map is not changed).

Wonder is tried first. When the Meta Model API refuses it (no key, billing), Muse Glimmer on
this Mac does the same job and the file is named for it.
Writes bench/results/sim-review-<model>.json and bench/results/map-tidy-<model>.json.
"""

import argparse
import copy
import dataclasses
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metasight.config import META_BASE_URL, Settings, secret  # noqa: E402
from metasight.pins import Pin, PinStore  # noqa: E402
from metasight.providers.glimmer import GlimmerLanguage  # noqa: E402
from metasight.providers.language import OpenAICompatLanguage  # noqa: E402
from metasight.review import apply_tidy, tidy  # noqa: E402
from metasight.sim_review import review  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="wonder")
ap.add_argument("--url", default="http://127.0.0.1:8445/api/state")
ap.add_argument(
    "--home", default="data/home/home.json", help="saved house map, used when live is empty"
)
a = ap.parse_args()

s = Settings.load()
fallback_reason = None
lang = None
if a.model == "wonder" and (key := secret("META_API_KEY", "MODEL_API_KEY")):
    lang = OpenAICompatLanguage(
        "muse-wonder", "muse-wonder-1.0", key, META_BASE_URL, timeout=120,
        extra={"reasoning_effort": "minimal"},
    )  # fmt: skip
    try:
        lang.complete("Reply ok.", "ok", max_tokens=5)
    except Exception as e:  # the API's own words, kept in the result file
        fallback_reason = str(e)[:160]
        lang = None
if lang is None:
    lang = GlimmerLanguage(s.glimmer_url, timeout=600, cap=4000)

stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
name = lang.name

sim = json.loads((ROOT / "bench/results/sim-walk.json").read_text())
r1 = review(lang, sim) | {"at": stamp, "wonder_refused": fallback_reason}
(ROOT / f"bench/results/sim-review-{name}.json").write_text(json.dumps(r1, indent=1))
print(
    "training review", r1["ms"], "ms, agree", r1["agree"], "unsupported", r1["unsupported_numbers"]
)
for it in r1["arms"]:
    print(f"  {it['arm']} {it.get('verdict')} (rule {it['rule']}) {it.get('why', '')[:110]}")
print("  next:", r1["next"])

state = json.load(urllib.request.urlopen(a.url))
source = a.url
if not state["pins"]:  # the server is up but has not seen the house this session
    state, source = json.loads((ROOT / a.home).read_text()), a.home
fields = {f.name for f in dataclasses.fields(Pin)}
store = PinStore()
for d in state["pins"]:
    store.pins[d["id"]] = Pin(**{k: v for k, v in d.items() if k in fields})
r2 = tidy(lang, store) | {
    "at": stamp,
    "source": source,
    "pins": len(store.pins),
    "wonder_refused": fallback_reason,
}
trial = copy.deepcopy(store)
r2["applied_on_copy"] = apply_tidy(trial, r2)
r2["pins_after"] = len(trial.pins)
(ROOT / f"bench/results/map-tidy-{name}.json").write_text(json.dumps(r2, indent=1))
print("map tidy", r2.get("ms"), "ms,", r2["candidates"], "candidates,", len(r2["items"]),
      "checked,", len(r2["rejected"]), "rejected,", r2["applied_on_copy"])  # fmt: skip
for it in r2["items"]:
    labels = [store.pins[i].label for i in it["ids"]]
    print(f"  {it['action']} {it['ids']} {labels} {it['why'][:100]}")
