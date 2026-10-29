"""One plan and one nudge pass from Muse Wonder on the live house pins.

python bench/wonder_plan_nudge.py -> bench/results/wonder-plan-nudge.json

Saves the raw replies, latency, cost, and every item the model named that is not in the bank
(`invented`; `grounded` drops those before anything is said).
"""

import argparse
import dataclasses
import json
import time
import urllib.request
from pathlib import Path

from metasight.config import Settings
from metasight.pins import Pin, PinStore
from metasight.providers import make_reviewer
from metasight.review import bank, nudges, plan

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8445/api/state")
ap.add_argument("--request", default="get ready to leave")
a = ap.parse_args()

state = json.load(urllib.request.urlopen(a.url))
fields = {f.name for f in dataclasses.fields(Pin)}
store = PinStore()
for d in state["pins"]:
    store.pins[d["id"]] = Pin(**{k: v for k, v in d.items() if k in fields})

lang = make_reviewer(Settings.load())
now = time.time()
b = bank(store, now)
out = {
    "model": lang.name,
    "pins": len(store.pins),
    "bank_labels": sorted(b),
    "request": a.request,
    "plan": plan(lang, store, a.request, now),
    "nudges": nudges(lang, store, now),
}
p = Path("bench/results/wonder-plan-nudge.json")
p.write_text(json.dumps(out, indent=1))
for k in ("plan", "nudges"):
    r = out[k]
    print(k, r["ms"], "ms", round(r.get("usd", 0), 4), "$", "invented:", r["invented"])
    for i in r["items"]:
        print("  ", i["object"], "|", i["say"])
