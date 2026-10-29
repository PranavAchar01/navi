"""Cost and latency of the deep map review on the live house pins.

python bench/map_review.py [--model wonder|glimmer] -> bench/results/map-review-<model>.json

Runs the review twice: on every change the house has ever logged (the worst case: the first
review after a long day) and on 10-minute windows, the normal cadence. Cost per hour assumes a
review every 10 minutes and one call only for windows that had changes.
"""

import argparse
import dataclasses
import json
import statistics
import urllib.request
from pathlib import Path

from metasight.config import Settings
from metasight.pins import Pin, PinStore
from metasight.providers import make_reviewer
from metasight.review import MapReview

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="wonder")
ap.add_argument("--url", default="http://127.0.0.1:8445/api/state")
a = ap.parse_args()

state = json.load(urllib.request.urlopen(a.url))
fields = {f.name for f in dataclasses.fields(Pin)}
store = PinStore()
for d in state["pins"]:
    store.pins[d["id"]] = Pin(**{k: v for k, v in d.items() if k in fields})
ts = sorted(p.first_ts for p in store.pins.values() if p.kind == "object")
end = max(p.ts for p in store.pins.values())

s = Settings.load()
s.review = a.model
lang = make_reviewer(s)
runs = []


def review(since: float, until: float) -> dict:
    r = MapReview(lang, store, now=lambda: until)
    r.since = since
    res = r.run()
    u = getattr(lang, "usage", None)
    res["tokens"] = [u.prompt_tokens, u.completion_tokens] if u and res["changes"] else None
    runs.append(res)
    return res


whole = review(ts[0] - 1, end)
windows, t = [], ts[0]
while t < end:
    windows.append(review(t, t + 600))
    t += 600
busy = [w for w in windows if w["changes"]]
usd = [w["usd"] for w in busy if "usd" in w]
out = {
    "model": lang.name,
    "pins": len(store.pins),
    "house_span_h": round((end - ts[0]) / 3600, 2),
    "whole_day": {k: whole.get(k) for k in ("changes", "ms", "usd", "tokens", "items")},
    "windows": len(windows),
    "windows_with_changes": len(busy),
    "window_ms_median": statistics.median(w["ms"] for w in busy) if busy else None,
    "window_usd_median": statistics.median(usd) if usd else None,
    "usd_per_hour_busy": round(6 * statistics.median(usd), 4) if usd else None,
    "usd_per_hour_observed": (
        round(sum(usd) / ((end - ts[0]) / 3600), 4) if usd and end > ts[0] else None
    ),
}
p = Path(f"bench/results/map-review-{a.model}.json")
p.write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "whole_day"}, indent=1))
print("whole day:", {k: whole.get(k) for k in ("changes", "ms", "usd", "tokens")})
for i in whole["items"][:8]:
    print(" ", i)
