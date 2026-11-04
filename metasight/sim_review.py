"""A second reader for the belt training rounds: the deep model reads the simulated walks and says
which change to keep, why, and what to try next.

Input is `bench/results/sim-walk.json` (scripts/run_sim_walk.py): each arm is one change to how
the belt cues, run on the same held-out seeds as its base arm, with a paired 95% interval on the
cost difference. The lab rule (docs/belt-sim-experiments.md) is fixed: keep a change only when
that interval is entirely below zero. The model gets a small digest, never the 2,520 raw rows.

Every reply is checked against the numbers, not trusted:
- `rule`: the verdict the lab rule gives from the interval alone.
- `agrees`: whether the model's verdict matches it.
- `unsupported`: numbers the model wrote that appear nowhere in the digest.
Nothing here runs on the belt; this is offline analysis for the person tuning it.
"""

from __future__ import annotations

import json
import re
import time

from .review import _usd

SYSTEM = (
    "You review simulated training rounds for a haptic belt that guides a blind person around "
    "obstacles. Each arm changes one thing against its base arm, on the same held-out seeds. "
    "cost is lower-is-better; paired is [mean, low, high] of the 95% interval of cost(arm) minus "
    "cost(base). The rule: keep only when high < 0; drop when low > 0; otherwise unclear. "
    "Reply with one JSON object per arm, nothing else: "
    '{"arm": str, "verdict": "keep"|"drop"|"unclear", "why": str, "watch": str}. '
    "why: one sentence citing numbers from the input. watch: one side effect to check on a real "
    "belt (flicker, clearance, time), or empty. Then one last line: "
    '{"next": str} naming the single most useful next experiment, in one sentence.'
)

KEYS = ("success", "collisions", "min_clear_m", "time_to_goal_s", "flicker_per_min", "cost")


def digest(sim: dict) -> list[dict]:
    """Per arm: the hypothesis, its base, the all-layout summary and the paired interval."""
    out = []
    for name, a in sim["arms"].items():
        s = a["summary"]["all"]
        d = {"arm": name, "change": a["hypothesis"], "base": a["base"]}
        d |= {k: s[k] for k in KEYS}
        if a.get("paired_cost_vs_base"):
            d["paired"] = a["paired_cost_vs_base"]
        out.append(d)
    return out


def rule(arm: dict) -> str | None:
    if "paired" not in arm:
        return None
    _, lo, hi = arm["paired"]
    return "keep" if hi < 0 else "drop" if lo > 0 else "unclear"


def _numbers(x) -> set[float]:
    found: set[float] = set()
    for m in re.finditer(r"-?\d+(?:\.\d+)?", json.dumps(x)):
        found.add(round(abs(float(m.group(0))), 2))
    return found


def check(items: list[dict], dig: list[dict]) -> list[dict]:
    """Attach the rule's verdict, agreement, and any number the model made up."""
    by = {d["arm"]: d for d in dig}
    known = _numbers(dig) | _numbers(SYSTEM)  # the prompt's own numbers (95%) are not made up
    out = []
    for i in items:
        d = by.get(i.get("arm", ""))
        if d is None:
            continue
        cited = _numbers(i.get("why", ""))
        unsupported = sorted(
            n for n in cited if n not in known and not any(abs(n - k) <= 0.011 for k in known)
        )
        r = rule(d)
        out.append(
            {**i, "rule": r, "agrees": None if r is None else i.get("verdict") == r}
            | {"unsupported": unsupported}
        )
    return out


def review(lang, sim: dict) -> dict:
    dig = digest(sim)
    t0 = time.perf_counter()
    raw = lang.complete(SYSTEM, json.dumps(dig), max_tokens=250 + 120 * len(dig))
    ms = round(1000 * (time.perf_counter() - t0))
    items, nxt = [], ""
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "arm" in d:
            items.append(d)
        elif "next" in d:
            nxt = str(d["next"])
    checked = check(items, dig)
    graded = [c for c in checked if c["agrees"] is not None]
    res = {
        "model": lang.name,
        "ms": ms,
        "arms": checked,
        "next": nxt,
        "agree": f"{sum(c['agrees'] for c in graded)}/{len(graded)}",
        "unsupported_numbers": sum(len(c["unsupported"]) for c in checked),
        "raw": raw,
    }
    if (usd := _usd(lang)) is not None:
        res["usd"] = usd
    return res
