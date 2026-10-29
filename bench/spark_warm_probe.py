"""Does a warm connection shorten Muse Spark's answer? Same request after a 45 s idle gap (no
ping), after 45 s with a models.list() ping every 15 s, and back to back.
Writes bench/results/spark-warm-probe.json."""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from openai import OpenAI  # noqa: E402

from metasight.config import META_BASE_URL, secret  # noqa: E402
from metasight.query import SYSTEM  # noqa: E402

FACTS = (
    "object: keys\nseen: 2 minutes ago\nnear: sofa\ndistance: about 3 metres\n"
    "direction: behind you to the left"
)
c = OpenAI(base_url=META_BASE_URL, api_key=secret("META_API_KEY", "MODEL_API_KEY"), timeout=30)


def call() -> float:
    t0 = time.perf_counter()
    c.chat.completions.create(
        model="muse-spark-1.3",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": FACTS}],
        max_tokens=420,
        extra_body={"reasoning_effort": "minimal"},
    )
    return 1000 * (time.perf_counter() - t0)


def idle(seconds: float, ping: bool) -> None:
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(min(15, end - time.time()))
        if ping and time.time() < end:
            c.models.list()


res = {"cold": [], "pinged": [], "back_to_back": []}
call()
for _ in range(5):
    idle(45, False)
    res["cold"].append(call())
    idle(45, True)
    res["pinged"].append(call())
for _ in range(5):
    res["back_to_back"].append(call())
out = {
    k: {"median_ms": round(statistics.median(v)), "all_ms": [round(x) for x in v]}
    for k, v in res.items()
}
(ROOT / "bench/results/spark-warm-probe.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
