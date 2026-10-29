"""Muse Glimmer, Meta's open-weight 30B model, on this Mac: the answer path with no internet.

Served by llama.cpp (`scripts/glimmer.sh`, the 17 GB Q4_K_M GGUF from
meta-models/Muse-Glimmer-30B-GGUF). Glimmer is a reasoning model: left to its chat template it
writes ~100 to 300 tokens to itself before answering, which at ~11 tokens a second is 10 s or more.
Phrasing one sentence from given facts needs none of that, so the prompt opens the assistant turn
straight on the user channel (`assistant to=user`), which skips the private reasoning turn
(`to=self`) the template would otherwise open with. That measured 2.2 to 2.6 s for one sentence
on the Mac mini at background priority (bench/results/glimmer-local.json).
"""

from __future__ import annotations

import httpx

STOP = ["<|eot|>", "<|eom|>"]


def prompt(system: str, user: str) -> str:
    """Muse Glimmer's chat format with the reply opened on the user channel (no reasoning)."""
    return (
        f"<|start|>system<|message|>{system}\n\n"
        '# Valid recipients: "user".<|eot|>'
        f"<|start|>user<|message|>{user}<|eot|>"
        "<|start|>assistant to=user<|message|>"
    )


class GlimmerLanguage:
    name = "muse-glimmer"

    def __init__(self, url: str = "http://127.0.0.1:8091", timeout: float = 8.0):
        self.url = url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str, max_tokens: int = 120) -> str:
        r = self._http.post(
            f"{self.url}/completion",
            json={
                "prompt": prompt(system, user),
                "n_predict": min(max_tokens, 80),
                "stop": STOP,
                "cache_prompt": True,
                "temperature": 0.2,
            },
        )
        r.raise_for_status()
        return r.json()["content"].strip()

    def up(self) -> bool:
        try:
            return self._http.get(f"{self.url}/health", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            return False


class FallbackLanguage:
    """Muse Spark first; when the Meta Model API fails (no network, an outage), Muse Glimmer
    local. The name reports whichever answered last, so answers say which model phrased them."""

    def __init__(self, primary, backup):
        self.primary, self.backup = primary, backup
        self.name = primary.name

    def complete(self, system: str, user: str, max_tokens: int = 120) -> str:
        try:
            out = self.primary.complete(system, user, max_tokens)
            self.name = self.primary.name
            return out
        except Exception:
            out = self.backup.complete(system, user, max_tokens)
            self.name = self.backup.name
            return out

    def __getattr__(self, attr):
        return getattr(self.primary, attr)
