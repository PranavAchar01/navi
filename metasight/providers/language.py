"""The language step (the Muse Spark role): phrase an answer, pick the object a question is about.

`openai` is the stand-in until the Meta key exists. `meta` is Muse Spark on the Meta Model API,
reached with the OpenAI SDK (Meta ships no client of its own), so switching is one config line.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

from openai import OpenAI

from ..config import META_BASE_URL, Settings, secret


class LanguageModel(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 120) -> str: ...


class TemplateLanguage:
    """No model: callers fall back to their own templates."""

    name = "template"

    def complete(self, system: str, user: str, max_tokens: int = 120) -> str:
        return ""


class OpenAICompatLanguage:
    """Chat completions through the OpenAI SDK; `base_url` picks OpenAI or the Meta Model API."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 20.0,
        extra: dict | None = None,
    ):
        self.name, self.model, self.extra = name, model, extra or {}
        # one retry: a spoken answer that waits through three attempts is too late to be useful
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)

    def complete(self, system: str, user: str, max_tokens: int = 120) -> str:
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens + (300 if self.extra else 0),
            temperature=0.2,
            extra_body=self.extra,
        )
        self.usage = r.usage
        return (r.choices[0].message.content or "").strip()

    def keep_warm(self, every_s: float = 15.0) -> None:
        """Touch the API now and then so a question finds the TLS connection already open.
        Idle 45 s, an answer took 1253 ms median; with this ping, 1000 ms; back to back, 714 ms
        (bench/results/spark-warm-probe.json)."""

        def loop():
            while True:
                time.sleep(every_s)
                try:
                    self._client.models.list()
                except Exception:
                    pass  # an outage shows up on the next real question

        threading.Thread(target=loop, name="spark-warm", daemon=True).start()


def make_language(s: Settings) -> LanguageModel:
    if s.language == "meta":
        key = secret("META_API_KEY", "MODEL_API_KEY")
        if not key:
            raise RuntimeError("language = meta but no META_API_KEY / MODEL_API_KEY found")
        # Muse Spark reasons before it answers; without this the token budget goes to reasoning.
        spark = OpenAICompatLanguage(
            "muse-spark",
            s.meta_model,
            key,
            base_url=META_BASE_URL,
            timeout=30.0 if s.offline != "glimmer" else 4.0,
            extra={"reasoning_effort": "minimal"},
        )
        spark.keep_warm()
        if s.offline == "glimmer":  # no network: Muse Glimmer on this Mac phrases the answer
            from .glimmer import FallbackLanguage, GlimmerLanguage

            return FallbackLanguage(spark, GlimmerLanguage(s.glimmer_url))
        return spark
    if s.language == "openai":
        key = secret("OPENAI_API_KEY")
        if key:
            return OpenAICompatLanguage("openai", s.openai_model, key)
    return TemplateLanguage()


def make_reviewer(s: Settings) -> LanguageModel | None:
    """The deep loop's model: Muse Wonder on the API, or Glimmer here when review = "glimmer"."""
    from .glimmer import GlimmerLanguage

    key = secret("META_API_KEY", "MODEL_API_KEY") if s.review == "wonder" else None
    if s.review == "glimmer" or (s.review == "wonder" and not key and s.offline == "glimmer"):
        # 20 changes: 44 s on this Mac against Wonder's 9 to 15 s, and no bill
        return GlimmerLanguage(s.glimmer_url, timeout=90, cap=600)
    if not key:
        return None
    # minimal effort: 21 changes take 10 s and 1.1k tokens; default effort spent 4k thinking
    return OpenAICompatLanguage(
        "muse-wonder",
        "muse-wonder-1.0",
        key,
        META_BASE_URL,
        timeout=90,
        extra={"reasoning_effort": "minimal"},
    )
