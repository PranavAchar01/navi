"""The language step (the Muse Spark role): phrase an answer, pick the object a question is about.

`openai` is the stand-in until the Meta key exists. `meta` is Muse Spark on the Meta Model API,
reached with the OpenAI SDK (Meta ships no client of its own), so switching is one config line.
"""

from __future__ import annotations

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
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(self, system: str, user: str, max_tokens: int = 120) -> str:
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens + (300 if self.extra else 0),
            temperature=0.2,
            extra_body=self.extra,
        )
        return (r.choices[0].message.content or "").strip()


def make_language(s: Settings) -> LanguageModel:
    if s.language == "meta":
        key = secret("META_API_KEY", "MODEL_API_KEY")
        if not key:
            raise RuntimeError("language = meta but no META_API_KEY / MODEL_API_KEY found")
        # Muse Spark reasons before it answers; without this the token budget goes to reasoning.
        return OpenAICompatLanguage(
            "muse-spark",
            s.meta_model,
            key,
            base_url=META_BASE_URL,
            timeout=30.0,
            extra={"reasoning_effort": "minimal"},
        )
    if s.language == "openai":
        key = secret("OPENAI_API_KEY")
        if key:
            return OpenAICompatLanguage("openai", s.openai_model, key)
    return TemplateLanguage()
