"""The language step (the Muse Spark role): phrase an answer, pick the object a question is about.

`openai` is the stand-in until the Meta key exists. `meta` is Muse Spark on the Meta Model API,
which speaks the same chat-completions protocol, so switching is one config line and no code.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from ..config import Settings, secret


class LanguageModel(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 120) -> str: ...


class TemplateLanguage:
    """No model: callers fall back to their own templates."""

    name = "template"

    def complete(self, system: str, user: str, max_tokens: int = 120) -> str:
        return ""


class ChatCompletionsLanguage:
    def __init__(self, name: str, base_url: str, model: str, api_key: str, timeout: float = 20.0):
        self.name, self.model = name, model
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def complete(self, system: str, user: str, max_tokens: int = 120) -> str:
        r = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
        )
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()


def make_language(s: Settings) -> LanguageModel:
    if s.language == "meta":
        key = secret("META_API_KEY", "MODEL_API_KEY")
        if not key:
            raise RuntimeError("language = meta but no META_API_KEY / MODEL_API_KEY found")
        return ChatCompletionsLanguage("muse-spark", "https://api.meta.ai/v1", s.meta_model, key)
    if s.language == "openai":
        key = secret("OPENAI_API_KEY")
        if key:
            return ChatCompletionsLanguage(
                "openai", "https://api.openai.com/v1", s.openai_model, key
            )
    return TemplateLanguage()
