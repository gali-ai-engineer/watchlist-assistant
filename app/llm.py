"""Pluggable LLM providers behind one small interface.

Providers talk to each vendor's REST API directly (no SDKs), so adding or
swapping a model is a config change. If no key is configured, the app runs
on deterministic heuristics so the demo works fully offline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from . import http
from .config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 600) -> str: ...


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences or leading prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model did not return valid JSON: {e}") from None


class OpenAIProvider:
    def __init__(self, key: str, model: str, timeout: float):
        self.key, self.model, self.timeout = key, model, timeout
        self.name = f"openai:{model}"

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 600) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            r = http.post_json(
                "https://api.openai.com/v1/chat/completions",
                payload,
                headers={"Authorization": f"Bearer {self.key}"},
                timeout=self.timeout,
            )
            return r["choices"][0]["message"]["content"]
        except (http.HttpError, KeyError, IndexError) as e:
            raise LLMError(str(e)) from None


class AnthropicProvider:
    def __init__(self, key: str, model: str, timeout: float):
        self.key, self.model, self.timeout = key, model, timeout
        self.name = f"anthropic:{model}"

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 600) -> str:
        if json_mode:
            system += "\nRespond with a single JSON object and nothing else."
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            r = http.post_json(
                "https://api.anthropic.com/v1/messages",
                payload,
                headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                timeout=self.timeout,
            )
            return "".join(b.get("text", "") for b in r["content"] if b.get("type") == "text")
        except (http.HttpError, KeyError) as e:
            raise LLMError(str(e)) from None


class GeminiProvider:
    def __init__(self, key: str, model: str, timeout: float):
        self.key, self.model, self.timeout = key, model, timeout
        self.name = f"gemini:{model}"

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 600) -> str:
        cfg: dict = {"temperature": 0.1, "maxOutputTokens": max_tokens}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        try:
            r = http.post_json(url, payload, headers={"x-goog-api-key": self.key}, timeout=self.timeout)
            return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])
        except (http.HttpError, KeyError, IndexError) as e:
            raise LLMError(str(e)) from None


def build_llm(s: Settings) -> LLM | None:
    """Return the configured provider, or None to use offline heuristics."""
    choice = s.llm_provider
    candidates = {
        "openai": (s.openai_api_key, lambda: OpenAIProvider(s.openai_api_key, s.openai_model, s.llm_timeout_s)),
        "anthropic": (s.anthropic_api_key, lambda: AnthropicProvider(s.anthropic_api_key, s.anthropic_model, s.llm_timeout_s)),
        "gemini": (s.gemini_api_key, lambda: GeminiProvider(s.gemini_api_key, s.gemini_model, s.llm_timeout_s)),
    }
    if choice == "heuristic":
        return None
    if choice in candidates:
        key, factory = candidates[choice]
        if not key:
            log.warning("LLM_PROVIDER=%s but no API key set; falling back to heuristics", choice)
            return None
        return factory()
    for key, factory in candidates.values():  # auto
        if key:
            return factory()
    return None
