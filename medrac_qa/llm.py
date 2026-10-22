"""Minimal client for any OpenAI-compatible chat-completions API.

The same few lines work with Google Gemini, Groq, OpenRouter and a local
Ollama server; only the base URL, key and model name change (see .env.example).
Plain HTTP via `requests` keeps the dependency list short.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests


class LLMError(RuntimeError):
    pass


@dataclass
class LLMReply:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        max_tokens: int = 400,
        reasoning_effort: str = "",
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def chat(self, messages: list[dict], temperature: float = 0.1, retries: int = 2) -> LLMReply:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,  # low temperature: factual, repeatable answers
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        start = time.perf_counter()
        for attempt in range(retries + 1):
            try:
                r = requests.post(self.url, json=body, headers=headers, timeout=self.timeout_s)
            except requests.RequestException as e:
                if attempt == retries:
                    raise LLMError(f"could not reach the language model: {e}") from e
                continue
            if r.status_code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(1.5 * (attempt + 1))  # rate limit or overload: back off and retry
                continue
            if r.status_code != 200:
                raise LLMError(f"language model returned HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            usage = data.get("usage") or {}
            return LLMReply(
                text=text,
                model=data.get("model", self.model),
                latency_ms=(time.perf_counter() - start) * 1000,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        raise LLMError("language model did not respond")
