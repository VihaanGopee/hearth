"""Ollama chat client (OpenAI-compatible endpoint, with tool calling)."""
from __future__ import annotations
import requests


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str, model: str, temperature: float = 0.6,
                 num_ctx: int = 16384, timeout: int = 300):
        self.base = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "options": {"num_ctx": self.num_ctx},
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        try:
            r = requests.post(f"{self.base}/v1/chat/completions", json=payload,
                              timeout=self.timeout)
        except requests.ConnectionError as e:
            raise LLMError(
                f"Can't reach Ollama at {self.base}. Is `ollama serve` running?"
            ) from e
        if r.status_code != 200:
            raise LLMError(f"Ollama returned {r.status_code}: {r.text[:500]}")
        data = r.json()
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Ollama response: {str(data)[:500]}") from e
