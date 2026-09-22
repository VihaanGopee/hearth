"""OpenAI-compatible HTTP chat backend.

Speaks the OpenAI /v1/chat/completions API over HTTP with zero
provider-specific fields, so Hearth can talk to any local
OpenAI-compatible server: `vmlx serve` (the rung-4 oQ2-smelt recipe),
`mlx_lm.server` (the MLX tool-calling transport follow-up), llama.cpp's
llama-server, LM Studio, etc.

Same interface as the other clients: chat(messages, tools) -> dict with
{"role": "assistant", "content": str, "tool_calls": [...]|None}.
Tool calling is passed through natively in the OpenAI format the rest of
the agent loop already uses — whether a given server actually invokes
tools depends on the server (mlx_lm.server support is a Mac-side
follow-up), so a reply with no tool_calls is normal, not an error.

Untested against a real server on this VM: the suite below tests the
client against a stub HTTP server.
"""
from __future__ import annotations

import os

import requests

from .llm import LLMError

_CHAT_COMPLETIONS_SUFFIX = "/v1/chat/completions"


def _normalize_base_url(base_url: str) -> str:
    """Turn a server root or chat-completions URL into the POST target."""
    url = base_url.rstrip("/")
    if url.endswith(_CHAT_COMPLETIONS_SUFFIX):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + _CHAT_COMPLETIONS_SUFFIX


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str,
                 temperature: float = 0.6,
                 max_tokens: int | None = None,
                 api_key: str | None = None,
                 timeout: int = 300):
        if not base_url:
            raise LLMError("openai backend needs a 'base_url' "
                           "(e.g. http://localhost:8000 for vmlx serve)")
        if not model:
            raise LLMError("openai backend needs a 'model' name")
        self.url = _normalize_base_url(base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Most local servers need no key; OPENAI_API_KEY lets one be set
        # without putting a secret in config.yaml.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    def _headers(self) -> dict:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def chat(self, messages: list[dict],
             tools: list[dict] | None = None) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = tools
        try:
            r = requests.post(self.url, json=payload, headers=self._headers(),
                              timeout=self.timeout)
        except requests.ConnectionError as e:
            raise LLMError(
                f"Can't reach the OpenAI-compatible server at {self.url}. "
                "Is it running?"
            ) from e
        except requests.Timeout as e:
            raise LLMError(
                f"OpenAI-compatible server at {self.url} timed out "
                f"after {self.timeout}s"
            ) from e
        if r.status_code != 200:
            raise LLMError(
                f"OpenAI-compatible server returned {r.status_code}: "
                f"{r.text[:500]}")
        try:
            data = r.json()
        except ValueError as e:
            raise LLMError(
                f"OpenAI-compatible server returned non-JSON: "
                f"{r.text[:500]}") from e
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(
                f"Unexpected OpenAI-compatible response: "
                f"{str(data)[:500]}") from e
