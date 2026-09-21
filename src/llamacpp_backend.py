"""Direct llama.cpp backend (via llama-cpp-python).

This is Hearth's "70B quest" engine: unlike Ollama it exposes the full GGUF
quant zoo (IQ1_M, TQ1_0, …), KV-cache quantization, and Metal GPU layers —
everything needed to try fitting huge models into 16 GB.

Same interface as OllamaClient: chat(messages, tools) -> OpenAI-style message.
"""
from __future__ import annotations
import os
from .llm import LLMError


class LlamaCppClient:
    def __init__(self, model_path: str, temperature: float = 0.6,
                 num_ctx: int = 8192, n_gpu_layers: int = -1,
                 n_threads: int = 0, n_batch: int = 512,
                 cache_type_k: str = "q8_0", cache_type_v: str = "q8_0",
                 timeout: int = 300):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise LLMError(
                "llama-cpp-python is not installed. Install it with:\n"
                "  Mac (Metal):  CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python\n"
                "  Linux (CPU):  pip install llama-cpp-python"
            ) from e
        self.model_path = os.path.expanduser(model_path)
        if not os.path.exists(self.model_path):
            raise LLMError(f"GGUF model not found: {self.model_path}")
        self.temperature = temperature
        self.timeout = timeout
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=num_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads or None,
            n_batch=n_batch,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            verbose=False,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        kwargs: dict = {
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            data = self._llm.create_chat_completion(**kwargs)
        except Exception as e:
            raise LLMError(f"llama.cpp error: {e}") from e
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected llama.cpp response: {str(data)[:500]}") from e
