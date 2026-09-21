"""Direct llama.cpp backend (via llama-cpp-python).

This is Hearth's "70B quest" engine: unlike Ollama it exposes the full GGUF
quant zoo (IQ1_M, TQ1_0, …), KV-cache quantization, and Metal GPU layers —
everything needed to try fitting huge models into 16 GB.

Same interface as OllamaClient: chat(messages, tools) -> OpenAI-style message.

Speculative decoding (draft + target verification) accelerates decode:
speculative="prompt_lookup" uses prompt n-gram lookup (no extra model,
verified API: llama_cpp.llama_speculative.LlamaPromptLookupDecoding);
speculative="draft_model" loads a small draft GGUF alongside the target
(marked untested-on-Mac below: LlamaDraftModel constructor signature may
vary across llama-cpp-python versions).
"""
from __future__ import annotations
import os
from .llm import LLMError


class LlamaCppClient:
    def __init__(self, model_path: str, temperature: float = 0.6,
                 num_ctx: int = 8192, n_gpu_layers: int = -1,
                 n_threads: int = 0, n_batch: int = 512,
                 cache_type_k: str = "q8_0", cache_type_v: str = "q8_0",
                 timeout: int = 300,
                 speculative: str = "off",
                 draft_model_path: str | None = None,
                 draft_n_tokens: int = 10):
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
        self.speculative = speculative
        self.draft_model_path = (os.path.expanduser(draft_model_path)
                                 if draft_model_path else None)
        self.draft_n_tokens = draft_n_tokens
        draft_model = self._build_draft_model()
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=num_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads or None,
            n_batch=n_batch,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            draft_model=draft_model,
            verbose=False,
        )

    def _build_draft_model(self):
        """Construct the draft model object for speculative decoding.

        Untested-on-Mac: the draft_model branch relies on the
        llama-cpp-python LlamaDraftModel constructor signature, which we
        could not verify on this VM (package not installed). The
        prompt_lookup branch uses the documented public API.
        """
        mode = self.speculative
        if mode == "off":
            return None
        if mode == "prompt_lookup":
            if self.draft_n_tokens < 1:
                raise LLMError(
                    f"draft_n_tokens must be >= 1, got {self.draft_n_tokens}")
            from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
            # num_pred_tokens: upstream default 10 is "generally good for
            # gpu"; 2 performs better on cpu-only machines.
            return LlamaPromptLookupDecoding(
                num_pred_tokens=self.draft_n_tokens)
        if mode == "draft_model":
            if not self.draft_model_path:
                raise LLMError(
                    'speculative="draft_model" requires draft_model_path')
            if not os.path.exists(self.draft_model_path):
                raise LLMError(
                    f"Draft GGUF not found: {self.draft_model_path}")
            from llama_cpp.llama_speculative import LlamaDraftModel
            try:
                return LlamaDraftModel(path_or_hf=self.draft_model_path)
            except TypeError as e:
                raise LLMError(
                    "Could not construct LlamaDraftModel — your "
                    "llama-cpp-python version may use a different "
                    f"constructor signature: {e}"
                ) from e
        raise LLMError(
            f'Unknown speculative mode {mode!r}: '
            'expected "off", "prompt_lookup", or "draft_model"')

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
