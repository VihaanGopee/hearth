"""Apple-Silicon-native backend via mlx-lm.

This is the likely eventual engine for the 70B quest on the Mac: MLX runs
natively on Apple Silicon unified memory and has the best ~2-bit story there
(oQ / JANG mixed-precision quants, mlx-community models). The MoE survey
(research/moe_survey_2026-09-21.md) flags a 35B-A3B-class MoE at ~2 bit as
the practical end-of-week path — this backend is how Hearth would run it.

MAC-ONLY and UNTESTED-ON-MAC: mlx does not install on Linux, so this VM
cannot run any of it. Written carefully against the long-stable mlx_lm
Python API (load / generate / tokenizer.apply_chat_template), verified
2026-09-21 against upstream docs and third-party skill notes, but real
behavior must be validated on Justin's Mac before any default use.

Same interface as the other clients: chat(messages, tools) -> dict with
{"role": "assistant", "content": str, "tool_calls": [...]|None}.

Tool calls: mlx_lm has no server-side function calling, so v1 uses the
standard text-based fallback (EXPERIMENTAL, untested-on-Mac): tool schemas
are described in a system instruction and the model is asked to reply with
a {"tool_calls": [...]} JSON object, which is parsed back into the
OpenAI-style shape the agent loop expects. If the model answers in plain
text, it is returned as a normal reply with no tool calls.
"""
from __future__ import annotations

import json
import re

from .llm import LLMError


_TOOL_INSTRUCTION = """\
You have access to these tools. If you need a tool to answer, reply with ONLY a single JSON object and no other text:
{"tool_calls": [{"name": "tool_name", "arguments": {...}}, ...]}
Use the exact parameter names from each schema. One tool call per entry; you may include several entries.
Available tools:
"""


def _describe_tools(tools: list[dict]) -> str:
    lines = [_TOOL_INSTRUCTION]
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"- {name}: {desc}\n  Parameters: {json.dumps(params)}")
    return "\n".join(lines)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_tool_calls(text: str, tools: list[dict]) -> list[dict] | None:
    """Parse a text reply into OpenAI-style tool_calls, or None if the
    reply is not a tool-call request. Unknown tool names and malformed
    JSON are treated as plain text (no tool calls) — the model may just
    be answering directly."""
    names = {t.get("function", {}).get("name") for t in tools}
    candidate = None
    m = _JSON_FENCE_RE.search(text)
    if m:
        candidate = m.group(1)
    else:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    calls = obj.get("tool_calls") if isinstance(obj, dict) else None
    if not isinstance(calls, list) or not calls:
        return None
    parsed = []
    for i, c in enumerate(calls):
        if not isinstance(c, dict):
            return None
        name = c.get("name")
        args = c.get("arguments", {})
        if name not in names or not isinstance(args, dict):
            return None
        parsed.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return parsed


class MlxClient:
    def __init__(self, model: str, temperature: float = 0.6,
                 top_p: float = 1.0, max_tokens: int = 1024,
                 repetition_penalty: float = 1.0,
                 seed: int | None = None,
                 adapter_path: str | None = None):
        try:
            import mlx_lm
        except ImportError as e:
            raise LLMError(
                "mlx-lm is not installed. It is macOS/Apple-Silicon only:\n"
                "  pip install mlx-lm\n"
                "This backend cannot run on Linux."
            ) from e
        self._mlx_lm = mlx_lm
        self.model_ref = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        try:
            # Untested-on-Mac: load() accepts a HuggingFace repo id or a
            # local model directory; adapter_path is keyword-only.
            self._model, self._tokenizer = mlx_lm.load(
                model, adapter_path=adapter_path)
        except Exception as e:
            raise LLMError(
                f"mlx-lm failed to load {model!r}: {e}") from e

    def _prompt_messages(self, messages: list[dict]) -> list[dict]:
        """Fold messages into the {role, content} shape chat templates
        expect. Tool results have no template role, so they are folded
        into user text (marked clearly)."""
        out = []
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "tool":
                out.append({"role": "user",
                            "content": (f"[tool '{m.get('name', '?')}' "
                                        f"returned]\n{content}")})
            elif role in ("system", "user", "assistant"):
                out.append({"role": role, "content": str(content)})
        return out

    def chat(self, messages: list[dict],
             tools: list[dict] | None = None) -> dict:
        prompt_messages = self._prompt_messages(messages)
        if tools:
            prompt_messages = ([{"role": "system",
                                 "content": _describe_tools(tools)}]
                               + prompt_messages)
        try:
            prompt = self._tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True)
        except Exception as e:
            raise LLMError(f"mlx-lm chat template failed: {e}") from e
        try:
            # Untested-on-Mac: generate() takes sampler kwargs as **kwargs;
            # temp/top_p/repetition_penalty/seed are the long-stable names.
            text = self._mlx_lm.generate(
                self._model, self._tokenizer, prompt=prompt,
                max_tokens=self.max_tokens,
                temp=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                seed=self.seed,
            )
        except TypeError as e:
            raise LLMError(
                "mlx-lm generate() rejected the sampler kwargs — your "
                f"mlx-lm version may use different names: {e}") from e
        except Exception as e:
            raise LLMError(f"mlx-lm generation failed: {e}") from e
        tool_calls = _parse_tool_calls(text, tools) if tools else None
        if tool_calls is None:
            return {"role": "assistant", "content": text, "tool_calls": None}
        return {"role": "assistant", "content": "", "tool_calls": tool_calls}
