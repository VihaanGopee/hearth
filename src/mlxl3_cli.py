"""mlxl3 CLI-subprocess bridge.

mlxl3 (0xZKnw/mlxl3) is a standalone EXL3 inference engine on MLX with no
documented HTTP/OpenAI server surface (checked 2026-09-22 against v1.1.1):
the only programmatic hook is the one-shot CLI mode
`mlxl3 run <model> --prompt "..." --max-tokens N`. This module adapts that
hook to Hearth's client interface — chat(messages, tools) ->
{"role": "assistant", "content": str, "tool_calls": None} — so the 2.08bpw
EXL3 rung-4 candidate (research/model_profiles.yaml, status: watch) can be
driven by the measurement tools without HTTP.

UNTESTED-ON-MAC: the mlxl3 binary does not exist on this VM, so the suite
below tests the subprocess plumbing against a fake CLI script. The
Mac-side validation step is: (1) stdout parsing against a real
`mlxl3 run --prompt` (echoes/banners would need stripping here), (2)
whether `run` reloads weights per invocation — if it does, this bridge is
a batch-measurement tool, not an interactive backend, (3) any CLI exposure
of the DFlash2 speculative toggle (currently a Desktop switch).

Hard constraints, stated up front rather than discovered at runtime:
- Greedy-only: one-shot mode is greedy, so there is no temperature knob
  and this client takes none.
- No tool surface: chat() with non-empty tools raises LLMError instead of
  silently dropping the schemas — an unvalidated text-based tool fallback
  would corrupt agent turns, so fail loud.
- Prompt rendering: the message list is rendered as a role-labeled
  transcript and passed whole as --prompt. Whether mlxl3 applies the
  model's chat template to --prompt as a single user turn (in which case a
  pre-rendered transcript nests oddly) is unverified — the Mac-side step
  is to compare this rendering against passing only the last user message.

Not wired into agent.py: use directly, or add an opt-in backend entry in a
follow-up once the Mac-side parse is validated.
"""
from __future__ import annotations

import subprocess

from .llm import LLMError

#: How much stderr we surface when the CLI fails — enough to diagnose,
#: not enough to flood the log.
_STDERR_TAIL = 500


def _render_prompt(messages: list[dict]) -> str:
    """Render a message list as a plain role-labeled transcript.

    Assistant messages that carried tool calls contribute their text only
    (the CLI has no tool surface, and chat() refuses tool schemas).
    """
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class Mlxl3CliClient:
    """Hearth client over the mlxl3 one-shot CLI.

    Same chat() interface as the other clients; one subprocess per chat()
    call. No temperature (greedy CLI), no tools.
    """

    def __init__(self, model: str, cli_bin: str = "mlxl3",
                 max_tokens: int | None = None,
                 extra_args: list[str] | tuple[str, ...] = (),
                 timeout: int = 300):
        if not model:
            raise LLMError("mlxl3 bridge needs a 'model' "
                           "(HF repo id or local name as mlxl3 expects it)")
        if not cli_bin:
            raise LLMError("mlxl3 bridge needs a 'cli_bin' path")
        self.model = model
        self.cli_bin = cli_bin
        self.max_tokens = max_tokens
        self.extra_args = list(extra_args)
        self.timeout = timeout

    def _argv(self, prompt: str) -> list[str]:
        argv = [self.cli_bin, "run", self.model,
                "--prompt", prompt]
        if self.max_tokens is not None:
            argv += ["--max-tokens", str(self.max_tokens)]
        argv += self.extra_args
        return argv

    def chat(self, messages: list[dict],
             tools: list[dict] | None = None) -> dict:
        if tools:
            raise LLMError("mlxl3 CLI bridge has no tool surface: the "
                           "one-shot `mlxl3 run --prompt` mode accepts plain "
                           "text only; refusing to silently drop tool "
                           "schemas")
        prompt = _render_prompt(messages)
        try:
            proc = subprocess.run(
                self._argv(prompt),
                capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError as e:
            raise LLMError(f"mlxl3 CLI not found at {self.cli_bin!r}: "
                           "install mlxl3 on the Mac and put it on PATH") from e
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"mlxl3 CLI timed out after {self.timeout}s "
                           f"(model={self.model!r})") from e
        if proc.returncode != 0:
            tail = proc.stderr[-_STDERR_TAIL:]
            raise LLMError(f"mlxl3 CLI exited {proc.returncode} "
                           f"(model={self.model!r}): {tail}")
        # The real CLI's one-shot stdout format is unvalidated on this VM
        # (echoes/banners would need stripping here — Mac-side step).
        return {"role": "assistant",
                "content": proc.stdout.strip(),
                "tool_calls": None}
