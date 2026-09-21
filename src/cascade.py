"""Model cascade: small fast model by default, escalate hard queries to a big model.

Why this exists (70B quest): the memory math says no dense 70B fits 16 GB,
but the MoE survey found ~35B-class IQ2_M GGUFs at ~10.6 GB that do. A cascade
lets Hearth keep a cheap small model resident for routine chatter and pay the
big model's RAM + latency only for queries that look like they need it.

Interface: wraps any two clients exposing ``chat(messages, tools)`` ->
OpenAI-style message dict (``OllamaClient`` and ``LlamaCppClient`` both
qualify). Tool schemas pass through untouched.

Routers:
  "heuristic" -- complexity rule on the last user message; above threshold the
                 query goes straight to the big model. One pass, cheap, but
                 can over-escalate.
  "verify"    -- always try the small model first; escalate only if it
                 refuses, returns empty text with no tool calls. Two passes
                 worst case, but no false escalations on easy queries.

The big-model client is built lazily via ``big_factory`` on the first
escalation, so the small model stays the only resident cost until the cascade
actually fires. There is no unload API in v1; restarting Hearth drops the big
model from RAM.
"""
from __future__ import annotations
import re

from .llm import LLMError

_ROUTERS = ("heuristic", "verify")

# Keywords that suggest a query benefits from a bigger model. Deliberately
# small and inspectable; tuned for an agent workload (code, math, planning).
_COMPLEXITY_KEYWORDS = frozenset({
    "implement", "algorithm", "debug", "refactor", "optimiz", "theorem",
    "proof", "lemma", "integral", "derivative", "eigenvalue", "matrix",
    "architecture", "research", "tradeoff", "trade-off", "strategy",
    "migrate", "```", "def ", "class ", "async ", "pointer", "recursion",
})

# Phrases that signal the small model gave up. Kept narrow on purpose: only
# match self-reported inability, not ordinary hedging.
_REFUSAL_RE = re.compile(
    r"\b(i (don't know|do not know|can't|cannot|am not able to|won't|will not)|"
    r"i'm not able|as an ai|unable to (help|answer|comply)|"
    r"not something i can|beyond my (capabilities|knowledge))\b",
    re.IGNORECASE)


def looks_like_refusal(text: str) -> bool:
    """True if the model text reads as a refusal / admission of inability."""
    return bool(_REFUSAL_RE.search(text or ""))


def looks_weak(msg: dict) -> bool:
    """True if a small-model reply is worth escalating in "verify" mode.

    Escalate on: refusal phrasing, or empty text with no tool calls (a bare
    empty reply is a failure; an empty reply *with* tool calls is a normal
    tool-use turn and must NOT escalate).
    """
    content = msg.get("content") or ""
    if msg.get("tool_calls"):
        return False
    if not content.strip():
        return True
    return looks_like_refusal(content)


def is_complex(text: str, len_chars: int = 2000, keyword_hits: int = 2) -> bool:
    """Heuristic complexity test for "heuristic" routing.

    Escalate when the last user message is long (> len_chars) or contains at
    least keyword_hits complexity keywords. Documented so the rule stays
    auditable; thresholds are constructor kwargs.
    """
    text = text or ""
    if len(text) > len_chars:
        return True
    lowered = text.lower()
    hits = sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in lowered)
    return hits >= keyword_hits


class CascadeClient:
    """Small-model-first client with big-model escalation.

    ``small`` is a ready client; ``big_factory`` is a zero-arg callable that
    builds the big client on first escalation (lazy, so the big model costs
    no RAM until it is actually used).
    """

    def __init__(self, small, big_factory,
                 router: str = "heuristic",
                 heuristic_len_chars: int = 2000,
                 heuristic_keyword_hits: int = 2):
        if router not in _ROUTERS:
            raise LLMError(
                f"unknown cascade router {router!r}; want one of {_ROUTERS}")
        self.small = small
        self._big_factory = big_factory
        self._big = None
        self.router = router
        self.heuristic_len_chars = heuristic_len_chars
        self.heuristic_keyword_hits = heuristic_keyword_hits
        # Observability for tests / logs; not part of the chat contract.
        self.last_route: str | None = None
        self.escalations = 0

    def _big_client(self):
        if self._big is None:
            self._big = self._big_factory()
        return self._big

    def _last_user_text(self, messages: list[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content") or ""
        return ""

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        if self.router == "heuristic" and is_complex(
                self._last_user_text(messages),
                self.heuristic_len_chars, self.heuristic_keyword_hits):
            return self._escalate(messages, tools)
        msg = self.small.chat(messages, tools)
        if self.router == "verify" and looks_weak(msg):
            return self._escalate(messages, tools)
        self.last_route = "small"
        return msg

    def _escalate(self, messages: list[dict],
                  tools: list[dict] | None) -> dict:
        self.escalations += 1
        self.last_route = "big"
        return self._big_client().chat(messages, tools)
