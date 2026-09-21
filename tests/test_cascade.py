"""Tests for the model cascade (small default + big-model escalation)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cascade import (CascadeClient, is_complex, looks_like_refusal,
                         looks_weak)
from src.llm import LLMError


class FakeClient:
    """Records chat() calls and returns a canned message."""

    def __init__(self, reply: dict):
        self.reply = reply
        self.calls: list[tuple] = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return dict(self.reply)


def msg(content: str, tool_calls=None) -> dict:
    return {"content": content, "tool_calls": tool_calls or []}


def make(router="heuristic", small_reply=msg("small answer"), **kw):
    small = FakeClient(small_reply)
    big = FakeClient(msg("big answer"))
    built = []
    c = CascadeClient(small, lambda: built.append(big) or big,
                      router=router, **kw)
    return c, small, big, built


class TestHeuristicRouting(unittest.TestCase):
    def test_simple_query_stays_small(self):
        c, small, big, built = make()
        out = c.chat([{"role": "user", "content": "what time is it?"}])
        self.assertEqual(out["content"], "small answer")
        self.assertEqual(c.last_route, "small")
        self.assertEqual(built, [])  # big never constructed (lazy)

    def test_long_query_escalates(self):
        c, small, big, built = make()
        out = c.chat([{"role": "user", "content": "x" * 2001}])
        self.assertEqual(out["content"], "big answer")
        self.assertEqual(c.last_route, "big")
        self.assertEqual(len(small.calls), 0)
        self.assertEqual(c.escalations, 1)

    def test_two_keywords_escalate(self):
        c, _, _, _ = make()
        out = c.chat([{"role": "user",
                       "content": "debug this recursion bug in my algorithm"}])
        self.assertEqual(out["content"], "big answer")
        self.assertEqual(c.last_route, "big")

    def test_single_keyword_stays_small(self):
        c, _, _, built = make()
        out = c.chat([{"role": "user", "content": "debug this for me"}])
        self.assertEqual(out["content"], "small answer")
        self.assertEqual(built, [])

    def test_code_fence_counts_as_keyword(self):
        c, _, _, _ = make()
        out = c.chat([{"role": "user",
                       "content": "fix ```python\nprint(1)\n``` the algorithm"}])
        self.assertEqual(c.last_route, "big")

    def test_big_built_at_most_once(self):
        c, _, big, built = make()
        long = [{"role": "user", "content": "y" * 2001}]
        c.chat(long)
        c.chat(long)
        self.assertEqual(len(built), 1)
        self.assertEqual(len(big.calls), 2)
        self.assertEqual(c.escalations, 2)

    def test_tools_pass_through(self):
        c, small, _, _ = make()
        tools = [{"type": "function", "function": {"name": "f"}}]
        c.chat([{"role": "user", "content": "hi"}], tools=tools)
        self.assertEqual(small.calls[0][1], tools)

    def test_routes_on_last_user_message(self):
        c, _, _, _ = make()
        messages = [
            {"role": "user", "content": "z" * 2001},  # old, complex
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "thanks"},    # last, simple
        ]
        c.chat(messages)
        self.assertEqual(c.last_route, "small")


class TestVerifyRouting(unittest.TestCase):
    def test_good_small_answer_no_escalation(self):
        c, small, _, built = make(router="verify")
        out = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out["content"], "small answer")
        self.assertEqual(built, [])

    def test_refusal_escalates(self):
        c, _, _, _ = make(router="verify",
                          small_reply=msg("I don't know how to do that."))
        out = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out["content"], "big answer")
        self.assertEqual(c.last_route, "big")

    def test_empty_reply_without_tools_escalates(self):
        c, _, _, _ = make(router="verify", small_reply=msg("   "))
        out = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(c.last_route, "big")

    def test_empty_reply_with_tool_calls_does_not_escalate(self):
        tc = [{"id": "1", "function": {"name": "f", "arguments": "{}"}}]
        c, _, _, built = make(router="verify", small_reply=msg("", tc))
        out = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out["tool_calls"], tc)
        self.assertEqual(c.last_route, "small")
        self.assertEqual(built, [])


class TestHelpers(unittest.TestCase):
    def test_is_complex_thresholds(self):
        self.assertTrue(is_complex("a" * 2001))
        self.assertFalse(is_complex("a" * 2000))
        self.assertTrue(is_complex("prove the theorem by lemma",
                                   keyword_hits=2))
        self.assertFalse(is_complex("prove the theorem", keyword_hits=2))
        self.assertTrue(is_complex("short", len_chars=3))

    def test_refusal_phrases(self):
        self.assertTrue(looks_like_refusal("I'm sorry, I can't help with that."))
        self.assertTrue(looks_like_refusal("As an AI, I cannot do this."))
        self.assertTrue(looks_like_refusal("I don't know how to do that."))
        self.assertFalse(looks_like_refusal("I can help with that."))

    def test_looks_weak(self):
        self.assertTrue(looks_weak(msg("")))
        self.assertTrue(looks_weak(msg("I don't know.")))
        self.assertFalse(looks_weak(msg("here you go")))
        tc = [{"id": "1"}]
        self.assertFalse(looks_weak(msg("", tc)))

    def test_unknown_router_rejected(self):
        with self.assertRaises(LLMError):
            CascadeClient(FakeClient(msg("x")), lambda: None, router="oracle")


class TestAgentWiring(unittest.TestCase):
    def _cfg(self):
        from src.config import load_config
        cfg = load_config()
        return cfg

    def test_cascade_backend_builds(self):
        from src.agent import Agent
        cfg = self._cfg()
        cfg["backend"] = "cascade"
        cfg["cascade"] = {
            "router": "heuristic",
            "small": {"backend": "ollama", "ollama": cfg["ollama"]},
            "big": {"backend": "ollama",
                    "ollama": dict(cfg["ollama"], model="qwen3:32b")},
        }
        agent = Agent(cfg)
        self.assertIsInstance(agent.llm, CascadeClient)
        self.assertIn("qwen3:8b", agent.model_label)
        self.assertIn("qwen3:32b", agent.model_label)

    def test_cascade_requires_big_spec(self):
        from src.agent import Agent
        cfg = self._cfg()
        cfg["backend"] = "cascade"
        cfg["cascade"] = {}
        with self.assertRaises(LLMError):
            Agent(cfg)

    def test_default_still_ollama(self):
        from src.agent import Agent
        from src.llm import OllamaClient
        agent = Agent(self._cfg())
        self.assertIsInstance(agent.llm, OllamaClient)


if __name__ == "__main__":
    unittest.main()
