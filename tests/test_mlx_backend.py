"""Tests for the MLX backend prototype.

mlx is macOS-only, so the tests inject a fake mlx_lm module that captures
load/generate kwargs and returns scripted text. This verifies the wiring
logic (prompt formatting, sampler mapping, tool-call parse); real behavior
still needs a Mac with mlx-lm installed.
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mlx_backend import MlxClient, _parse_tool_calls
from src.llm import LLMError


class FakeTokenizer:
    instances = []

    def __init__(self):
        FakeTokenizer.instances.append(self)

    def apply_chat_template(self, messages, add_generation_prompt=False):
        self.last_messages = messages
        self.last_add_generation_prompt = add_generation_prompt
        return "PROMPT:" + "|".join(
            f"{m['role']}:{m['content']}" for m in messages)


class FakeMlxLm:
    load_calls = []
    generate_calls = []
    scripted_text = "hello from mlx"

    @staticmethod
    def reset():
        FakeMlxLm.load_calls = []
        FakeMlxLm.generate_calls = []
        FakeMlxLm.scripted_text = "hello from mlx"


def install_fake_mlx_lm():
    mod = types.ModuleType("mlx_lm")

    def fake_load(model, adapter_path=None):
        FakeMlxLm.load_calls.append(
            {"model": model, "adapter_path": adapter_path})
        return object(), FakeTokenizer()

    def fake_generate(model, tokenizer, prompt=None, **kwargs):
        FakeMlxLm.generate_calls.append(
            {"prompt": prompt, **kwargs})
        return FakeMlxLm.scripted_text

    mod.load = fake_load
    mod.generate = fake_generate
    sys.modules["mlx_lm"] = mod
    return mod


TOOLS = [{"function": {"name": "read_file", "description": "Read a file",
                       "parameters": {"path": {"type": "string"}}}}]


class TestMlxBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_fake_mlx_lm()

    def setUp(self):
        FakeMlxLm.reset()
        FakeTokenizer.instances.clear()

    def test_load_called_with_model_ref(self):
        MlxClient("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        self.assertEqual(FakeMlxLm.load_calls[0]["model"],
                         "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        self.assertIsNone(FakeMlxLm.load_calls[0]["adapter_path"])

    def test_adapter_path_forwarded(self):
        MlxClient("repo", adapter_path="/tmp/adapter")
        self.assertEqual(FakeMlxLm.load_calls[0]["adapter_path"],
                         "/tmp/adapter")

    def test_prompt_uses_chat_template(self):
        c = MlxClient("repo")
        c.chat([{"role": "user", "content": "hi"}])
        tok = FakeTokenizer.instances[-1]
        self.assertTrue(tok.last_add_generation_prompt)
        self.assertEqual(tok.last_messages,
                         [{"role": "user", "content": "hi"}])
        self.assertEqual(FakeMlxLm.generate_calls[0]["prompt"],
                         "PROMPT:user:hi")

    def test_sampler_kwargs_mapped(self):
        c = MlxClient("repo", temperature=0.7, top_p=0.9, max_tokens=512,
                      repetition_penalty=1.2, seed=42)
        c.chat([{"role": "user", "content": "hi"}])
        kw = FakeMlxLm.generate_calls[0]
        self.assertEqual(kw["temp"], 0.7)
        self.assertEqual(kw["top_p"], 0.9)
        self.assertEqual(kw["max_tokens"], 512)
        self.assertEqual(kw["repetition_penalty"], 1.2)
        self.assertEqual(kw["seed"], 42)

    def test_tool_messages_folded_to_user_text(self):
        c = MlxClient("repo")
        c.chat([{"role": "tool", "name": "read_file",
                 "content": "file contents", "tool_call_id": "call_0"}])
        tok = FakeTokenizer.instances[-1]
        self.assertEqual(len(tok.last_messages), 1)
        m = tok.last_messages[0]
        self.assertEqual(m["role"], "user")
        self.assertIn("read_file", m["content"])
        self.assertIn("file contents", m["content"])

    def test_plain_reply_returns_no_tool_calls(self):
        c = MlxClient("repo")
        msg = c.chat([{"role": "user", "content": "hi"}], tools=TOOLS)
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "hello from mlx")
        self.assertIsNone(msg["tool_calls"])

    def test_tool_instruction_prepended_when_tools(self):
        c = MlxClient("repo")
        c.chat([{"role": "user", "content": "hi"}], tools=TOOLS)
        tok = FakeTokenizer.instances[-1]
        self.assertEqual(tok.last_messages[0]["role"], "system")
        self.assertIn("read_file", tok.last_messages[0]["content"])

    def test_tool_calls_parsed_from_json_reply(self):
        FakeMlxLm.scripted_text = (
            '```json\n{"tool_calls": [{"name": "read_file", '
            '"arguments": {"path": "/tmp/x"}}]}\n```')
        c = MlxClient("repo")
        msg = c.chat([{"role": "user", "content": "read it"}],
                     tools=TOOLS)
        self.assertIsNotNone(msg["tool_calls"])
        tc = msg["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "read_file")
        self.assertEqual(tc["id"], "call_0")

    def test_unknown_tool_name_treated_as_plain_text(self):
        FakeMlxLm.scripted_text = (
            '{"tool_calls": [{"name": "nope", "arguments": {}}]}')
        c = MlxClient("repo")
        msg = c.chat([{"role": "user", "content": "hi"}], tools=TOOLS)
        self.assertIsNone(msg["tool_calls"])
        self.assertIn("nope", msg["content"])

    def test_malformed_json_treated_as_plain_text(self):
        FakeMlxLm.scripted_text = '{"tool_calls": [not json'
        c = MlxClient("repo")
        msg = c.chat([{"role": "user", "content": "hi"}], tools=TOOLS)
        self.assertIsNone(msg["tool_calls"])

    def test_generate_type_error_becomes_llm_error(self):
        import mlx_lm
        orig = mlx_lm.generate

        def bad_generate(*a, **k):
            raise TypeError("unexpected keyword 'temp'")
        mlx_lm.generate = bad_generate
        try:
            c = MlxClient("repo")
            with self.assertRaises(LLMError) as cm:
                c.chat([{"role": "user", "content": "hi"}])
            self.assertIn("sampler kwargs", str(cm.exception))
        finally:
            mlx_lm.generate = orig

    def test_load_failure_becomes_llm_error(self):
        import mlx_lm
        orig = mlx_lm.load

        def bad_load(*a, **k):
            raise RuntimeError("no such repo")
        mlx_lm.load = bad_load
        try:
            with self.assertRaises(LLMError) as cm:
                MlxClient("bogus")
            self.assertIn("failed to load", str(cm.exception))
        finally:
            mlx_lm.load = orig

    def test_missing_package_raises_llm_error(self):
        saved = sys.modules.get("mlx_lm")
        sys.modules["mlx_lm"] = None  # makes `import mlx_lm` raise
        try:
            with self.assertRaises(LLMError) as cm:
                MlxClient("repo")
            self.assertIn("mlx-lm is not installed", str(cm.exception))
        finally:
            sys.modules["mlx_lm"] = saved


class TestParseToolCalls(unittest.TestCase):
    def test_unfenced_object_parsed(self):
        text = '{"tool_calls": [{"name": "read_file", "arguments": {"path": "x"}}]}'
        calls = _parse_tool_calls(text, TOOLS)
        self.assertEqual(calls[0]["function"]["name"], "read_file")

    def test_plain_text_returns_none(self):
        self.assertIsNone(_parse_tool_calls("just an answer", TOOLS))

    def test_no_tools_returns_none(self):
        self.assertIsNone(_parse_tool_calls("hello", []))

    def test_empty_tool_calls_returns_none(self):
        self.assertIsNone(
            _parse_tool_calls('{"tool_calls": []}', TOOLS))


if __name__ == "__main__":
    unittest.main()
