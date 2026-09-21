"""Tests for speculative decoding support in the llamacpp backend.

llama_cpp is not installed on this VM, so the tests inject a fake
llama_cpp / llama_cpp.llama_speculative module pair that captures
constructor kwargs. This verifies the wiring logic; real behavior still
needs a Mac with llama-cpp-python.
"""
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llamacpp_backend import LlamaCppClient
from src.llm import LLMError


class FakeLlama:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeLlama.instances.append(self)

    def create_chat_completion(self, **kwargs):
        self.last_call = kwargs
        return {"choices": [{"message": {"role": "assistant",
                                         "content": "hello"}}]}


class FakePromptLookup:
    instances = []

    def __init__(self, num_pred_tokens=10):
        self.num_pred_tokens = num_pred_tokens
        FakePromptLookup.instances.append(self)


class FakeDraftModel:
    instances = []

    def __init__(self, path_or_hf=None):
        self.path_or_hf = path_or_hf
        FakeDraftModel.instances.append(self)


def install_fake_llama_cpp():
    pkg = types.ModuleType("llama_cpp")
    pkg.Llama = FakeLlama
    spec_mod = types.ModuleType("llama_cpp.llama_speculative")
    spec_mod.LlamaPromptLookupDecoding = FakePromptLookup
    spec_mod.LlamaDraftModel = FakeDraftModel
    pkg.llama_speculative = spec_mod
    sys.modules["llama_cpp"] = pkg
    sys.modules["llama_cpp.llama_speculative"] = spec_mod
    return pkg


class TestSpeculativeDecoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_fake_llama_cpp()
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.model = os.path.join(cls.tmpdir.name, "model.gguf")
        cls.draft = os.path.join(cls.tmpdir.name, "draft.gguf")
        for p in (cls.model, cls.draft):
            with open(p, "w") as f:
                f.write("fake")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        FakeLlama.instances.clear()
        FakePromptLookup.instances.clear()
        FakeDraftModel.instances.clear()

    def _client(self, **kw):
        kw.setdefault("model_path", self.model)
        return LlamaCppClient(**kw)

    def test_default_is_off_and_passes_no_draft(self):
        c = self._client()
        self.assertEqual(c.speculative, "off")
        self.assertIsNone(FakeLlama.instances[-1].kwargs["draft_model"])
        self.assertEqual(FakePromptLookup.instances, [])
        self.assertEqual(FakeDraftModel.instances, [])

    def test_prompt_lookup_constructs_with_n_tokens(self):
        self._client(speculative="prompt_lookup", draft_n_tokens=6)
        llm = FakeLlama.instances[-1]
        draft = llm.kwargs["draft_model"]
        self.assertIsInstance(draft, FakePromptLookup)
        self.assertEqual(draft.num_pred_tokens, 6)

    def test_prompt_lookup_default_n_tokens(self):
        self._client(speculative="prompt_lookup")
        self.assertEqual(
            FakeLlama.instances[-1].kwargs["draft_model"].num_pred_tokens, 10)

    def test_prompt_lookup_rejects_nonpositive_n_tokens(self):
        with self.assertRaises(LLMError):
            self._client(speculative="prompt_lookup", draft_n_tokens=0)

    def test_draft_model_constructs_from_path(self):
        self._client(speculative="draft_model",
                     draft_model_path=self.draft)
        llm = FakeLlama.instances[-1]
        draft = llm.kwargs["draft_model"]
        self.assertIsInstance(draft, FakeDraftModel)
        self.assertEqual(draft.path_or_hf, self.draft)

    def test_draft_model_requires_path(self):
        with self.assertRaises(LLMError) as cm:
            self._client(speculative="draft_model")
        self.assertIn("draft_model_path", str(cm.exception))

    def test_draft_model_path_must_exist(self):
        with self.assertRaises(LLMError) as cm:
            self._client(speculative="draft_model",
                         draft_model_path="/nonexistent/draft.gguf")
        self.assertIn("not found", str(cm.exception))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(LLMError) as cm:
            self._client(speculative="eagle")
        self.assertIn("Unknown speculative mode", str(cm.exception))

    def test_chat_still_works_with_speculative_on(self):
        c = self._client(speculative="prompt_lookup")
        msg = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(msg["content"], "hello")

    def test_chat_passes_tools_through(self):
        c = self._client(speculative="prompt_lookup")
        tools = [{"type": "function", "function": {"name": "t"}}]
        c.chat([{"role": "user", "content": "hi"}], tools=tools)
        self.assertEqual(
            FakeLlama.instances[-1].last_call["tools"], tools)


class TestAgentSpeculativeWiring(unittest.TestCase):
    """Config -> LlamaCppClient wiring keeps defaults off."""

    @classmethod
    def setUpClass(cls):
        install_fake_llama_cpp()
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.model = os.path.join(cls.tmpdir.name, "model.gguf")
        with open(cls.model, "w") as f:
            f.write("fake")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        FakeLlama.instances.clear()

    def _base_cfg(self):
        from src.config import load_config
        cfg = load_config()
        cfg = dict(cfg)
        cfg["backend"] = "llamacpp"
        cfg["llamacpp"] = {"model_path": self.model}
        return cfg

    def test_agent_llamacpp_defaults_speculative_off(self):
        from src.agent import Agent
        agent = Agent(self._base_cfg())
        self.assertIsInstance(agent.llm, LlamaCppClient)
        self.assertEqual(agent.llm.speculative, "off")
        self.assertIsNone(FakeLlama.instances[-1].kwargs["draft_model"])

    def test_agent_llamacpp_passes_speculative_config(self):
        from src.agent import Agent
        cfg = self._base_cfg()
        cfg["llamacpp"]["speculative"] = "prompt_lookup"
        cfg["llamacpp"]["draft_n_tokens"] = 4
        agent = Agent(cfg)
        draft = FakeLlama.instances[-1].kwargs["draft_model"]
        self.assertIsInstance(draft, FakePromptLookup)
        self.assertEqual(draft.num_pred_tokens, 4)


if __name__ == "__main__":
    unittest.main()
