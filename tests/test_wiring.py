"""Tests for backend wiring: defaults never break, new backends degrade cleanly."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.agent import Agent
from src.llm import LLMError


class TestWiring(unittest.TestCase):
    def test_default_backend_is_ollama(self):
        cfg = load_config()
        self.assertEqual(cfg.get("backend", "ollama"), "ollama")

    def test_agent_builds_with_defaults(self):
        cfg = load_config()
        agent = Agent(cfg)
        self.assertEqual(agent.model_label, cfg["ollama"]["model"])

    def test_llamacpp_backend_degrades_without_package(self):
        try:
            import llama_cpp  # noqa: F401
            self.skipTest("llama_cpp installed; graceful-degradation N/A")
        except ImportError:
            pass
        cfg = load_config()
        cfg = dict(cfg)
        cfg["backend"] = "llamacpp"
        with self.assertRaises(LLMError) as cm:
            Agent(cfg)
        self.assertIn("llama-cpp-python is not installed", str(cm.exception))

    def test_mlx_backend_degrades_without_package(self):
        # Order-independent: temporarily force `import mlx_lm` to fail even
        # if another test file injected a fake mlx_lm into sys.modules.
        saved = sys.modules.pop("mlx_lm", None)
        sys.modules["mlx_lm"] = None
        try:
            cfg = load_config()
            cfg = dict(cfg)
            cfg["backend"] = "mlx"
            with self.assertRaises(LLMError) as cm:
                Agent(cfg)
            self.assertIn("mlx-lm is not installed", str(cm.exception))
        finally:
            if saved is not None:
                sys.modules["mlx_lm"] = saved
            else:
                sys.modules.pop("mlx_lm", None)

    def test_estimate_fit_registered(self):
        from src.tools import registry
        from src.tools import fitcheck
        fitcheck.register({"memory_budget_gb": 11.0})
        names = [s["function"]["name"] for s in registry.tool_schemas()]
        self.assertIn("estimate_fit", names)
        res = registry.call_tool("estimate_fit", {
            "params_b": 70, "quant": "iq1_m", "n_ctx": 8192, "arch": "llama-70b"})
        self.assertTrue(res["ok"])
        self.assertIn("verdict", res)

    def test_memory_budget_flows_to_ctx(self):
        cfg = load_config()
        agent = Agent(cfg)
        self.assertIn("memory_budget_gb", agent.ctx)
        self.assertGreater(agent.ctx["memory_budget_gb"], 0)


if __name__ == "__main__":
    unittest.main()
