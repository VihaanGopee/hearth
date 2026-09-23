"""Tests for the mlxl3 CLI-subprocess bridge.

mlxl3 is Mac-only, so the tests drive Mlxl3CliClient against a fake CLI
script (written to tmp_path) that records argv, sleeps, fails, or echoes
a canned reply on env-var instruction. This verifies the subprocess
plumbing (arg passing, prompt rendering, stdout capture, error paths);
real one-shot stdout parsing still needs a Mac with mlxl3 installed.
"""
import json
import os
import stat
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMError
from src.mlxl3_cli import Mlxl3CliClient, _render_prompt

FAKE_CLI = """\
#!/usr/bin/env python3
import json, os, sys, time
time.sleep(float(os.environ.get("FAKE_MLXL3_SLEEP", "0")))
if os.environ.get("FAKE_MLXL3_FAIL"):
    sys.stderr.write(os.environ["FAKE_MLXL3_FAIL"])
    sys.exit(1)
# Line 1 of stdout: the argv we were called with, for assertion.
sys.stdout.write(json.dumps(sys.argv[1:]) + "\\n")
sys.stdout.write(os.environ.get("FAKE_MLXL3_REPLY", "canned reply"))
"""


def _write_fake_cli(tmp_dir):
    path = os.path.join(tmp_dir, "fake_mlxl3")
    with open(path, "w") as f:
        f.write(FAKE_CLI)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


class Mlxl3CliTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        os.environ.pop("FAKE_MLXL3_FAIL", None)
        os.environ.pop("FAKE_MLXL3_SLEEP", None)
        os.environ.pop("FAKE_MLXL3_REPLY", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _client(self, tmp_path, **kw):
        kw.setdefault("cli_bin", _write_fake_cli(tmp_path))
        kw.setdefault("model", "yeasah/Qwen3.6-35B-A3B-exl3")
        return Mlxl3CliClient(**kw)

    def test_missing_model_rejected(self):
        with self.assertRaises(LLMError):
            Mlxl3CliClient(model="")

    def test_missing_cli_bin_rejected(self):
        with self.assertRaises(LLMError):
            Mlxl3CliClient(model="m", cli_bin="")

    def test_tools_refused_loudly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d)
            with self.assertRaises(LLMError) as cm:
                client.chat([{"role": "user", "content": "hi"}],
                            tools=[{"type": "function",
                                    "function": {"name": "f"}}])
            self.assertIn("no tool surface", str(cm.exception))

    def test_prompt_rendering_and_argv(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d, max_tokens=64,
                                  extra_args=["--some-flag"])
            reply = client.chat([
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "and 3+3?"},
            ])
            # The fake CLI's stdout line 1 is the argv it received.
            argv = json.loads(reply["content"].splitlines()[0])
            self.assertEqual(argv[0], "run")
            self.assertEqual(argv[1], "yeasah/Qwen3.6-35B-A3B-exl3")
            prompt = argv[argv.index("--prompt") + 1]
            self.assertIn("system: Be brief.", prompt)
            self.assertIn("user: 2+2?", prompt)
            self.assertIn("assistant: 4", prompt)
            self.assertIn("user: and 3+3?", prompt)
            self.assertIn("--max-tokens", argv)
            self.assertIn("64", argv)
            self.assertIn("--some-flag", argv)
            self.assertEqual(reply["role"], "assistant")
            self.assertIsNone(reply["tool_calls"])

    def test_max_tokens_omitted_when_none(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d)
            reply = client.chat([{"role": "user", "content": "hi"}])
            argv = json.loads(reply["content"].splitlines()[0])
            self.assertNotIn("--max-tokens", argv)

    def test_canned_reply_becomes_content(self):
        import tempfile
        os.environ["FAKE_MLXL3_REPLY"] = "  the answer is 42\n"
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d)
            reply = client.chat([{"role": "user", "content": "hi"}])
            # Content is the full stdout stripped (argv line + reply).
            self.assertTrue(reply["content"].endswith("the answer is 42"))

    def test_nonzero_exit_surfaces_stderr(self):
        import tempfile
        os.environ["FAKE_MLXL3_FAIL"] = "model file not found: boom"
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d)
            with self.assertRaises(LLMError) as cm:
                client.chat([{"role": "user", "content": "hi"}])
            self.assertIn("exited 1", str(cm.exception))
            self.assertIn("boom", str(cm.exception))

    def test_missing_binary_is_llmerror(self):
        client = Mlxl3CliClient(model="m", cli_bin="/nonexistent/mlxl3-bin")
        with self.assertRaises(LLMError) as cm:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("not found", str(cm.exception))

    def test_timeout_is_llmerror(self):
        import tempfile
        os.environ["FAKE_MLXL3_SLEEP"] = "5"
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d, timeout=1)
            with self.assertRaises(LLMError) as cm:
                client.chat([{"role": "user", "content": "hi"}])
            self.assertIn("timed out", str(cm.exception))

    def test_render_prompt_handles_tool_messages(self):
        prompt = _render_prompt([
            {"role": "assistant", "content": None,
             "tool_calls": [{"name": "f"}]},
            {"role": "user", "content": "go"},
        ])
        self.assertIn("assistant: ", prompt)
        self.assertIn("user: go", prompt)


if __name__ == "__main__":
    unittest.main()
