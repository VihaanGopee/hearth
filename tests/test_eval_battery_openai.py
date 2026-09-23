"""Tests for tools/eval_battery_openai.py: the OpenAI-compatible battery.

The thin HTTP layer is tested against a real local HTTP server emitting
OpenAI chat-completions JSON (non-streamed); the battery definition, check
engine, and scorecard are reused from eval_battery via import. Live-model
runs happen on the Mac via the CLI.
"""

import json
import os
import stat
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval_battery_openai import (
    BATTERY,
    BatteryError,
    _api_error_message,
    generate,
    main,
    parse_args,
    run_model,
    run_model_mlxl3,
)
from src.mlxl3_cli import Mlxl3CliClient


def chat_body(content, completion_tokens=None, think_prefix=False):
    """Build a non-streamed chat/completions response body."""
    text = ("<think>pondering</think>" if think_prefix else "") + content
    body = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if completion_tokens is not None:
        body["usage"] = {"completion_tokens": completion_tokens,
                         "prompt_tokens": 10,
                         "total_tokens": 10 + completion_tokens}
    return json.dumps(body).encode("utf-8")


def error_body_dict():
    return json.dumps({"error": {"message": "model not loaded",
                                 "type": "server_error"}}).encode("utf-8")


def error_body_str():
    return json.dumps({"error": "boom"}).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    body = b""
    status = 200
    seen = []  # (path, headers, request-body) tuples

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(length) if length else b""
        type(self).seen.append((self.path, dict(self.headers), payload))
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).body)


class _Server:
    def __init__(self, body, status=200):
        _Handler.body = body
        _Handler.status = status
        _Handler.seen = []
        self.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.srv.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.srv.server_address[1]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join()


class TestOpenAIBattery(unittest.TestCase):
    def setUp(self):
        self.servers = []
        self._saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self):
        for s in self.servers:
            s.close()
        if self._saved_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._saved_key

    def _serve(self, body, status=200):
        s = _Server(body, status)
        self.servers.append(s)
        return s

    # -- battery reuse -----------------------------------------------------
    def test_reuses_battery_definition(self):
        import tools.eval_battery as ollama_battery
        self.assertIs(BATTERY, ollama_battery.BATTERY)
        self.assertEqual(len(BATTERY), 19)

    # -- generate() ---------------------------------------------------------
    def test_generate_returns_answer_and_usage(self):
        s = self._serve(chat_body("5", completion_tokens=3))
        answer, usage = generate(s.url, "m", "q")
        self.assertEqual(answer, "5")
        self.assertEqual(usage["completion_tokens"], 3)

    def test_think_stripped_before_checks(self):
        s = self._serve(chat_body("5", think_prefix=True))
        answer, _ = generate(s.url, "m", "q")
        self.assertEqual(answer, "5")

    def test_request_shape(self):
        s = self._serve(chat_body("x", completion_tokens=1))
        generate(s.url, "m", "hi", num_predict=64)
        path, headers, body = _Handler.seen[-1]
        self.assertEqual(path, "/v1/chat/completions")
        req = json.loads(body.decode("utf-8"))
        self.assertEqual(req["model"], "m")
        self.assertEqual(req["messages"],
                         [{"role": "user", "content": "hi"}])
        self.assertEqual(req["temperature"], 0)
        self.assertEqual(req["max_tokens"], 64)
        self.assertFalse(req["stream"])
        self.assertNotIn("Authorization", headers)

    def test_api_key_header(self):
        s = self._serve(chat_body("x"))
        generate(s.url, "m", "hi", api_key="sekret")
        _, headers, _ = _Handler.seen[-1]
        self.assertEqual(headers.get("Authorization"), "Bearer sekret")

    def test_http_error_raises(self):
        s = self._serve(b"", status=500)
        with self.assertRaises(BatteryError):
            generate(s.url, "m", "hi")

    def test_api_error_dict_shape_raises(self):
        s = self._serve(error_body_dict())
        with self.assertRaises(BatteryError) as cm:
            generate(s.url, "m", "hi")
        self.assertIn("model not loaded", str(cm.exception))

    def test_api_error_string_shape_raises(self):
        s = self._serve(error_body_str())
        with self.assertRaises(BatteryError) as cm:
            generate(s.url, "m", "hi")
        self.assertIn("boom", str(cm.exception))

    def test_bad_json_raises(self):
        s = self._serve(b"not json{{")
        with self.assertRaises(BatteryError):
            generate(s.url, "m", "hi")

    def test_empty_choices_returns_empty_answer(self):
        s = self._serve(json.dumps({"choices": []}).encode())
        answer, _ = generate(s.url, "m", "hi")
        self.assertEqual(answer, "")

    def test_api_error_message_shapes(self):
        self.assertEqual(_api_error_message({"error": {"message": "m"}}), "m")
        self.assertEqual(_api_error_message({"error": "s"}), "s")

    # -- run_model() end to end with the real check engine -------------------
    def test_run_model_scores_via_real_checks(self):
        # The server always answers "5": exact-match items expecting "5"
        # pass, everything else fails.
        s = self._serve(chat_body("5", completion_tokens=2))
        items = [i for i in BATTERY if i["id"] in
                 ("math-widgets", "trick-batball", "instr-done")]
        results, total_gen = run_model(s.url, "m", items)
        by_id = {i["id"]: (a, p) for i, a, p, _ in results}
        self.assertTrue(by_id["math-widgets"][1])   # exact "5"
        self.assertTrue(by_id["trick-batball"][1])  # exact "5"
        self.assertFalse(by_id["instr-done"][1])    # exact "done"
        self.assertEqual(total_gen, 2 * 3)           # usage summed per item

    def test_run_model_request_error_marks_failed(self):
        s = self._serve(b"", status=500)
        items = [i for i in BATTERY if i["id"] == "math-widgets"]
        results, total_gen = run_model(s.url, "m", items)
        _item, answer, passed, failed = results[0]
        self.assertIsNone(answer)
        self.assertFalse(passed)
        self.assertEqual(failed, ["request error"])
        self.assertEqual(total_gen, 0)

    # -- CLI ------------------------------------------------------------------
    def test_parse_args_defaults(self):
        args = parse_args(["--models", "m"])
        self.assertEqual(args.base_url, "http://localhost:8080")
        self.assertIsNone(args.items)
        self.assertIsNone(args.api_key)
        self.assertFalse(args.show_answers)

    def test_parse_args_two_models(self):
        args = parse_args(["--models", "a", "b", "--items", "math-widgets"])
        self.assertEqual(args.models, ["a", "b"])
        self.assertEqual(args.items, "math-widgets")


FAKE_CLI = """\
#!/usr/bin/env python3
import json, os, sys, time
dest = os.environ.get("FAKE_MLXL3_ARGV_FILE")
if dest:
    with open(dest, "w") as f:
        f.write(json.dumps(sys.argv[1:]))
time.sleep(float(os.environ.get("FAKE_MLXL3_SLEEP", "0")))
if os.environ.get("FAKE_MLXL3_FAIL"):
    sys.stderr.write(os.environ["FAKE_MLXL3_FAIL"])
    sys.exit(1)
sys.stdout.write(os.environ.get("FAKE_MLXL3_REPLY", "canned reply"))
"""


class TestMlxl3BatteryTransport(unittest.TestCase):
    """--transport mlxl3 drives the battery through the mlxl3 one-shot CLI.

    The mlxl3 binary is Mac-only, so these tests run run_model_mlxl3 (and
    main()'s mlxl3 branch) against a fake CLI script. This verifies the
    transport plumbing (client construction, prompt passing, answer
    extraction, error mapping); the real one-shot stdout shape still needs
    a Mac (see src/mlxl3_cli.py).
    """

    def setUp(self):
        self._env = dict(os.environ)
        for k in ("FAKE_MLXL3_ARGV_FILE", "FAKE_MLXL3_FAIL",
                  "FAKE_MLXL3_SLEEP", "FAKE_MLXL3_REPLY"):
            os.environ.pop(k, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _fake_cli(self, tmp_dir):
        path = os.path.join(tmp_dir, "fake_mlxl3")
        with open(path, "w") as f:
            f.write(FAKE_CLI)
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        return path

    def _client(self, tmp_dir, **kw):
        kw.setdefault("cli_bin", self._fake_cli(tmp_dir))
        kw.setdefault("model", "yeasah/Qwen3.6-35B-A3B-exl3")
        kw.setdefault("max_tokens", 512)
        return Mlxl3CliClient(**kw)

    def test_scoring_via_real_checks(self):
        # The fake CLI always answers "5": exact-match items expecting "5"
        # pass, everything else fails — same shape as the HTTP test.
        import tempfile
        os.environ["FAKE_MLXL3_REPLY"] = "5"
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d)
            items = [i for i in BATTERY if i["id"] in
                     ("math-widgets", "trick-batball", "instr-done")]
            results, total_gen = run_model_mlxl3(client, items)
            by_id = {i["id"]: (a, p) for i, a, p, _ in results}
            self.assertTrue(by_id["math-widgets"][1])   # exact "5"
            self.assertTrue(by_id["trick-batball"][1])  # exact "5"
            self.assertFalse(by_id["instr-done"][1])    # exact "done"
            self.assertEqual(total_gen, 0)  # CLI exposes no token counts

    def test_think_stripped(self):
        import tempfile
        os.environ["FAKE_MLXL3_REPLY"] = "<think>pondering</think>5"
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d)
            items = [i for i in BATTERY if i["id"] == "math-widgets"]
            results, _ = run_model_mlxl3(client, items)
            _item, answer, passed, _failed = results[0]
            self.assertTrue(passed)
            self.assertEqual(answer, "5")

    def test_prompt_and_argv_shape(self):
        import tempfile
        os.environ["FAKE_MLXL3_REPLY"] = "ok"
        with tempfile.TemporaryDirectory() as d:
            argv_file = os.path.join(d, "argv.json")
            os.environ["FAKE_MLXL3_ARGV_FILE"] = argv_file
            client = self._client(d)
            items = [i for i in BATTERY if i["id"] == "math-widgets"]
            run_model_mlxl3(client, items)
            with open(argv_file) as f:
                argv = json.load(f)
            self.assertEqual(argv[0], "run")
            self.assertEqual(argv[1], "yeasah/Qwen3.6-35B-A3B-exl3")
            prompt = argv[argv.index("--prompt") + 1]
            self.assertIn(items[0]["prompt"], prompt)
            self.assertEqual(argv[argv.index("--max-tokens") + 1], "512")

    def test_cli_failure_marks_item_failed(self):
        import tempfile
        os.environ["FAKE_MLXL3_FAIL"] = "boom"
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d)
            items = [i for i in BATTERY if i["id"] == "math-widgets"]
            results, total_gen = run_model_mlxl3(client, items)
            _item, answer, passed, failed = results[0]
            self.assertIsNone(answer)
            self.assertFalse(passed)
            self.assertEqual(failed, ["request error"])
            self.assertEqual(total_gen, 0)

    def test_missing_binary_marks_item_failed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            client = Mlxl3CliClient(model="m",
                                    cli_bin=os.path.join(d, "no-such-bin"))
            items = [i for i in BATTERY if i["id"] == "math-widgets"]
            results, _ = run_model_mlxl3(client, items)
            _item, answer, passed, failed = results[0]
            self.assertIsNone(answer)
            self.assertFalse(passed)
            self.assertEqual(failed, ["request error"])

    def test_cli_timeout_marks_item_failed(self):
        import tempfile
        os.environ["FAKE_MLXL3_SLEEP"] = "2"
        with tempfile.TemporaryDirectory() as d:
            client = self._client(d, timeout=1)
            items = [i for i in BATTERY if i["id"] == "math-widgets"]
            results, _ = run_model_mlxl3(client, items)
            _item, answer, passed, failed = results[0]
            self.assertIsNone(answer)
            self.assertFalse(passed)
            self.assertEqual(failed, ["request error"])

    def test_parse_args_mlxl3_defaults(self):
        args = parse_args(["--models", "m"])
        self.assertEqual(args.transport, "http")
        self.assertIsNone(args.mlxl3_model)
        self.assertEqual(args.mlxl3_bin, "mlxl3")
        self.assertEqual(args.mlxl3_timeout, 300)

    def test_parse_args_mlxl3_options(self):
        args = parse_args(["--models", "m", "--transport", "mlxl3",
                           "--mlxl3-model", "custom", "--mlxl3-bin", "/b",
                           "--mlxl3-timeout", "42"])
        self.assertEqual(args.transport, "mlxl3")
        self.assertEqual(args.mlxl3_model, "custom")
        self.assertEqual(args.mlxl3_bin, "/b")
        self.assertEqual(args.mlxl3_timeout, 42)

    def test_main_mlxl3_end_to_end(self):
        import contextlib
        import io
        import tempfile
        os.environ["FAKE_MLXL3_REPLY"] = "5"
        with tempfile.TemporaryDirectory() as d:
            fake = self._fake_cli(d)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = main(["--transport", "mlxl3", "--models", "m",
                           "--mlxl3-bin", fake, "--items", "math-widgets"])
            self.assertEqual(rc, 0)
            text = out.getvalue()
            self.assertIn("PASS", text)
            self.assertIn("1/1", text)  # print_scorecard tally for the item
            self.assertIn("mlxl3 one-shot", text)

    def test_main_mlxl3_model_override(self):
        import contextlib
        import io
        import tempfile
        os.environ["FAKE_MLXL3_REPLY"] = "ok"
        with tempfile.TemporaryDirectory() as d:
            argv_file = os.path.join(d, "argv.json")
            os.environ["FAKE_MLXL3_ARGV_FILE"] = argv_file
            fake = self._fake_cli(d)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = main(["--transport", "mlxl3", "--models", "entry-name",
                           "--mlxl3-model", "override-name",
                           "--mlxl3-bin", fake, "--items", "math-widgets"])
            self.assertEqual(rc, 1)  # "ok" fails the exact checks
            with open(argv_file) as f:
                argv = json.load(f)
            self.assertEqual(argv[1], "override-name")


if __name__ == "__main__":
    unittest.main()
