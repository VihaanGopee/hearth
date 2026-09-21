"""Tests for the rung tok/s measurement harness (tools/measure_rung.py).

The pure logic (rate math, quality checks, run aggregation) is tested with a
fake generate function; the thin HTTP layer is tested against a real local
HTTP server returning canned Ollama-style JSON.
"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.measure_rung import (
    MeasureError,
    QUALITY_CHECKS,
    SPEED_PROMPTS,
    check_quality,
    format_report,
    generate,
    parse_args,
    prompt_tok_s,
    run_rung,
    tok_s,
)


def canned(eval_count=200, eval_ns=10_000_000_000, response="ok",
           prompt_count=10, prompt_ns=1_000_000_000):
    return {
        "model": "qwen3:8b",
        "response": response,
        "done": True,
        "eval_count": eval_count,
        "eval_duration": eval_ns,
        "prompt_eval_count": prompt_count,
        "prompt_eval_duration": prompt_ns,
    }


class TestTokS(unittest.TestCase):
    def test_basic(self):
        # 200 tokens in 10 s -> 20.0 tok/s
        self.assertAlmostEqual(tok_s(canned()), 20.0)

    def test_fractional(self):
        self.assertAlmostEqual(tok_s(canned(eval_count=256,
                                           eval_ns=12_800_000_000)), 20.0)
        self.assertAlmostEqual(tok_s(canned(eval_count=100,
                                           eval_ns=4_000_000_000)), 25.0)

    def test_missing_or_zero_is_zero_not_crash(self):
        self.assertEqual(tok_s({}), 0.0)
        self.assertEqual(tok_s({"eval_count": 0, "eval_duration": 0}), 0.0)
        self.assertEqual(tok_s({"eval_count": 100, "eval_duration": 0}), 0.0)
        self.assertEqual(tok_s({"eval_count": "x", "eval_duration": "y"}), 0.0)

    def test_prefill_rate(self):
        self.assertAlmostEqual(prompt_tok_s(canned()), 10.0)
        self.assertEqual(prompt_tok_s({}), 0.0)


class TestQualityChecks(unittest.TestCase):
    def test_needles_pass(self):
        ok, _ = check_quality(QUALITY_CHECKS[0], "391")
        self.assertTrue(ok)

    def test_needles_fail(self):
        ok, detail = check_quality(QUALITY_CHECKS[0], "392")
        self.assertFalse(ok)
        self.assertIn("391", detail)

    def test_needles_case_insensitive(self):
        ok, _ = check_quality(QUALITY_CHECKS[1], "Paris")
        self.assertTrue(ok)

    def test_count_pass(self):
        ok, _ = check_quality(QUALITY_CHECKS[2], "quasar quasar quasar")
        self.assertTrue(ok)

    def test_count_fail(self):
        ok, detail = check_quality(QUALITY_CHECKS[2], "quasar quasar")
        self.assertFalse(ok)
        self.assertIn("2", detail)

    def test_empty_response_fails(self):
        ok, _ = check_quality(QUALITY_CHECKS[0], "")
        self.assertFalse(ok)


def fake_gen_factory(responses):
    """responses: list of payloads (or MeasureErrors) to return in order."""
    it = iter(responses)

    def fake(base_url, model, prompt, num_predict, num_ctx):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    return fake


class TestRunRung(unittest.TestCase):
    def test_speed_pass_and_mean(self):
        # 20, 25, 30 tok/s -> mean 25 -> pass at target 20
        gen = fake_gen_factory([
            canned(eval_count=200, eval_ns=10_000_000_000),
            canned(eval_count=250, eval_ns=10_000_000_000),
            canned(eval_count=300, eval_ns=10_000_000_000),
            canned(response="391"), canned(response="paris"),
            canned(response="quasar quasar quasar"),
        ])
        rep = run_rung("m", "http://x", 256, 0, 20.0, generate_fn=gen)
        self.assertAlmostEqual(rep["mean_tok_s"], 25.0)
        self.assertTrue(rep["speed_pass"])
        self.assertTrue(rep["quality_pass"])
        self.assertTrue(rep["rung_pass"])

    def test_speed_fail_below_target(self):
        gen = fake_gen_factory([
            canned(eval_count=100, eval_ns=10_000_000_000),  # 10 tok/s
            canned(eval_count=100, eval_ns=10_000_000_000),
            canned(eval_count=100, eval_ns=10_000_000_000),
            canned(response="391"), canned(response="paris"),
            canned(response="quasar quasar quasar"),
        ])
        rep = run_rung("m", "http://x", 256, 0, 20.0, generate_fn=gen)
        self.assertAlmostEqual(rep["mean_tok_s"], 10.0)
        self.assertFalse(rep["speed_pass"])
        self.assertFalse(rep["rung_pass"])

    def test_quality_fail_blocks_rung(self):
        gen = fake_gen_factory([
            canned(eval_count=300, eval_ns=10_000_000_000),
            canned(eval_count=300, eval_ns=10_000_000_000),
            canned(eval_count=300, eval_ns=10_000_000_000),
            canned(response="392"),  # wrong arithmetic
            canned(response="paris"),
            canned(response="quasar quasar quasar"),
        ])
        rep = run_rung("m", "http://x", 256, 0, 20.0, generate_fn=gen)
        self.assertTrue(rep["speed_pass"])
        self.assertFalse(rep["quality_pass"])
        self.assertFalse(rep["rung_pass"])
        arith = [r for r in rep["quality"] if r["name"] == "arithmetic"][0]
        self.assertFalse(arith["pass"])

    def test_generate_error_recorded_not_raised(self):
        gen = fake_gen_factory([
            MeasureError("boom"),
            canned(eval_count=200, eval_ns=10_000_000_000),
            canned(eval_count=200, eval_ns=10_000_000_000),
            canned(response="391"), canned(response="paris"),
            canned(response="quasar quasar quasar"),
        ])
        rep = run_rung("m", "http://x", 256, 0, 20.0, generate_fn=gen)
        self.assertEqual(rep["speed"][0]["tok_s"], 0.0)
        self.assertEqual(rep["speed"][0]["error"], "boom")
        # mean over the two good prompts only -> 20.0 -> still passes
        self.assertAlmostEqual(rep["mean_tok_s"], 20.0)
        self.assertTrue(rep["rung_pass"])

    def test_all_errors_fail(self):
        gen = fake_gen_factory([MeasureError("down")] * 6)
        rep = run_rung("m", "http://x", 256, 0, 20.0, generate_fn=gen)
        self.assertFalse(rep["speed_pass"])
        self.assertFalse(rep["rung_pass"])

    def test_report_format(self):
        gen = fake_gen_factory([
            canned(eval_count=200, eval_ns=10_000_000_000),
            canned(eval_count=200, eval_ns=10_000_000_000),
            canned(eval_count=200, eval_ns=10_000_000_000),
            canned(response="391"), canned(response="paris"),
            canned(response="quasar quasar quasar"),
        ])
        rep = run_rung("qwen3:8b", "http://x", 256, 0, 20.0, generate_fn=gen)
        text = format_report(rep)
        self.assertIn("model=qwen3:8b", text)
        self.assertIn("20.00 tok/s", text)
        self.assertIn("RESULT: RUNG PASS", text)
        self.assertIn("[arithmetic] PASS", text)


class CannedHandler(BaseHTTPRequestHandler):
    """Records the request body, replies with canned Ollama-style JSON."""
    last_body = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        CannedHandler.last_body = json.loads(self.rfile.read(length))
        payload = canned(eval_count=200, eval_ns=10_000_000_000,
                         response="391")
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class TestGenerateHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), CannedHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()

    def test_request_shape_and_parsing(self):
        base = "http://127.0.0.1:%d" % self.port
        payload = generate(base, "qwen3:8b", "hi", 256, 2048)
        self.assertEqual(payload["eval_count"], 200)
        body = CannedHandler.last_body
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertEqual(body["prompt"], "hi")
        self.assertFalse(body["stream"])
        self.assertEqual(body["options"]["num_predict"], 256)
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["num_ctx"], 2048)

    def test_num_ctx_zero_omitted(self):
        base = "http://127.0.0.1:%d" % self.port
        generate(base, "qwen3:8b", "hi", 256, 0)
        self.assertNotIn("num_ctx", CannedHandler.last_body["options"])

    def test_unreachable_server_raises_measure_error(self):
        with self.assertRaises(MeasureError):
            generate("http://127.0.0.1:1", "qwen3:8b", "hi", 8, 0, timeout=2)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.model, "qwen3:8b")
        self.assertEqual(args.base_url, "http://localhost:11434")
        self.assertEqual(args.target, 20.0)
        self.assertEqual(args.num_predict, 256)
        self.assertEqual(args.num_ctx, 0)

    def test_overrides(self):
        args = parse_args(["--model", "qwen3:14b", "--target", "20",
                           "--num-ctx", "2048"])
        self.assertEqual(args.model, "qwen3:14b")
        self.assertEqual(args.num_ctx, 2048)


class TestPromptMaterial(unittest.TestCase):
    def test_speed_prompts_nonempty(self):
        self.assertGreaterEqual(len(SPEED_PROMPTS), 3)
        for p in SPEED_PROMPTS:
            self.assertTrue(isinstance(p, str) and len(p) > 10)

    def test_quality_checks_have_names(self):
        names = [c["name"] for c in QUALITY_CHECKS]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
