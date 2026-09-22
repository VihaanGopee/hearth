"""Tests for tools/measure_openai.py: the OpenAI-compatible rung harness.

The thin HTTP layer is tested against a real local HTTP server emitting
SSE chunks (OpenAI chat-completions style); the shared run_rung/format_report
machinery is reused from measure_rung and exercised via injection.
"""

import io
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.measure_openai import (
    MeasureError,
    chat_completions,
    normalize_base_url,
    parse_args,
    strip_think,
)
from tools.measure_rung import run_rung


def sse_body(chunks, usage_tokens=None, think_prefix=False):
    """Build an SSE response body: content chunks + optional usage + [DONE]."""
    lines = []
    if think_prefix:
        lines.append('data: {"choices":[{"delta":{"content":"<think>hmm</think>"}}]}')
    for c in chunks:
        lines.append('data: {"choices":[{"delta":{"content":%s}}]}'
                     % json.dumps(c))
    if usage_tokens is not None:
        lines.append('data: {"choices":[],"usage":{"completion_tokens":%d}}'
                     % usage_tokens)
    lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode("utf-8")


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
        self.send_header("Content-Type", "text/event-stream")
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
        self.thread.join()


class TestOpenAIMeasure(unittest.TestCase):
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

    # -- URL normalization -------------------------------------------------
    def test_normalize_base_url(self):
        self.assertEqual(normalize_base_url("http://h:8080"),
                         "http://h:8080/v1/chat/completions")
        self.assertEqual(normalize_base_url("http://h:8080/"),
                         "http://h:8080/v1/chat/completions")
        self.assertEqual(normalize_base_url("http://h:8080/v1"),
                         "http://h:8080/v1/chat/completions")
        self.assertEqual(normalize_base_url("http://h:8080/v1/chat/completions"),
                         "http://h:8080/v1/chat/completions")

    # -- think stripping ----------------------------------------------------
    def test_strip_think(self):
        self.assertEqual(strip_think("<think>reasoning</think>391"), "391")
        self.assertEqual(strip_think("no think here"), "no think here")
        self.assertEqual(strip_think(None), "")

    # -- generate_fn --------------------------------------------------------
    def test_usage_token_count_preferred(self):
        s = self._serve(sse_body(["hello", " world"], usage_tokens=42))
        meta = []
        payload = chat_completions(s.url, "m", "hi", 64, 0, meta=meta)
        self.assertEqual(payload["response"], "hello world")
        self.assertEqual(payload["eval_count"], 42)  # usage, not 2 chunks
        self.assertGreater(payload["eval_duration"], 0)
        self.assertEqual(meta[0]["tok_source"], "usage")
        self.assertIsNotNone(meta[0]["ttft_s"])

    def test_chunk_count_fallback(self):
        s = self._serve(sse_body(["a", "b", "c"]))
        meta = []
        payload = chat_completions(s.url, "m", "hi", 64, 0, meta=meta)
        self.assertEqual(payload["eval_count"], 3)
        self.assertEqual(meta[0]["tok_source"], "chunks")

    def test_think_stripped_from_quality_text(self):
        s = self._serve(sse_body(["391"], usage_tokens=5, think_prefix=True))
        payload = chat_completions(s.url, "m", "hi", 64, 0)
        self.assertEqual(payload["response"], "391")

    def test_api_key_header(self):
        s = self._serve(sse_body(["x"], usage_tokens=1))
        chat_completions(s.url, "m", "hi", 64, 0, api_key="sekret")
        path, headers, body = _Handler.seen[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers.get("Authorization"), "Bearer sekret")
        req = json.loads(body.decode("utf-8"))
        self.assertEqual(req["model"], "m")
        self.assertEqual(req["max_tokens"], 64)
        self.assertTrue(req["stream"])

    def test_no_auth_header_without_key(self):
        s = self._serve(sse_body(["x"], usage_tokens=1))
        chat_completions(s.url, "m", "hi", 64, 0)
        _, headers, _ = _Handler.seen[-1]
        self.assertNotIn("Authorization", headers)

    def test_http_error_raises(self):
        s = self._serve(b"", status=500)
        with self.assertRaises(MeasureError):
            chat_completions(s.url, "m", "hi", 64, 0)

    def test_empty_stream_raises(self):
        s = self._serve(b"data: [DONE]\n")
        with self.assertRaises(MeasureError):
            chat_completions(s.url, "m", "hi", 64, 0)

    # -- integration with the shared rung machinery -------------------------
    def test_run_rung_with_openai_generate(self):
        body = sse_body(["tok"], usage_tokens=50)
        s = self._serve(body)

        def gen(base_url, model, prompt, num_predict, num_ctx):
            return chat_completions(base_url, model, prompt,
                                    num_predict, num_ctx)

        report = run_rung("m", s.url, 64, 0, 0.0,  # target 0: speed passes
                          speed_prompts=["say hi"],
                          quality_checks=[{"name": "n", "prompt": "q",
                                           "needles": ["tok"]}],
                          generate_fn=gen)
        self.assertTrue(report["rung_pass"])
        self.assertEqual(report["speed"][0]["eval_count"], 50)
        self.assertGreater(report["mean_tok_s"], 0)

    # -- CLI ----------------------------------------------------------------
    def test_parse_args_defaults(self):
        args = parse_args(["--model", "m"])
        self.assertEqual(args.target, 10.0)  # rung-4 pass bar
        self.assertEqual(args.base_url, "http://localhost:8080")
        self.assertEqual(args.num_predict, 256)


if __name__ == "__main__":
    unittest.main()
