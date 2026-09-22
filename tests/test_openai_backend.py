"""Tests for the OpenAI-compatible HTTP backend, against a stub server."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.llm import LLMError
from src.openai_backend import OpenAICompatClient, _normalize_base_url


class _StubState:
    """Per-server mutable state: next response body/status, last request."""
    def __init__(self):
        self.body = {"choices": [{"message": {"role": "assistant",
                                              "content": "hello"}}]}
        self.status = 200
        self.last_path = None
        self.last_headers = {}
        self.last_json = None


class _StubHandler(BaseHTTPRequestHandler):
    state: _StubState = None

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler convention
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        _StubHandler.state.last_path = self.path
        _StubHandler.state.last_headers = dict(self.headers)
        try:
            _StubHandler.state.last_json = json.loads(raw or b"{}")
        except ValueError:
            _StubHandler.state.last_json = None
        body = json.dumps(_StubHandler.state.body).encode()
        self.send_response(_StubHandler.state.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the suite output clean
        pass


class _StubServer:
    def __init__(self):
        self.state = _StubState()
        _StubHandler.state = self.state
        self.server = HTTPServer(("127.0.0.1", 0), _StubHandler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                        daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.thread.join()


class TestUrlNormalization(unittest.TestCase):
    def test_bare_root(self):
        self.assertEqual(_normalize_base_url("http://x:8000"),
                         "http://x:8000/v1/chat/completions")

    def test_trailing_slash(self):
        self.assertEqual(_normalize_base_url("http://x:8000/"),
                         "http://x:8000/v1/chat/completions")

    def test_v1_suffix(self):
        self.assertEqual(_normalize_base_url("http://x:8000/v1"),
                         "http://x:8000/v1/chat/completions")

    def test_full_path_passthrough(self):
        self.assertEqual(
            _normalize_base_url("http://x:8000/v1/chat/completions"),
            "http://x:8000/v1/chat/completions")


class TestConstructor(unittest.TestCase):
    def test_missing_base_url(self):
        with self.assertRaises(LLMError):
            OpenAICompatClient("", "model")

    def test_missing_model(self):
        with self.assertRaises(LLMError):
            OpenAICompatClient("http://x:8000", "")


class TestChatAgainstStub(unittest.TestCase):
    def setUp(self):
        self.stub = _StubServer()
        self.addCleanup(self.stub.close)

    def _client(self, **kw):
        kw.setdefault("base_url", self.stub.url)
        kw.setdefault("model", "test-model")
        return OpenAICompatClient(**kw)

    def test_basic_chat(self):
        msg = self._client().chat([{"role": "user", "content": "hi"}])
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "hello")

    def test_payload_shape(self):
        self._client(temperature=0.2, max_tokens=50).chat(
            [{"role": "user", "content": "hi"}])
        payload = self.stub.state.last_json
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 50)
        self.assertFalse(payload["stream"])
        # Provider-neutral: no Ollama-specific fields leak through.
        self.assertNotIn("options", payload)

    def test_max_tokens_absent_when_unset(self):
        self._client().chat([{"role": "user", "content": "hi"}])
        self.assertNotIn("max_tokens", self.stub.state.last_json)

    def test_tools_passed_through(self):
        tools = [{"type": "function",
                  "function": {"name": "f", "parameters": {}}}]
        self._client().chat([{"role": "user", "content": "hi"}], tools=tools)
        self.assertEqual(self.stub.state.last_json["tools"], tools)

    def test_tool_calls_reply_returned_as_is(self):
        self.stub.state.body = {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "f",
                                         "arguments": '{"a": 1}'}}]}}]}
        msg = self._client().chat([{"role": "user", "content": "go"}])
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "f")

    def test_path_is_chat_completions(self):
        self._client().chat([{"role": "user", "content": "hi"}])
        self.assertEqual(self.stub.state.last_path, "/v1/chat/completions")

    def test_http_error_becomes_llmerror(self):
        self.stub.state.status = 500
        self.stub.state.body = {"error": "boom"}
        with self.assertRaises(LLMError) as cm:
            self._client().chat([{"role": "user", "content": "hi"}])
        self.assertIn("500", str(cm.exception))

    def test_malformed_response_becomes_llmerror(self):
        for body in ({"no": "choices"}, {"choices": []}):
            with self.subTest(body=body):
                self.stub.state.body = body
                with self.assertRaises(LLMError):
                    self._client().chat([{"role": "user", "content": "hi"}])

    def test_connection_refused_becomes_llmerror(self):
        # Fully close the listening socket: the next connect is refused.
        self.stub.close()
        self.stub.server.server_close()
        with self.assertRaises(LLMError) as cm:
            self._client(timeout=2).chat([{"role": "user", "content": "hi"}])
        self.assertIn("Can't reach", str(cm.exception))

    def test_api_key_header_sent(self):
        self._client(api_key="sekret").chat(
            [{"role": "user", "content": "hi"}])
        self.assertEqual(
            self.stub.state.last_headers.get("Authorization"), "Bearer sekret")

    def test_no_auth_header_without_key(self):
        self._client().chat([{"role": "user", "content": "hi"}])
        self.assertNotIn("Authorization", self.stub.state.last_headers)


class TestWiring(unittest.TestCase):
    def test_cascade_spec_builds_openai_client(self):
        from src.agent import _build_llm_client, _spec_label
        client = _build_llm_client(
            {"backend": "openai",
             "openai": {"base_url": "http://localhost:8000",
                        "model": "oQ2-smelt"}})
        self.assertIsInstance(client, OpenAICompatClient)
        self.assertEqual(_spec_label(
            {"backend": "openai",
             "openai": {"base_url": "http://localhost:8000",
                        "model": "oQ2-smelt"}}), "oQ2-smelt")

    def test_cascade_spec_missing_fields_errors(self):
        from src.agent import _build_llm_client
        with self.assertRaises(LLMError):
            _build_llm_client({"backend": "openai", "openai": {}})


if __name__ == "__main__":
    unittest.main()
