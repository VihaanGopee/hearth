#!/usr/bin/env python3
"""Rung tok/s measurement harness for OpenAI-compatible endpoints (Mac-side).

Same protocol as tools/measure_rung.py (the 3 speed prompts + 3 quality
sanity checks, same PASS/FAIL report), but talks to an OpenAI-compatible
``/v1/chat/completions`` endpoint instead of Ollama's ``/api/generate``.
Used to validate the Hearth 70B-on-16GB capability-ladder rungs that run
behind vmlx / mlx_lm.server (rung 4: Qwen3.6-35B-A3B-oQ2 under --smelt):

    vmlx serve ~/models/Qwen3.6-35B-A3B-oQ2 --smelt --smelt-experts 50
    python3 tools/measure_openai.py --model Qwen3.6-35B-A3B-oQ2 --target 10

Exit code: 0 if the speed target is met AND the quality sanity checks pass,
1 otherwise. The full text report is printed to stdout so it can be pasted
back as the measurement record.

Speed method: stream=True, wall-clock timing. Token counts come from the
stream's ``usage.completion_tokens`` (requested via ``stream_options``) when
the server provides it, otherwise from counting chunks that carried content
(a chunk is not a token, so the report labels which source was used). The
payload is shaped exactly like measure_rung's Ollama payload
(``eval_count`` / ``eval_duration`` in ns, plus ``response`` text) so the
shared run_rung / format_report machinery computes the same numbers.

Thinking models: Ollama's harness disables Qwen <think> via ``think: false``.
OpenAI-compatible servers have no standard equivalent, so thinking tokens —
if the served model emits them — are included in the decode tok/s figure
(they genuinely cost decode time). For the quality checks, ``<think>`` blocks
are stripped from the text before needle/count matching so a reasoning trace
can't break the sanity checks; the stripping is logged in the report.

Tested on Linux against a fake SSE server (see tests/test_measure_openai.py).

This harness measures decode tok/s, which needs token counts. The mlxl3
one-shot CLI (src/mlxl3_cli.py) exposes NO token counts and its
per-invocation model-load cost is unmeasured, so a tok/s figure would be
meaningless — --transport mlxl3 is therefore refused loudly (use
tools/eval_battery_openai.py --transport mlxl3 for the EXL3 intelligence
battery instead).
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.measure_rung import (
    MeasureError,
    format_report,
    run_rung,
)

# Re-exported so the protocol stays identical to the Ollama harness.
from tools.measure_rung import SPEED_PROMPTS, QUALITY_CHECKS  # noqa: F401

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text):
    """Remove <think>...</think> reasoning blocks from model text."""
    return _THINK_RE.sub("", text or "").strip()


def normalize_base_url(base_url):
    """Accept a root URL, a /v1 URL, or the full chat/completions path."""
    u = base_url.rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return u + "/chat/completions"
    return u + "/v1/chat/completions"


def chat_completions(base_url, model, prompt, num_predict, num_ctx,
                     api_key=None, timeout=300, meta=None):
    """One streamed /v1/chat/completions call.

    Returns an Ollama-shaped payload (``response``, ``done``, ``eval_count``,
    ``eval_duration``) so run_rung's math applies unchanged. Appends a per-call
    dict ``{"ttft_s", "tok_source"}`` to ``meta`` when provided.
    """
    url = normalize_base_url(base_url)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": num_predict,
        "stream": True,
        # Ask for usage in the terminal chunk so token counts are exact.
        # Servers that don't support stream_options just ignore it.
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        raise MeasureError(
            "cannot reach OpenAI-compatible server at %s (%s) — is it serving?"
            % (url, e.reason if hasattr(e, "reason") else e))
    t_start = time.time()
    t_first = None
    chunks_with_content = 0
    completion_tokens = None
    pieces = []
    try:
        with resp:
            if getattr(resp, "status", 200) != 200:
                raise MeasureError("server returned HTTP %s" % resp.status)
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except ValueError:
                    continue
                usage = evt.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    completion_tokens = usage["completion_tokens"]
                for choice in evt.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        if t_first is None:
                            t_first = time.time()
                        chunks_with_content += 1
                        pieces.append(content)
    except urllib.error.URLError as e:
        raise MeasureError("stream interrupted: %s" % e)
    t_end = time.time()
    text = "".join(pieces)
    if not text and completion_tokens is None:
        raise MeasureError("empty stream from %s (no content, no usage)" % url)
    tok_source = "usage" if completion_tokens is not None else "chunks"
    eval_count = completion_tokens if completion_tokens is not None \
        else chunks_with_content
    elapsed_ns = int((t_end - t_start) * 1e9)
    if meta is not None:
        meta.append({
            "ttft_s": (t_first - t_start) if t_first is not None else None,
            "tok_source": tok_source,
        })
    return {
        "response": strip_think(text),
        "done": True,
        "eval_count": eval_count,
        "eval_duration": elapsed_ns,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure sustained decode tok/s behind an "
                    "OpenAI-compatible endpoint (Hearth capability-ladder "
                    "rung validation for vmlx / mlx_lm.server).")
    p.add_argument("--model", required=True,
                   help="Model name as the server knows it "
                        "(e.g. Qwen3.6-35B-A3B-oQ2)")
    p.add_argument("--base-url", default="http://localhost:8080",
                   help="Server URL: root, /v1, or full chat/completions path "
                        "(default: http://localhost:8080)")
    p.add_argument("--target", type=float, default=10.0,
                   help="Pass threshold in decode tok/s (default: 10, "
                        "the rung-4 pass bar)")
    p.add_argument("--num-predict", type=int, default=256,
                   help="Max tokens per speed prompt (default: 256)")
    p.add_argument("--api-key", default=None,
                   help="Bearer token; default from OPENAI_API_KEY env")
    p.add_argument("--transport", choices=("http", "mlxl3"), default="http",
                   help="http: OpenAI-compatible server (default); "
                        "mlxl3: refused — the one-shot CLI exposes no token "
                        "counts, so tok/s would be meaningless")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.transport == "mlxl3":
        print("error: --transport mlxl3 is not supported by "
              "tools/measure_openai.py: the one-shot mlxl3 CLI exposes no "
              "token counts and its per-invocation model-load cost is "
              "unmeasured, so a tok/s figure would be meaningless. Use "
              "tools/eval_battery_openai.py --transport mlxl3 for the "
              "intelligence battery (the rung verdict that matters).",
              file=sys.stderr)
        return 2
    meta = []
    report = run_rung(
        args.model, args.base_url, args.num_predict, 0, args.target,
        generate_fn=lambda bu, m, pr, np_, nc: chat_completions(
            bu, m, pr, np_, nc, api_key=args.api_key, meta=meta),
    )
    print(format_report(report))
    print("STREAM META (OpenAI-compatible transport):")
    for i, (row, m) in enumerate(zip(report["speed"], meta)):
        ttft = "n/a" if m["ttft_s"] is None else "%.2fs" % m["ttft_s"]
        src = m["tok_source"]
        note = "" if src == "usage" else "  (chunk-counted: approximate)"
        print("  [%d] ttft %s, tokens from %s%s" % (i + 1, ttft, src, note))
    return 0 if report["rung_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
