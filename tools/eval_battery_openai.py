#!/usr/bin/env python3
"""Side-by-side intelligence battery for OpenAI-compatible endpoints (Mac-side).

Same 19 prompts, same temp=0 discipline, same check engine and scorecard as
tools/eval_battery.py (the Ollama variant), but talks to an OpenAI-compatible
``/v1/chat/completions`` endpoint instead of Ollama's ``/api/generate``.
Used for the Hearth capability-ladder rung-4 intelligence verdict: the
vmlx-served Qwen3.6-35B-A3B-oQ2 under --smelt scored with the same checks as
rungs 1-3:

    vmlx serve ~/models/Qwen3.6-35B-A3B-oQ2 --smelt --smelt-experts 50
    python3 tools/eval_battery_openai.py --models Qwen3.6-35B-A3B-oQ2

Then compare the scorecard against the recorded rung-3 battery numbers
(`qwen35-35b-a3b` via tools/eval_battery.py) — that side-by-side is the
intelligence verdict, not a pass/fail line.

Non-streamed requests (timing isn't measured here; fewer moving parts than
the SSE path). Two-model mode works only if the server can hold or swap both
models — a single-model vmlx server serves whichever model it was started
with, so for cross-endpoint comparisons run the tool twice and compare the
scorecards. There is no `ollama stop` equivalent: the server owns residency.

Thinking models: OpenAI-compatible servers have no standard think-off
switch, so if the served model emits ``<think>`` blocks they are stripped
from the answer before checking (same as tools/measure_openai.py); the
strip is noted in the per-item output.

Exit code: 0 when all checks pass, 1 otherwise (and 2 for CLI/request
errors). The full per-item log is printed to stdout so it can be pasted
back as the measurement record.

Tested on Linux against a fake chat/completions server (see
tests/test_eval_battery_openai.py).
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval_battery import (  # noqa: F401  (re-exported: same battery)
    BATTERY,
    CATEGORIES,
    BatteryError,
    check_item,
    print_scorecard,
    select_items,
)
from tools.measure_openai import normalize_base_url, strip_think


def _api_error_message(payload):
    """Extract a human-readable message from an OpenAI-style error payload."""
    err = payload.get("error")
    if isinstance(err, dict):
        return err.get("message", json.dumps(err))
    return err


def generate(base_url, model, prompt, num_predict=512, api_key=None,
             timeout=300):
    """One non-streamed /v1/chat/completions call.

    Returns (answer_text, usage_dict). The answer has <think> blocks
    stripped so reasoning traces can't break the sanity checks. API-level
    errors surface loudly — never silently become an empty answer.
    """
    url = normalize_base_url(base_url)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": num_predict,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                raise BatteryError("server returned HTTP %s" % resp.status)
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise BatteryError(
            "cannot reach OpenAI-compatible server at %s (%s) — is it serving?"
            % (url, e.reason if hasattr(e, "reason") else e))
    except (ValueError, UnicodeDecodeError) as e:
        raise BatteryError("bad JSON from server: %s" % e)
    if not isinstance(payload, dict):
        raise BatteryError("unexpected response shape: %r" % (payload,))
    if payload.get("error"):
        raise BatteryError("server error: %s" % _api_error_message(payload))
    choices = payload.get("choices") or []
    content = ((choices[0].get("message") or {}) if choices else {}).get(
        "content") or ""
    usage = payload.get("usage") or {}
    return strip_think(content), usage


def run_model(base_url, model, items, api_key=None, show_answers=False):
    results = []
    total_gen = 0
    print("MODEL %s (%d items)" % (model, len(items)), flush=True)
    for i, item in enumerate(items, 1):
        try:
            answer, usage = generate(base_url, model, item["prompt"],
                                     api_key=api_key)
            total_gen += usage.get("completion_tokens") or 0
        except BatteryError as e:
            print("  [%s] ERROR: %s" % (item["id"], e), flush=True)
            results.append((item, None, False, ["request error"]))
            continue
        passed, failed = check_item(item, answer)
        mark = "PASS" if passed else "FAIL"
        print("  [%2d/%2d] %-16s %-4s  %s" % (
            i, len(items), item["id"], mark,
            ("; ".join(failed) if failed else answer[:70].replace("\n", " "))),
            flush=True)
        if show_answers:
            print("       answer: %s" % answer.strip()[:4000], flush=True)
        results.append((item, answer, passed, failed))
    return results, total_gen


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Side-by-side intelligence battery over an "
                    "OpenAI-compatible endpoint (Hearth capability-ladder "
                    "rung validation for vmlx / mlx_lm.server).")
    p.add_argument("--models", nargs="+", required=True, metavar="MODEL",
                   help="one or two model names as the server knows them "
                        "(two = side-by-side, only if the server can serve "
                        "both)")
    p.add_argument("--items", default=None, metavar="ID,...",
                   help="comma-separated item ids to run (default: all 19)")
    p.add_argument("--show-answers", action="store_true",
                   help="print each model's full answer text")
    p.add_argument("--base-url", default="http://localhost:8080",
                   help="Server URL: root, /v1, or full chat/completions path "
                        "(default: http://localhost:8080)")
    p.add_argument("--api-key", default=None,
                   help="Bearer token; default from OPENAI_API_KEY env")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if len(args.models) not in (1, 2):
        print("error: --models takes one or two model names",
              file=sys.stderr)
        return 2
    items = select_items(args.items)

    all_results = {}
    try:
        for model in args.models:
            results, total_gen = run_model(args.base_url, model, items,
                                           api_key=args.api_key,
                                           show_answers=args.show_answers)
            all_results[model] = results
            print("  (completion tokens generated: %d)" % total_gen)
    except BatteryError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2

    ok = all(passed
             for results in all_results.values()
             for (_item, _ans, passed, _failed) in results)
    if len(args.models) == 1:
        print_scorecard(args.models[0], all_results[args.models[0]])
        return 0 if ok else 1

    model_a, model_b = args.models
    print("\n================ SCORECARD ================")
    score_a = print_scorecard(model_a, all_results[model_a])
    score_b = print_scorecard(model_b, all_results[model_b])
    print("\nTOTAL: %s %d  vs  %s %d" % (model_a, score_a, model_b, score_b))
    if score_b > score_a:
        print("VERDICT: %s wins by %d — bigger model is genuinely smarter "
              "on this battery."
              % (model_b, score_b - score_a))
    elif score_a > score_b:
        print("VERDICT: %s wins by %d — the bigger model did NOT earn its "
              "keep here."
              % (model_a, score_a - score_b))
    else:
        print("VERDICT: tie — no intelligence difference detected on this "
              "battery.")
    print("\nNote: exact-match checks are strict by design; eyeball any FAIL "
          "lines above before declaring a model dumb — a right answer in "
          "unexpected wrapping fails the check, not the model.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
