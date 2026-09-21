#!/usr/bin/env python3
"""Rung tok/s measurement harness (Mac-side).

Measures sustained *decode* tokens/sec for an Ollama model via the
``/api/generate`` endpoint (``stream=false``), over a fixed prompt set, plus
a small quality sanity check. Used to validate the Hearth 70B-on-16GB
capability-ladder rungs on Justin's M1 Pro:

  rung 1: 7B/8B-class sustaining >= 20 tok/s  (candidate: qwen3:8b)
  rung 2: 14B-class sustaining >= 20 tok/s
  rung 3: 35B-class MoE at usable speed

Usage (on the Mac, with ``ollama serve`` running)::

    python3 tools/measure_rung.py --model qwen3:8b --target 20

Exit code: 0 if the speed target is met AND the quality sanity checks pass,
1 otherwise. The full text report is printed to stdout so it can be pasted
back as the measurement record.

Tested on Linux against a fake HTTP server (see tests/test_measure_rung.py);
the Ollama ``/api/generate`` request/response shape it relies on
(``eval_count`` / ``eval_duration`` in nanoseconds) is Ollama's long-stable
generate API.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Fixed evaluation material (deterministic at temperature 0)
# ---------------------------------------------------------------------------

SPEED_PROMPTS = [
    "Explain why the sky is blue, in exactly two sentences.",
    "Write a Python function `fib(n)` that returns the nth Fibonacci number. "
    "Include a brief docstring.",
    "PASSAGE:\n"
    "The M1 Pro is a system-on-a-chip with a unified memory architecture: "
    "the CPU, GPU, and Neural Engine share one pool of high-bandwidth memory, "
    "so a model loaded once is visible to every processor without copies. "
    "Memory bandwidth on the M1 Pro is about 200 gigabytes per second, which "
    "sets the ceiling for token generation speed: each generated token must "
    "stream the model's weights through the processor, so tokens per second "
    "is roughly bandwidth divided by model size. Quantization shrinks the "
    "model, raising that ceiling; the cost is precision, and the research "
    "question is how far precision can fall before quality collapses.\n"
    "TASK: Summarize the passage above in exactly three bullet points.",
]

# Each check is {"name", "prompt", "needles"} (all needles must appear,
# case-insensitive) or {"name", "prompt", "count_word", "count"} (the word
# must appear exactly `count` times, case-insensitive).
QUALITY_CHECKS = [
    {
        "name": "arithmetic",
        "prompt": "What is 17 multiplied by 23? Reply with only the number, "
                  "nothing else.",
        "needles": ["391"],
    },
    {
        "name": "factual",
        "prompt": "What is the capital of France? Reply with only the city "
                  "name, nothing else.",
        "needles": ["paris"],
    },
    {
        "name": "repeat-3x",
        "prompt": "Repeat the word 'quasar' exactly three times, separated by "
                  "single spaces. Reply with nothing else.",
        "count_word": "quasar",
        "count": 3,
    },
]


class MeasureError(Exception):
    """The measurement could not be taken (server down, bad response, ...)."""


# ---------------------------------------------------------------------------
# Core math / evaluation (pure, fully unit-testable)
# ---------------------------------------------------------------------------

def tok_s(payload):
    """Decode tokens/sec from an /api/generate response payload.

    Ollama reports ``eval_count`` (generated tokens) and ``eval_duration``
    (nanoseconds). Missing/zero values -> 0.0 (never divide by zero).
    """
    try:
        count = float(payload.get("eval_count") or 0)
        dur_ns = float(payload.get("eval_duration") or 0)
    except (TypeError, ValueError):
        return 0.0
    if count <= 0 or dur_ns <= 0:
        return 0.0
    return count / (dur_ns / 1e9)


def prompt_tok_s(payload):
    """Prompt-processing (prefill) tokens/sec; informational only."""
    try:
        count = float(payload.get("prompt_eval_count") or 0)
        dur_ns = float(payload.get("prompt_eval_duration") or 0)
    except (TypeError, ValueError):
        return 0.0
    if count <= 0 or dur_ns <= 0:
        return 0.0
    return count / (dur_ns / 1e9)


def check_quality(check, response_text):
    """Run one quality check against a model response. Returns (pass, detail)."""
    text = (response_text or "").lower()
    if "count_word" in check:
        n = text.split().count(check["count_word"].lower())
        ok = n == check["count"]
        return ok, "found %d, expected %d" % (n, check["count"])
    missing = [w for w in check.get("needles", []) if w.lower() not in text]
    ok = not missing
    return ok, ("all needles found" if ok else "missing: %s" % ", ".join(missing))


def run_rung(model, base_url, num_predict, num_ctx, target,
             speed_prompts=None, quality_checks=None, generate_fn=None):
    """Run the full rung measurement. Returns a report dict.

    ``generate_fn(base_url, model, prompt, num_predict, num_ctx)`` performs
    one generation and returns the parsed /api/generate JSON payload; it
    defaults to the real HTTP implementation. A fake can be injected for
    tests. Per-prompt MeasureErrors are recorded (0.0 tok/s) rather than
    aborting the run.
    """
    speed_prompts = SPEED_PROMPTS if speed_prompts is None else speed_prompts
    quality_checks = QUALITY_CHECKS if quality_checks is None else quality_checks
    gen = generate if generate_fn is None else generate_fn

    speed_rows = []
    for i, prompt in enumerate(speed_prompts):
        try:
            payload = gen(base_url, model, prompt, num_predict, num_ctx)
            tps = tok_s(payload)
            speed_rows.append({
                "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                "tok_s": tps,
                "eval_count": payload.get("eval_count"),
                "prefill_tok_s": prompt_tok_s(payload),
                "error": None,
            })
        except MeasureError as e:
            speed_rows.append({
                "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                "tok_s": 0.0,
                "eval_count": None,
                "prefill_tok_s": 0.0,
                "error": str(e),
            })

    ok_speeds = [r["tok_s"] for r in speed_rows if r["error"] is None]
    mean_tps = sum(ok_speeds) / len(ok_speeds) if ok_speeds else 0.0
    speed_pass = bool(ok_speeds) and mean_tps >= target

    quality_rows = []
    for check in quality_checks:
        try:
            payload = gen(base_url, model, check["prompt"], 64, num_ctx)
            ok, detail = check_quality(check, payload.get("response", ""))
            quality_rows.append({"name": check["name"], "pass": ok,
                                 "detail": detail, "error": None})
        except MeasureError as e:
            quality_rows.append({"name": check["name"], "pass": False,
                                 "detail": "", "error": str(e)})
    quality_pass = bool(quality_rows) and all(r["pass"] for r in quality_rows)

    return {
        "model": model,
        "target_tok_s": target,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "speed": speed_rows,
        "mean_tok_s": mean_tps,
        "speed_pass": speed_pass,
        "quality": quality_rows,
        "quality_pass": quality_pass,
        "rung_pass": speed_pass and quality_pass,
    }


def format_report(report):
    """Human-readable measurement record (this is what gets pasted back)."""
    L = []
    L.append("Rung measurement: model=%s target=%.1f tok/s "
             "(num_predict=%d, num_ctx=%s, temp=0)"
             % (report["model"], report["target_tok_s"],
                report["num_predict"],
                report["num_ctx"] if report["num_ctx"] else "server-default"))
    L.append("SPEED (decode):")
    for i, r in enumerate(report["speed"]):
        if r["error"]:
            L.append("  [%d] ERROR: %s  prompt=%r" % (i + 1, r["error"], r["prompt"]))
        else:
            L.append("  [%d] %6.2f tok/s  (eval %s tok, prefill %6.1f tok/s)  %r"
                     % (i + 1, r["tok_s"], r["eval_count"],
                        r["prefill_tok_s"], r["prompt"]))
    L.append("  mean decode: %.2f tok/s  -> %s (>= %.1f)"
             % (report["mean_tok_s"],
                "PASS" if report["speed_pass"] else "FAIL",
                report["target_tok_s"]))
    L.append("QUALITY sanity (temp=0):")
    for r in report["quality"]:
        if r["error"]:
            L.append("  [%s] ERROR: %s" % (r["name"], r["error"]))
        else:
            L.append("  [%s] %s (%s)" % (r["name"],
                                        "PASS" if r["pass"] else "FAIL",
                                        r["detail"]))
    L.append("RESULT: RUNG %s" % ("PASS" if report["rung_pass"] else "FAIL"))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# HTTP layer (thin; exercised against a fake server in tests)
# ---------------------------------------------------------------------------

def generate(base_url, model, prompt, num_predict, num_ctx, timeout=300):
    """One non-streaming /api/generate call. Returns the parsed JSON payload."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Qwen3-family models think by default; the <think> block would eat
        # the token budget (quality checks use num_predict=64) and pollute
        # the speed measurement. Disable it for a clean, comparable number.
        "think": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0,
        },
    }
    if num_ctx:
        body["options"]["num_ctx"] = num_ctx
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise MeasureError(
            "cannot reach Ollama at %s (%s) — is `ollama serve` running?"
            % (base_url, e.reason if hasattr(e, "reason") else e))
    except (ValueError, UnicodeDecodeError) as e:
        raise MeasureError("bad JSON from Ollama: %s" % e)
    if not isinstance(payload, dict) or not payload.get("done", True):
        raise MeasureError("unexpected Ollama response shape: %r" % (payload,))
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure sustained decode tok/s for an Ollama model "
                    "(Hearth capability-ladder rung validation).")
    p.add_argument("--model", default="qwen3:8b",
                   help="Ollama model tag (default: qwen3:8b)")
    p.add_argument("--base-url", default="http://localhost:11434",
                   help="Ollama server URL (default: http://localhost:11434)")
    p.add_argument("--target", type=float, default=20.0,
                   help="Pass threshold in decode tok/s (default: 20)")
    p.add_argument("--num-predict", type=int, default=256,
                   help="Tokens to generate per speed prompt (default: 256)")
    p.add_argument("--num-ctx", type=int, default=0,
                   help="Context size; 0 = server default (default: 0)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_rung(args.model, args.base_url, args.num_predict,
                      args.num_ctx, args.target, generate_fn=generate)
    print(format_report(report))
    return 0 if report["rung_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
