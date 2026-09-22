#!/usr/bin/env python3
"""Side-by-side intelligence battery for two Ollama models.

Rung speed tests only prove a model isn't broken. This battery asks whether
the bigger model is actually *smarter*: 19 checkable prompts across math
reasoning, logic, coding, instruction-following, hallucination resistance,
reading comprehension, and classic trick questions. Same prompts, temp=0,
thinking disabled, both models — then a per-category scorecard.

Usage (from the hearth repo on the Mac):
    python3 tools/eval_battery.py --models qwen3:14b qwen35-35b-a3b

The script unloads model A (``ollama stop``) before loading model B so the
two big models are never resident at the same time.
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Battery definition. Each item: id, category, prompt, checks.
# Check kinds:
#   contains      — every value must appear (case-insensitive substring)
#   contains_any  — at least one value must appear (hedging/honesty probes)
#   not_contains  — no value may appear
#   exact         — whole answer (stripped, lowercased) must equal value
#   sentence_count— number of [.!?]-terminated sentences must equal value
#   line_count    — number of non-empty lines must equal value
# ---------------------------------------------------------------------------

BATTERY = [
    # ---- math reasoning (multi-step, numeric answers) ----
    dict(id="math-trains", category="math",
         prompt=("A train leaves Chicago at 60 mph. Two hours later a second "
                 "train leaves the same station on a parallel track at 90 mph "
                 "in the same direction. How many hours after the FIRST train "
                 "left will the second train catch it? Answer with just the "
                 "final number of hours."),
         checks=[dict(kind="exact", value="6")]),
    dict(id="math-bookstore", category="math",
         prompt=("A bookstore sold 240 books on Monday. On Tuesday it sold "
                 "twice as many as Monday. On Wednesday it sold 50 fewer than "
                 "Tuesday. How many books total over the three days? Answer "
                 "with just the final number."),
         checks=[dict(kind="exact", value="1150")]),
    dict(id="math-widgets", category="math",
         prompt=("If 5 machines take 5 minutes to make 5 widgets, how many "
                 "minutes does it take 100 machines to make 100 widgets? "
                 "Answer with just the number."),
         checks=[dict(kind="exact", value="5")]),
    dict(id="math-rectangle", category="math",
         prompt=("A rectangle's length is 3 more than twice its width. Its "
                 "perimeter is 36. What is its area? Answer with just the "
                 "number."),
         checks=[dict(kind="exact", value="65")]),
    # ---- logic ----
    dict(id="logic-boxes", category="logic",
         prompt=("Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. "
                 "Every label is wrong. You may take exactly one fruit from "
                 "exactly one box. From which box (give its label) should you "
                 "take the fruit so you can then correctly relabel all three "
                 "boxes? Answer with just the label."),
         checks=[dict(kind="contains", values=["mixed"])]),
    dict(id="logic-knave", category="logic",
         prompt=("On an island, knights always tell the truth and knaves "
                 "always lie. Person A says: 'B and I are both knaves.' "
                 "Person B says nothing. Is A a knight or a knave? Answer "
                 "with one word."),
         checks=[dict(kind="exact", value="knave")]),
    dict(id="logic-schedule", category="logic",
         prompt=("Alice, Bob, and Carol each need a one-hour meeting slot. "
                 "Available slots: 9am, 10am, 11am. Alice cannot do 9am. Bob "
                 "can only do 10am. Carol cannot do 10am. Assign each person "
                 "a slot as 'Alice: X, Bob: Y, Carol: Z'."),
         checks=[dict(kind="contains",
                      values=["alice", "11", "bob", "10", "carol", "9"])]),
    # ---- coding ----
    dict(id="code-isprime", category="coding",
         prompt=("Write a Python function `is_prime(n)` that returns True "
                 "for prime numbers and False otherwise, handling n < 2. "
                 "Return only the code."),
         checks=[dict(kind="contains",
                      values=["def is_prime", "return false"])]),
    dict(id="code-tracebug", category="coding",
         prompt=("What does this function return for total([10, 20, 30])?\n\n"
                 "def total(xs):\n    s = 0\n    for i in range(len(xs)):\n"
                 "        s = i\n    return s\n\n"
                 "Answer with just the number."),
         checks=[dict(kind="exact", value="2")]),
    dict(id="code-fib", category="coding",
         prompt=("Write a Python function `fib(n)` returning the nth "
                 "Fibonacci number (fib(0)=0, fib(1)=1), iteratively with no "
                 "recursion. Return only the code."),
         checks=[dict(kind="contains",
                      values=["def fib", "return"])]),
    # ---- instruction following ----
    dict(id="instr-ocean", category="instruction",
         prompt=("Write exactly three sentences about the ocean. Do not use "
                 "the word 'blue'."),
         checks=[dict(kind="sentence_count", value=3),
                 dict(kind="not_contains", values=["blue"])]),
    dict(id="instr-planets", category="instruction",
         prompt=("List the planets of the solar system in order from the "
                 "sun, one per line, with no extra text."),
         checks=[dict(kind="line_count", value=8),
                 dict(kind="contains",
                      values=["mercury", "venus", "earth", "mars",
                              "jupiter", "saturn", "uranus", "neptune"])]),
    dict(id="instr-done", category="instruction",
         prompt="Reply with the single word 'done' and nothing else.",
         checks=[dict(kind="exact", value="done")]),
    # ---- hallucination resistance (honesty probes) ----
    dict(id="halluc-accord", category="honesty",
         prompt="Tell me about the 2019 Oslo Quantum Computing Accord.",
         checks=[dict(kind="contains_any",
                      values=["no record", "don't have", "do not have",
                              "not aware", "fictional", "doesn't exist",
                              "does not exist", "cannot verify",
                              "can't verify", "no evidence",
                              "i don't know", "i do not know",
                              "made up", "not real"])]),
    dict(id="halluc-nobel", category="honesty",
         prompt="Who won the 2031 Nobel Prize in Physics?",
         checks=[dict(kind="contains_any",
                      values=["hasn't happened", "has not happened",
                              "not yet", "in the future", "can't know",
                              "cannot know", "cannot predict",
                              "don't know", "do not know", "not awarded",
                              "hasn't been awarded"])]),
    # ---- reading comprehension ----
    dict(id="read-tower", category="comprehension",
         prompt=("PASSAGE:\nThe Verdant Tower was completed in 1988. Its "
                 "architect, Mara Ellison, was born in 1941 and retired in "
                 "2003. The tower's copper facade was chosen to echo the "
                 "harbor warehouses Ellison admired as a child.\n\n"
                 "In what decade of her life did Ellison complete the tower? "
                 "Answer like '40s'."),
         checks=[dict(kind="contains", values=["40"])]),
    dict(id="read-alloy", category="comprehension",
         prompt=("PASSAGE:\nDr. Osei's team first synthesized the alloy "
                 "ferrite-9 in a basement lab in Gdansk in 1974, though the "
                 "breakthrough was not published until a 1979 paper written "
                 "in Krakow. The Gdansk lab was demolished in 1991.\n\n"
                 "Which city hosted the lab where the alloy was first "
                 "synthesized, and in what year? Answer as 'City, Year'."),
         checks=[dict(kind="contains", values=["gdansk", "1974"])]),
    # ---- trick / cognitive-reflection questions ----
    dict(id="trick-months", category="trick",
         prompt=("How many months have 28 days? Answer with just the number."),
         checks=[dict(kind="exact", value="12")]),
    dict(id="trick-batball", category="trick",
         prompt=("A bat and a ball cost $1.10 in total. The bat costs $1.00 "
                 "more than the ball. How much does the ball cost? Answer "
                 "with just the amount in cents."),
         checks=[dict(kind="exact", value="5")]),
]

CATEGORIES = sorted({item["category"] for item in BATTERY})

# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------

def _norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _norm_exact(text):
    # Exact-answer checks ignore trailing punctuation: "Knave." must count
    # the same as "knave". A right answer in polite wrapping is still right.
    return re.sub(r"\s+", " ", text.strip().lower().rstrip(".!?;:"))


def check_item(item, answer):
    """Return (passed, failed_check_descriptions)."""
    normed = _norm(answer)
    failed = []
    for check in item["checks"]:
        kind = check["kind"]
        if kind == "contains":
            missing = [v for v in check["values"] if v.lower() not in normed]
            if missing:
                failed.append("missing %r" % (missing,))
        elif kind == "contains_any":
            if not any(v.lower() in normed for v in check["values"]):
                failed.append("no hedging/honesty phrase found")
        elif kind == "not_contains":
            present = [v for v in check["values"] if v.lower() in normed]
            if present:
                failed.append("forbidden present %r" % (present,))
        elif kind == "exact":
            if _norm_exact(answer) != check["value"].lower():
                failed.append("expected exactly %r, got %r"
                              % (check["value"], answer.strip()[:60]))
        elif kind == "sentence_count":
            sentences = [s for s in re.split(r"[.!?]+", answer.strip())
                         if s.strip()]
            if len(sentences) != check["value"]:
                failed.append("expected %d sentences, got %d"
                              % (check["value"], len(sentences)))
        elif kind == "line_count":
            lines = [ln for ln in answer.strip().splitlines() if ln.strip()]
            if len(lines) != check["value"]:
                failed.append("expected %d lines, got %d"
                              % (check["value"], len(lines)))
        else:
            raise ValueError("unknown check kind: %r" % kind)
    return (not failed), failed


# ---------------------------------------------------------------------------
# Ollama HTTP layer
# ---------------------------------------------------------------------------

class BatteryError(Exception):
    pass


def extract_response(payload):
    """Pull the text out of an /api/generate payload.

    API-level errors (e.g. model failed to load, template errors) must
    surface loudly — never silently become an empty answer that the
    checkers then score as a model failure.
    """
    if not isinstance(payload, dict):
        raise BatteryError("unexpected Ollama response shape: %r" % (payload,))
    if payload.get("error"):
        raise BatteryError("Ollama error: %s" % payload["error"])
    return payload.get("response", "")


def generate(base_url, model, prompt, num_predict=512, num_ctx=4096,
             timeout=300):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Qwen3-family models think by default; the <think> block would eat
        # the token budget and pollute exact-match checks. Disable it.
        "think": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0,
            "num_ctx": num_ctx,
        },
    }
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
        raise BatteryError(
            "cannot reach Ollama at %s (%s) — is `ollama serve` running?"
            % (base_url, e.reason if hasattr(e, "reason") else e))
    except (ValueError, UnicodeDecodeError) as e:
        raise BatteryError("bad JSON from Ollama: %s" % e)
    if not isinstance(payload, dict):
        raise BatteryError("unexpected Ollama response shape: %r" % (payload,))
    # Surface API errors loudly; return the full payload so callers can also
    # inspect the `thinking` field (a model that wraps its whole reply in
    # <think> tags leaves `response` empty when thinking is stripped).
    if payload.get("error"):
        raise BatteryError("Ollama error: %s" % payload["error"])
    return payload


def ollama_stop(model):
    """Unload a model so the next one gets the full RAM budget."""
    try:
        subprocess.run(["ollama", "stop", model], check=False,
                       capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def select_items(item_ids):
    """Filter BATTERY to a comma-separated list of ids (None/empty = all)."""
    if not item_ids:
        return list(BATTERY)
    wanted = [i.strip() for i in item_ids.split(",") if i.strip()]
    known = {item["id"] for item in BATTERY}
    unknown = [i for i in wanted if i not in known]
    if unknown:
        raise BatteryError("unknown item ids: %s (choose from: %s)"
                           % (", ".join(unknown), ", ".join(sorted(known))))
    return [item for item in BATTERY if item["id"] in wanted]


def run_model(base_url, model, items, show_answers=False):
    results = []
    print("MODEL %s (%d items)" % (model, len(items)), flush=True)
    for i, item in enumerate(items, 1):
        try:
            payload = generate(base_url, model, item["prompt"])
            answer = extract_response(payload)
            thinking = payload.get("thinking") or ""
        except BatteryError as e:
            print("  [%s] ERROR: %s" % (item["id"], e), flush=True)
            results.append((item, None, False, ["request error"]))
            continue
        passed, failed = check_item(item, answer)
        mark = "PASS" if passed else "FAIL"
        print("  [%2d/%2d] %-16s %-4s  %s" % (
            i, len(items), item["id"], mark,
            ("; ".join(failed) if failed else _norm(answer)[:70])),
            flush=True)
        if show_answers:
            print("       answer: %s" % answer.strip(), flush=True)
            if thinking.strip():
                print("       thinking: %s" % thinking.strip()[:2000],
                      flush=True)
        results.append((item, answer, passed, failed))
    return results


def scorecard(results):
    by_cat = {}
    for item, _answer, passed, _failed in results:
        cat = item["category"]
        ok, total = by_cat.get(cat, (0, 0))
        by_cat[cat] = (ok + (1 if passed else 0), total + 1)
    return by_cat


def print_scorecard(name, results):
    by_cat = scorecard(results)
    total_ok = sum(ok for ok, _ in by_cat.values())
    total = sum(t for _, t in by_cat.values())
    print("\n== %s: %d/%d ==" % (name, total_ok, total))
    for cat in sorted(by_cat):
        ok, t = by_cat[cat]
        bar = "#" * ok + "-" * (t - ok)
        print("  %-14s %2d/%-2d  [%s]" % (cat, ok, t, bar))
    return total_ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Side-by-side intelligence battery for two Ollama models.")
    ap.add_argument("--models", nargs="+", required=True, metavar="MODEL",
                    help="one or two Ollama model names (two = side-by-side)")
    ap.add_argument("--items", default=None, metavar="ID,...",
                    help="comma-separated item ids to run (default: all 19)")
    ap.add_argument("--show-answers", action="store_true",
                    help="print each model's full answer text")
    ap.add_argument("--base-url", default="http://localhost:11434")
    args = ap.parse_args(argv)

    if len(args.models) not in (1, 2):
        ap.error("--models takes one or two model names")
    items = select_items(args.items)

    all_results = {}
    for idx, model in enumerate(args.models):
        all_results[model] = run_model(args.base_url, model, items,
                                       show_answers=args.show_answers)
        if len(args.models) == 2 and idx == 0:
            print("\nUnloading %s before loading %s ..."
                  % (args.models[0], args.models[1]))
            ollama_stop(args.models[0])

    if len(args.models) == 1:
        print_scorecard(args.models[0], all_results[args.models[0]])
        return

    model_a, model_b = args.models
    print("\n================ SCORECARD ================")
    score_a = print_scorecard(model_a, all_results[model_a])
    score_b = print_scorecard(model_b, all_results[model_b])
    print("\nTOTAL: %s %d  vs  %s %d" % (model_a, score_a, model_b, score_b))
    if score_b > score_a:
        print("VERDICT: %s wins by %d — bigger model is genuinely smarter "
              "on this battery." % (model_b, score_b - score_a))
    elif score_a > score_b:
        print("VERDICT: %s wins by %d — the bigger model did NOT earn its "
              "keep here." % (model_a, score_a - score_b))
    else:
        print("VERDICT: tie — no intelligence difference detected on this "
              "battery.")
    print("\nNote: exact-match checks are strict by design; eyeball any FAIL "
          "lines above before declaring a model dumb — a right answer in "
          "unexpected wrapping fails the check, not the model.")


if __name__ == "__main__":
    sys.exit(main())
