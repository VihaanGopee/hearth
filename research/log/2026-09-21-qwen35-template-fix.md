# 2026-09-21 — 35B tool-call 500: root-caused to Qwen3.5 chat template

## Symptom (Justin's Mac, ~19:00 PDT)
Hearth on `qwen35-35b-a3b-full`: plain chat worked, but the first tool call
crashed the follow-up request:
`Ollama 500 ... Jinja Exception: No user query found in messages.`

## Root cause
Qwen3.5's bundled chat template is hostile to agentic tool loops (documented
upstream as "tool calling chat template is broken" for Qwen3.5-35B-A3B).
Two concrete defects vs. our message shape:
1. `tool_call.arguments|items` assumes arguments is a mapping, but Ollama's
   OpenAI-compatible endpoint delivers arguments as a JSON **string**.
   Local jinja2 repro of the official template with our exact messages raises
   `TypeError: Can only get item pairs from a mapping` at render time.
2. The `multi_step_tool` guard raises `No user query found in messages`
   instead of degrading gracefully — the #1 crash reported for Qwen3.x with
   agent frameworks (LangChain/AutoGen/OpenClaw/Claude Code all hit it).

Note: the official template's detection loop *should* find our real user
message, so the exact 500 Justin saw likely comes from a template variant
(GGUF-embedded or Ollama-fallback) with stricter logic at a different line.
Either way the fix is the same: override the template.

## Fix
`research/templates/qwen3.5-agentic-chat-template.jinja` — official
Qwen3.5-35B-A3B template plus two surgical agentic fixes:
- arguments: iterate when mapping, emit verbatim when a JSON string.
- user-query guard: fall back to last-message-as-query instead of raising.
Tool-call output syntax (`<tool_call>/<function=>/<parameter=>`) unchanged.
Validated locally with jinja2: 4/4 render cases pass (string args, mapping
args + parallel calls, pure tool chain with no user message, plain chat).

Also silenced the `duckduckgo_search` -> `ddgs` rename RuntimeWarning in
`src/tools/web.py` (cosmetic noise on every web_search).

## Justin's action (Mac)
```bash
cd ~/Desktop/hearth && git pull
# save the fixed template next to the Modelfile, then:
cat >> Modelfile <<'EOF'
TEMPLATE ./qwen3.5-agentic-chat-template.jinja
EOF
cp ~/workspace/local-agent/research/templates/qwen3.5-agentic-chat-template.jinja ./qwen3.5-agentic-chat-template.jinja
ollama create qwen35-35b-a3b-full -f Modelfile   # instant, blob deduped
```
Then retry the Apple/Wikipedia prompt. If it still 500s, run
`ollama show qwen35-35b-a3b-full --template | sed -n '70,85p'` and send the
output — that shows exactly which template is in effect.

## VM note
Playwright Chromium download failed on the VM again (second attempt,
proc_313fa906cb33, empty ms-playwright dir). Does not block the Mac path;
real-browser validation still needs Justin's Mac.
