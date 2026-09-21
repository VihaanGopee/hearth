# Hearth — your own local AI agent

Hearth is a personal AI agent that runs **entirely on your Mac**: no cloud account,
no subscription, your data never leaves the machine. It chats, works with files,
runs shell commands, searches the web, remembers things about you, runs scheduled
jobs, and can drive a headless browser — powered by a local model through Ollama.

Built for a **16 GB M-series Mac** (M1 Pro and similar).

---

## What it can do

| Capability | How |
|---|---|
| Chat | Terminal REPL (`./run.sh`) or web UI (`./run.sh web`) |
| Files | Read / write / edit / list, sandboxed to `~/hearth/workspace` |
| Shell | Run commands in the workspace (dangerous patterns blocked, 60s timeout) |
| Web | Search (DuckDuckGo) and fetch pages as text |
| Memory | SQLite-backed facts it remembers across conversations |
| Scheduling | Recurring jobs via **launchd** on macOS (`schedule_add`) |
| Browser | Headless Chromium via Playwright (optional install) |

## What it can't do (honest version)

- It's powered by an ~8B-parameter local model, not a frontier cloud model. It's
  genuinely useful for concrete tasks, but weaker at long reasoning chains,
  ambiguous requests, and flawless tool use. Keep tasks small and concrete.
- No built-in Gmail/Calendar connectors — those need OAuth app setup and aren't
  included. The shell + browser tools can cover some of it with effort.
- Browser automation is basic (no captchas, logins are fragile).
- Scheduled jobs run headless with the same small model — good for briefings and
  reminders, not deep research.

---

## Setup (macOS, ~10 minutes)

### 1. Install Ollama

Download from [ollama.com](https://ollama.com/download) and open the app
(or `brew install ollama`), then pull a model:

```bash
ollama pull qwen3:8b
```

Model guide for 16 GB unified memory:

| Model | Size | When to pick it |
|---|---|---|
| `qwen3:8b` (default) | ~5.2 GB | Best all-round: good tool calling, fits with headroom |
| `llama3.1:8b` | ~4.9 GB | Solid alternative if you prefer Llama |
| `qwen3:4b` | ~2.6 GB | Tighter RAM situations / faster responses |
| `devstral:24b` etc. | ~14 GB+ | Too big for 16 GB — don't |

Change the model in `config.yaml` (`ollama.model`). Keep `num_ctx: 16384`
unless you mostly chat with other apps closed — then `32768` is fine.

### 2. Install Python deps and run

```bash
cd /path/to/local-agent
./run.sh
```

The first run creates a virtualenv and installs `requirements.txt`
(`requests`, `pyyaml`, `duckduckgo-search`). Then:

```
Hearth — local agent (qwen3:8b). Type /quit to exit, /new for a fresh chat.

You: remember that I like espresso
You: search the web for M4 MacBook deals and save the best three to deals.md
```

For the web UI instead: `./run.sh web`, then open http://127.0.0.1:8765.

### 3. Optional: browser automation

```bash
./run.sh  # (once, to create the .venv)
.venv/bin/pip install playwright
.venv/bin/playwright install chromium
```

The `browser_*` tools register automatically when Playwright is present.

---

## Scheduling example

Ask in chat:

> "Every weekday at 8am, check the top tech headlines and save a briefing to
> workspace/briefing-YYYY-MM-DD.md"

Hearth calls `schedule_add` with `at_time: "08:30"`, which installs a launchd
agent (`~/Library/LaunchAgents/com.hearth.job.<name>.plist`). Logs land in
`~/.hearth/jobs/`. Manage with `schedule_list` / `schedule_remove`, or ask
Hearth in chat.

## The 70B quest (experimental)

Hearth is actively being developed toward running a **70B-class model on this
16 GB Mac**. Two pieces of that are already in the codebase:

- **`backend: llamacpp`** in `config.yaml` — a direct llama.cpp backend
  (via `llama-cpp-python`) that unlocks the full GGUF quant zoo (`IQ1_M`,
  `TQ1_0`, …), KV-cache quantization, and Metal GPU layers. On a
  unified-memory Mac, GPU layers don't cost extra RAM.
- **`estimate_fit` tool** — the agent can now do the RAM math itself
  (weights + KV cache + runtime vs your usable budget) before trying a model.

To try a 70B experiment on your Mac:

```bash
# 1. llama.cpp with Metal support
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
# 2. Download a GGUF, e.g. a 70B IQ1_M (~15 GB) from Hugging Face,
#    into ~/.hearth/models/
# 3. First, ask the agent: "will llama-70b IQ1_M at 8k context fit?"
#    (it will run estimate_fit and give you an honest verdict)
# 4. If the math is close, flip config.yaml: backend: "llamacpp",
#    set llamacpp.model_path, and run.
```

Honest status: dense 70B at sub-2-bit quants is ~15 GB on disk, and macOS
leaves ~10–11 GB usable — so it likely doesn't fit *yet*. The realistic
near-term wins are 30–35B MoE models at 2-bit mixed precision (~9 GB, see
`research/model_profiles.yaml`). Progress is tracked in
`research/ROADMAP_70B.md`, and every change is a git commit — if an
experiment breaks something, `git log` + `git revert` gets you back.

## Files & data

- `~/hearth/workspace/` — the agent's working files (sandbox boundary)
- `~/.hearth/memory.db` — everything it remembers about you (SQLite, yours)
- `~/.hearth/jobs/` — scheduled-job logs
- `config.yaml` — model, context size, limits

## Project layout

```
local-agent/
  config.yaml        # model, paths, limits
  requirements.txt
  run.sh             # launcher (terminal chat or web UI)
  src/
    agent.py         # the ReAct loop
    llm.py           # Ollama client
    llamacpp_backend.py # direct llama.cpp client (70B quest)
    cli.py           # terminal chat
    server.py        # local web UI (stdlib only)
    run_once.py      # one-shot prompt runner (for scheduled jobs)
    config.py        # config loading
    memory.py        # SQLite memory
    tools/
      registry.py    # tool registry
      files.py       # read/write/edit/list (sandboxed)
      shell.py       # shell (denylist + timeout)
      web.py         # search + fetch
      memory_tools.py
      schedule.py    # launchd / cron jobs
      browser_tool.py# Playwright (optional)
      fitcheck.py    # estimate_fit: model RAM-budget math
  research/          # 70B-on-16GB quest: roadmap, profiles, dated logs
```

## Troubleshooting

- **"Can't reach Ollama"** — open the Ollama app (or run `ollama serve`), then
  `ollama pull qwen3:8b`.
- **Slow first reply** — the model loads into memory on first use (~10–30s),
  then stays warm.
- **Tool calls misbehaving** — smaller models sometimes fumble JSON args;
  `qwen3:8b` is the most reliable of the 16-GB-friendly options. Keep tool
  schemas simple and tasks concrete.
- **Out of memory** — drop to `qwen3:4b` or lower `num_ctx` to `8192`.
