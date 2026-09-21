"""The agent loop: chat with the local model, call tools, repeat."""
from __future__ import annotations
import json
from .llm import OllamaClient, LLMError
from .llamacpp_backend import LlamaCppClient
from .memory import Memory
from .tools import load_all
from .tools import registry

SYSTEM_PROMPT = """You are Hearth, a personal AI assistant running entirely on the user's own Mac. You are helpful, direct, and a little warm. You get things done with tools instead of just talking about them.

Your tools:
- read_file / write_file / edit_file / list_dir — work with files in your workspace
- run_shell — run shell commands inside the workspace (no sudo; destructive patterns blocked)
- web_search / web_fetch — look things up online
- remember / recall / forget_memory — durable memory across conversations
- estimate_fit — check whether a model fits in this machine's RAM before trying it
- schedule_add / schedule_list / schedule_remove — recurring background jobs
- browser_open / browser_snapshot / browser_click / browser_type / browser_close — headless browser (only if installed)

Rules:
- Prefer doing over asking. If a request is ambiguous in a small way, pick the reasonable interpretation and say what you assumed.
- Confirm before anything destructive or irreversible (deleting files, sending messages, spending money, publishing).
- Never invent file contents, prices, or facts — read the file or look it up.
- Keep replies short and conversational. Match the user's language.
- When you use tools, don't narrate the internals; just give the result.
- If a tool is unavailable (e.g. browser not installed), say so plainly and suggest the fix.
- When asked to remember something about the user, use the remember tool."""


class Agent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        backend = cfg.get("backend", "ollama")
        if backend == "llamacpp":
            lc = cfg.get("llamacpp", {})
            self.llm = LlamaCppClient(
                lc.get("model_path", "~/.hearth/models/model.gguf"),
                lc.get("temperature", 0.6),
                lc.get("num_ctx", 8192),
                lc.get("n_gpu_layers", -1),
                lc.get("n_threads", 0),
                lc.get("n_batch", 512),
                lc.get("cache_type_k", "q8_0"),
                lc.get("cache_type_v", "q8_0"))
            self.model_label = f"llamacpp:{lc.get('model_path', '?')}"
        else:
            oc = cfg["ollama"]
            self.llm = OllamaClient(
                oc["host"], oc["model"],
                oc.get("temperature", 0.6), oc.get("num_ctx", 16384))
            self.model_label = oc["model"]
        mem_cfg = cfg.get("memory", {})
        budget = float(mem_cfg.get("total_ram_gb", 16)) - \
            float(mem_cfg.get("reserved_gb", 5))
        self.memory = Memory(cfg["_data_dir"] / "memory.db")
        self.ctx = {
            "workspace": cfg["_workspace"],
            "data_dir": cfg["_data_dir"],
            "project_dir": cfg["_project_dir"],
            "shell_timeout": cfg["limits"]["shell_timeout"],
            "memory": self.memory,
            "memory_budget_gb": budget,
        }
        load_all(self.ctx)
        self.max_rounds = cfg["limits"]["max_tool_rounds"]
        self.max_out = cfg["limits"]["max_tool_output"]

    def run(self, user_text: str, messages: list[dict] | None = None) -> str:
        messages = messages if messages is not None else []
        if not messages:
            mem = self.memory.recent(20)
            sys_prompt = SYSTEM_PROMPT
            if mem:
                sys_prompt += ("\n\nThings you remember about the user:\n" +
                               "\n".join(f"- {m['fact']}" for m in mem))
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_text})

        for _ in range(self.max_rounds):
            try:
                msg = self.llm.chat(messages, tools=registry.tool_schemas())
            except LLMError as e:
                return f"\u26a0\ufe0f {e}"
            calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            if not calls:
                messages.append({"role": "assistant", "content": content})
                return content
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": calls})
            for tc in calls:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                try:
                    result = registry.call_tool(fn, args)
                except Exception as e:
                    result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn,
                    "content": json.dumps(result, default=str)[:self.max_out],
                })
        messages.append({"role": "assistant",
                         "content": "(stopped: tool round limit)"})
        return ("I ran out of steps on that one \u2014 "
                "try breaking it into smaller pieces.")
