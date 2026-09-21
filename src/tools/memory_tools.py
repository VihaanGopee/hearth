"""Memory tools: remember / recall / forget."""
from __future__ import annotations
from .registry import register_tool


def register(ctx: dict) -> None:
    mem = ctx["memory"]

    def remember(fact: str):
        fid = mem.add(fact)
        return {"ok": True, "id": fid}

    def recall(query: str):
        return {"ok": True, "memories": mem.search(query)}

    def forget_memory(fact_id: int):
        ok = mem.forget(fact_id)
        return {"ok": ok, "id": fact_id}

    register_tool(
        "remember", "Save a durable fact or preference about the user "
                   "(name, likes, ongoing projects…). Use sparingly, only for things "
                   "worth remembering across conversations.",
        {"properties": {"fact": {"type": "string"}}, "required": ["fact"]},
        remember)
    register_tool(
        "recall", "Search saved memories for a topic.",
        {"properties": {"query": {"type": "string"}}, "required": ["query"]},
        recall)
    register_tool(
        "forget_memory", "Delete a saved memory by its id.",
        {"properties": {"fact_id": {"type": "integer"}}, "required": ["fact_id"]},
        forget_memory)
