"""Tool registry: name -> OpenAI-style schema + callable."""
from __future__ import annotations

REGISTRY: dict[str, dict] = {}


def register_tool(name: str, description: str, parameters: dict, fn) -> None:
    REGISTRY[name] = {
        "schema": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters.get("properties", {}),
                    "required": parameters.get("required", []),
                },
            },
        },
        "fn": fn,
    }


def tool_schemas() -> list[dict]:
    return [t["schema"] for t in REGISTRY.values()]


def call_tool(name: str, args: dict):
    if name not in REGISTRY:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    return REGISTRY[name]["fn"](**args)
