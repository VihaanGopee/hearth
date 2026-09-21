"""File tools, sandboxed to the agent workspace."""
from __future__ import annotations
from pathlib import Path
from .registry import register_tool

MAX_READ = 20000


def _safe(ctx: dict, p: str) -> Path:
    ws = Path(ctx["workspace"]).resolve()
    target = (ws / p).resolve()
    if target != ws and ws not in target.parents:
        raise ValueError("Path escapes the agent workspace")
    return target


def register(ctx: dict) -> None:
    def read_file(path: str):
        t = _safe(ctx, path)
        if not t.is_file():
            return {"ok": False, "error": f"Not a file: {path}"}
        text = t.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "path": path, "content": text[:MAX_READ]}

    def write_file(path: str, content: str):
        t = _safe(ctx, path)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}

    def list_dir(path: str = "."):
        t = _safe(ctx, path)
        if not t.is_dir():
            return {"ok": False, "error": f"Not a directory: {path}"}
        items = sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in t.iterdir() if not p.name.startswith(".")
        )
        return {"ok": True, "path": path, "items": items[:200]}

    def edit_file(path: str, old_text: str, new_text: str):
        t = _safe(ctx, path)
        if not t.is_file():
            return {"ok": False, "error": f"Not a file: {path}"}
        text = t.read_text(encoding="utf-8", errors="replace")
        if old_text not in text:
            return {"ok": False, "error": "old_text not found in file"}
        t.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return {"ok": True, "path": path}

    register_tool("read_file", "Read a text file from the agent workspace.",
                  {"properties": {"path": {"type": "string", "description": "Workspace-relative path"}},
                   "required": ["path"]}, read_file)
    register_tool("write_file", "Create or overwrite a text file in the agent workspace.",
                  {"properties": {
                      "path": {"type": "string"},
                      "content": {"type": "string"}},
                   "required": ["path", "content"]}, write_file)
    register_tool("list_dir", "List files in a workspace directory.",
                  {"properties": {"path": {"type": "string", "description": "Defaults to workspace root"}},
                   "required": []}, list_dir)
    register_tool("edit_file", "Replace the first occurrence of old_text with new_text in a file.",
                  {"properties": {
                      "path": {"type": "string"},
                      "old_text": {"type": "string"},
                      "new_text": {"type": "string"}},
                   "required": ["path", "old_text", "new_text"]}, edit_file)
