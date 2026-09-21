"""Shell tool, confined to the workspace with a denylist."""
from __future__ import annotations
import subprocess
from .registry import register_tool

DENY = ("sudo", "rm -rf /", "rm -rf ~", "mkfs", "dd if=", ":(){",
        "shutdown", "reboot", "halt", "poweroff")


def register(ctx: dict) -> None:
    timeout = ctx.get("shell_timeout", 60)

    def run_shell(command: str):
        for bad in DENY:
            if bad in command:
                return {"ok": False, "error": f"Blocked dangerous pattern: {bad!r}"}
        try:
            p = subprocess.run(
                command, shell=True, cwd=str(ctx["workspace"]),
                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Timed out after {timeout}s"}
        return {"ok": p.returncode == 0,
                "returncode": p.returncode,
                "stdout": (p.stdout or "")[-4000:],
                "stderr": (p.stderr or "")[-4000:]}

    register_tool(
        "run_shell",
        "Run a shell command inside the agent workspace (macOS/Linux). "
        "Use for scripting, data processing, git, conversions. "
        "Dangerous patterns (sudo, rm -rf /, shutdown…) are blocked.",
        {"properties": {"command": {"type": "string", "description": "The shell command to run"}},
         "required": ["command"]},
        run_shell)
