"""Tool package: loads every tool module against the agent context."""
from __future__ import annotations
from . import registry  # noqa: F401
from . import files, shell, web, memory_tools, schedule, browser_tool, fitcheck

_MODULES = (files, shell, web, memory_tools, schedule, browser_tool, fitcheck)


def load_all(ctx: dict) -> None:
    for mod in _MODULES:
        reg = getattr(mod, "register", None)
        if reg is not None:
            reg(ctx)
