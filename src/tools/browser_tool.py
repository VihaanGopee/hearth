"""Optional browser automation via Playwright.

Install with: pip install playwright && playwright install chromium
If Playwright isn't installed this module registers nothing.
"""
from __future__ import annotations
import json
from .registry import register_tool


def register(ctx: dict) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return

    state: dict = {"pw": None, "browser": None, "page": None}

    def _page():
        if state["page"] is None:
            state["pw"] = sync_playwright().start()
            state["browser"] = state["pw"].chromium.launch(headless=True)
            state["page"] = state["browser"].new_page()
        return state["page"]

    def browser_open(url: str):
        p = _page()
        p.goto(url, wait_until="domcontentloaded", timeout=30000)
        return {"ok": True, "url": p.url, "title": p.title()}

    def browser_snapshot():
        p = _page()
        try:
            snap = p.accessibility.snapshot()
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "url": p.url,
                "snapshot": json.dumps(snap, default=str)[:8000]}

    def browser_click(selector: str):
        p = _page()
        p.click(selector, timeout=10000)
        p.wait_for_timeout(1000)
        return {"ok": True, "url": p.url}

    def browser_type(selector: str, text: str, submit: bool = False):
        p = _page()
        p.fill(selector, text, timeout=10000)
        if submit:
            p.keyboard.press("Enter")
            p.wait_for_timeout(1500)
        return {"ok": True, "url": p.url}

    def browser_close():
        if state["browser"]:
            state["browser"].close()
        if state["pw"]:
            state["pw"].stop()
        state.update(pw=None, browser=None, page=None)
        return {"ok": True}

    register_tool(
        "browser_open", "Open a URL in a headless browser.",
        {"properties": {"url": {"type": "string"}}, "required": ["url"]},
        browser_open)
    register_tool(
        "browser_snapshot", "Return the page's accessibility tree (interactive elements).",
        {"properties": {}, "required": []}, browser_snapshot)
    register_tool(
        "browser_click", "Click an element by CSS selector.",
        {"properties": {"selector": {"type": "string"}}, "required": ["selector"]},
        browser_click)
    register_tool(
        "browser_type", "Type text into an input (CSS selector); optionally submit.",
        {"properties": {
            "selector": {"type": "string"},
            "text": {"type": "string"},
            "submit": {"type": "boolean"}},
         "required": ["selector", "text"]},
        browser_type)
    register_tool(
        "browser_close", "Close the headless browser.",
        {"properties": {}, "required": []}, browser_close)
