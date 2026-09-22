"""Browser automation via Playwright, with a persistent profile.

Install with: pip install playwright && playwright install chromium
If Playwright isn't installed this module registers nothing.

The browser uses a persistent profile directory so cookies and logins
survive between sessions. That is the whole anti-detection story: a
returning, logged-in user in a real Chromium does not look like a bot.
There is no fingerprint spoofing here — just calm, low-volume browsing.

Profile location: <data_dir>/browser-profile by default, overridable with
HEARTH_BROWSER_PROFILE. Set HEARTH_BROWSER_HEADED=1 to show the window
(macOS) so the user can watch or take over — e.g. to solve a captcha
by hand, which is the honest answer to captchas no model can beat.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from .registry import register_tool


def profile_dir(ctx: dict) -> Path:
    override = os.environ.get("HEARTH_BROWSER_PROFILE")
    if override:
        return Path(override).expanduser()
    data_dir = ctx.get("data_dir")
    if data_dir:
        return Path(data_dir) / "browser-profile"
    return Path.home() / ".hearth" / "browser-profile"


def headed() -> bool:
    return os.environ.get("HEARTH_BROWSER_HEADED", "0") == "1"


def register(ctx: dict) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return

    state: dict = {"pw": None, "context": None, "page": None}

    def _page():
        if state["page"] is None:
            profile = profile_dir(ctx)
            profile.mkdir(parents=True, exist_ok=True)
            state["pw"] = sync_playwright().start()
            # Persistent context: cookies/logins survive restarts, and the
            # browser presents as an ordinary returning user.
            state["context"] = state["pw"].chromium.launch_persistent_context(
                str(profile),
                headless=not headed(),
            )
            state["page"] = state["context"].new_page()
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

    def browser_screenshot(full_page: bool = False):
        """Capture a PNG screenshot (base64) — the future 'eyes' hook."""
        p = _page()
        png = p.screenshot(full_page=full_page)
        return {"ok": True, "url": p.url,
                "png_base64": base64.b64encode(png).decode("ascii")}

    def browser_close():
        if state["context"]:
            state["context"].close()
        if state["pw"]:
            state["pw"].stop()
        state.update(pw=None, context=None, page=None)
        return {"ok": True}

    register_tool(
        "browser_open", "Open a URL in Chromium (persistent profile: stays logged in).",
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
        "browser_screenshot", "Capture a PNG screenshot of the page (base64).",
        {"properties": {"full_page": {"type": "boolean"}}, "required": []},
        browser_screenshot)
    register_tool(
        "browser_close", "Close the browser.",
        {"properties": {}, "required": []}, browser_close)
