"""Web search + page fetching."""
from __future__ import annotations
import re
from html.parser import HTMLParser
from .registry import register_tool


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            s = data.strip()
            if s:
                self.parts.append(s)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts))


def register(ctx: dict) -> None:
    def web_search(query: str, max_results: int = 5):
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*renamed to.*ddgs.*")
                    from duckduckgo_search import DDGS  # older package name
        except ImportError:
            return {"ok": False,
                    "error": "Web search needs duckduckgo-search: pip install duckduckgo-search"}
        try:
            with DDGS() as ddg:
                hits = list(ddg.text(query, max_results=max(1, min(max_results, 10))))
        except Exception as e:
            return {"ok": False, "error": f"Search failed: {e}"}
        return {"ok": True, "results": [
            {"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")}
            for h in hits]}

    def web_fetch(url: str, max_chars: int = 8000):
        import requests
        try:
            r = requests.get(url, timeout=20,
                             headers={"User-Agent": "Mozilla/5.0 (Macintosh) Hearth/1.0"})
            r.raise_for_status()
        except Exception as e:
            return {"ok": False, "error": f"Fetch failed: {e}"}
        parser = _Text()
        try:
            parser.feed(r.text)
        except Exception:
            pass
        return {"ok": True, "url": url, "text": parser.text()[:max_chars]}

    register_tool(
        "web_search", "Search the web and return titles, URLs and snippets.",
        {"properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "description": "1-10, default 5"}},
         "required": ["query"]},
        web_search)
    register_tool(
        "web_fetch", "Fetch a web page and return its readable text.",
        {"properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "description": "Default 8000"}},
         "required": ["url"]},
        web_fetch)
