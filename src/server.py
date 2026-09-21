"""Minimal local web UI at http://127.0.0.1:8765 (stdlib only)."""
from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .config import load_config
from .agent import Agent

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hearth</title>
<style>
:root { color-scheme: light dark; }
body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px;
       margin: 0 auto; padding: 16px; display: flex; flex-direction: column;
       height: 100dvh; box-sizing: border-box; }
#log { flex: 1; overflow-y: auto; padding: 8px 0; }
.msg { margin: 8px 0; padding: 10px 14px; border-radius: 14px; line-height: 1.45;
       white-space: pre-wrap; word-wrap: break-word; }
.user { background: #0a84ff; color: #fff; margin-left: 15%; }
.bot { background: #e9e9eb; color: #111; margin-right: 15%; }
@media (prefers-color-scheme: dark) { .bot { background: #2c2c2e; color: #eee; } }
form { display: flex; gap: 8px; padding-top: 8px; }
input { flex: 1; padding: 12px; border-radius: 20px;
        border: 1px solid #ccc; font-size: 16px; }
button { padding: 0 20px; border-radius: 20px; border: none;
         background: #0a84ff; color: #fff; font-size: 16px; }
#new { margin-top: 8px; background: none; border: none; color: #888;
       font-size: 13px; cursor: pointer; }
</style></head>
<body>
<div id="log"></div>
<form id="f"><input id="i" autocomplete="off" placeholder="Ask Hearth\u2026">
<button>Send</button></form>
<button id="new">start a new chat</button>
<script>
const log = document.getElementById('log');
const input = document.getElementById('i');
function add(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls; d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight;
}
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  add('user', text);
  add('bot', '\u2026');
  const thinking = log.lastChild;
  const r = await fetch('/api/chat', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text})});
  const data = await r.json();
  thinking.textContent = data.reply;
  log.scrollTop = log.scrollHeight;
};
document.getElementById('new').onclick = async () => {
  await fetch('/api/new', {method: 'POST'});
  log.innerHTML = '';
};
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    agent: Agent | None = None
    messages: list[dict] = []

    def _send(self, body: str, ctype: str = "text/html") -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        if self.path == "/api/chat":
            try:
                msg = json.loads(raw).get("message", "")
            except json.JSONDecodeError:
                msg = ""
            reply = self.agent.run(msg, self.messages) if self.agent else "not ready"
            self._send(json.dumps({"reply": reply}), "application/json")
        elif self.path == "/api/new":
            self.messages = []
            self._send(json.dumps({"ok": True}), "application/json")
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


def main() -> None:
    cfg = load_config()
    Handler.agent = Agent(cfg)
    srv = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Hearth web UI at http://127.0.0.1:8765  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
