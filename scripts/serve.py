"""Local development server: the viewer, plus reading and saving ground-truth labels.

Serves web/ at /, and eval/truth/<clip>.json at /truth/<clip>.json (GET and PUT).
Binds to localhost only. Not for deployment.

Usage: python scripts/serve.py [port]
"""
import json
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / "eval" / "truth"
NAME = re.compile(r"^/truth/([a-z0-9_]+)\.json$")


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        m = NAME.match(self.path.split("?")[0])
        if not m:
            return super().do_GET()
        f = TRUTH / f"{m[1]}.json"
        body = f.read_bytes() if f.exists() else json.dumps({"clip": m[1], "verified": False, "animals": []}).encode()
        self._send(200, body)

    def do_PUT(self):
        m = NAME.match(self.path)
        size = int(self.headers.get("Content-Length", 0))
        if not m or not 0 < size < 1_000_000:
            return self._send(400, b'{"error":"bad request"}')
        try:
            data = json.loads(self.rfile.read(size))
            assert isinstance(data.get("animals"), list)
        except (ValueError, AssertionError, AttributeError):
            return self._send(400, b'{"error":"bad json"}')
        TRUTH.mkdir(parents=True, exist_ok=True)
        (TRUTH / f"{m[1]}.json").write_text(json.dumps(data, indent=1))
        self._send(200, b'{"ok":true}')

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://localhost:{port}/  (labels: http://localhost:{port}/label.html)")
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT / "web"))).serve_forever()
