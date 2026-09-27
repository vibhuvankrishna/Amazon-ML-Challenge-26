"""Local receiver for browser-proxied OneDrive downloads."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(r"C:\Users\vibhu\amlc26-data")
HOST, PORT = "127.0.0.1", 8765
LOG = ROOT / "logs" / "upload_server.log"
LOG.parent.mkdir(parents=True, exist_ok=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        msg = "%s - %s\n" % (self.address_string(), fmt % args)
        LOG.write_text(LOG.read_text(encoding="utf-8") + msg if LOG.exists() else msg, encoding="utf-8")
        print(msg, end="", flush=True)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/status"):
            body = json.dumps({"ok": True, "root": str(ROOT)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            self.send_response(404)
            self.end_headers()
            return
        qs = parse_qs(parsed.query)
        rel = unquote((qs.get("path") or ["missing"])[0]).lstrip("/\\")
        expected = int((qs.get("size") or ["0"])[0])
        dest = (ROOT / rel).resolve()
        if not str(dest).startswith(str(ROOT.resolve())):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"bad path")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        length = int(self.headers.get("Content-Length", "0"))
        print(f"UPLOAD start {rel} content-length={length} expected={expected}", flush=True)
        written = 0
        with open(dest, "wb") as out:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                remaining -= len(chunk)
                if written % (64 * 1024 * 1024) < 8 * 1024 * 1024:
                    print(f"  {rel}: {written/1e6:.0f} MB", flush=True)
        ok = expected == 0 or written == expected
        print(f"UPLOAD done {rel} bytes={written} ok={ok}", flush=True)
        body = json.dumps({"path": rel, "bytes": written, "ok": ok}).encode()
        self.send_response(200 if ok else 500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"Serving on http://{HOST}:{PORT} root={ROOT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
