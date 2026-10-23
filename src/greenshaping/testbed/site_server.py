"""Site tool server: one HTTP/1.1 endpoint per site holding a table partition (Sec. 6.1).

POST /tool  body {"family": ..., "params": {...}}  ->  {"result": ..., "matching_rows": n}

HTTP/1.1 keep-alive gives one persistent connection per orchestrator-site link.
TLS: pass `certfile`/`keyfile` (the paper requires TLS on every link). Shaping of
replies happens in the transport layer (testbed.transport), not here.
"""
from __future__ import annotations

import json
import logging
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from .queries import run_query

log = logging.getLogger(__name__)


def make_handler(df: pd.DataFrame) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if self.path != "/tool":
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n))
                result, rows = run_query(df, req["family"], req.get("params", {}))
                body = json.dumps({"result": result, "matching_rows": rows}).encode()
                self.send_response(200)
            except (KeyError, ValueError, TypeError) as e:
                body = json.dumps({"error": str(e)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            log.debug(fmt, *args)

    return Handler


class SiteServer:
    def __init__(self, table: pd.DataFrame | str | Path, host: str = "127.0.0.1", port: int = 0,
                 certfile: str | None = None, keyfile: str | None = None) -> None:
        df = table if isinstance(table, pd.DataFrame) else pd.read_csv(table)
        self.httpd = ThreadingHTTPServer((host, port), make_handler(df))
        if certfile:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile, keyfile)
            self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> SiteServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
