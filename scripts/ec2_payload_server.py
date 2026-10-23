"""Bounded synthetic HTTP endpoint for four-host NetShaper functional checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SIZES = (1024, 16384, 65536, 262144)


def payload(size: int) -> bytes:
    return hashlib.shake_256(f'netshaper-ec2-functional-{size}'.encode()).digest(size)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bind', required=True)
    p.add_argument('--port', type=int, default=5555)
    p.add_argument('--duration', type=int, default=900)
    args = p.parse_args()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self) -> None:
            try:
                size = int(self.path.split('?')[0].removeprefix('/payload/'))
                if size not in SIZES:
                    raise ValueError('unsupported size')
            except ValueError:
                self.send_error(404)
                return
            data = payload(size)
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Payload-SHA256', hashlib.sha256(data).hexdigest())
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True
            print(json.dumps({'epoch': time.time(), 'source_ip': self.client_address[0],
                              'size': size, 'path': self.path}), flush=True)

        def log_message(self, fmt: str, *values: object) -> None:
            pass

    with ThreadingHTTPServer((args.bind, args.port), Handler) as server:
        timer = threading.Timer(args.duration, server.shutdown)
        timer.daemon = True
        timer.start()
        server.serve_forever()


if __name__ == '__main__':
    main()
