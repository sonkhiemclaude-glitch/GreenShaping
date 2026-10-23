"""Private bounded evidence receiver; use a run-specific TLS cert and token."""
import argparse
import time
from pathlib import Path

from greenshaping.utils.evidence import atomic_json
from greenshaping.utils.evidence_replica import make_server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cert', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--token', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--ready', type=Path)
    a = p.parse_args()
    if not 1 <= a.max_seconds <= 5400 or a.token.stat().st_mode & 0o077 or len(a.token.read_bytes()) < 32:
        p.error('invalid receiver deadline or token permissions/length')
    server = make_server(a.host, a.port, a.root, a.token.read_bytes(), a.cert, a.key)
    if a.ready:
        atomic_json(a.ready, {'address': server.server_address, 'ready': True})
    deadline = time.monotonic() + a.max_seconds
    try:
        while time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
