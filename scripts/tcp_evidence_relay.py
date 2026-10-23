"""Bounded opaque TCP relay for a common four-role observer topology.

No TLS termination, parsing or application padding. Relay overhead must be
reported; this does not emulate NetShaper's QUIC transport.
"""
import argparse
import socket
import threading
import time
from pathlib import Path

from greenshaping.utils.evidence import atomic_json


def relay(downstream, target, stop):
    upstream = None
    try:
        upstream = socket.create_connection(target, timeout=5)
        upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if downstream.family != socket.AF_UNIX:
            downstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        downstream.settimeout(.5)
        upstream.settimeout(.5)
        closed = threading.Event()

        def pump(source, destination):
            try:
                while not stop.is_set() and not closed.is_set():
                    try:
                        chunk = source.recv(65536)
                    except TimeoutError:
                        continue
                    if not chunk:
                        break
                    # Keep backpressure without imposing a 500ms deadline on
                    # a whole burst. Socket timeouts only allow stop checks;
                    # the enclosing relay process has a fixed run deadline.
                    pending = memoryview(chunk)
                    while pending and not stop.is_set():
                        try:
                            count = destination.send(pending)
                            if not count:
                                raise ConnectionError('relay destination closed')
                            pending = pending[count:]
                        except TimeoutError:
                            continue
            except OSError:
                pass
            finally:
                # Persistent TLS links close only after the session completion
                # barrier. Do not retain a relay forever when the TLS server
                # keeps its already-ended socket object until server shutdown.
                closed.set()
                try:
                    destination.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        thread = threading.Thread(target=pump, args=(downstream, upstream), daemon=True)
        thread.start()
        pump(upstream, downstream)
        thread.join(timeout=2)
    except OSError:
        pass
    finally:
        downstream.close()
        if upstream:
            upstream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--target-host', required=True)
    p.add_argument('--target-port', type=int, required=True)
    p.add_argument('--ready', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    a = p.parse_args()
    if not 1 <= a.max_seconds <= 5400:
        p.error('bounded lifetime required')
    stop = threading.Event()
    threads = []
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((a.host, a.port))
        listener.listen(64)
        listener.settimeout(.5)
        atomic_json(a.ready, {'address': listener.getsockname(), 'opaque_tcp_relay': True})
        deadline = time.monotonic() + a.max_seconds
        try:
            while time.monotonic() < deadline:
                try:
                    peer, _ = listener.accept()
                except TimeoutError:
                    continue
                threads = [t for t in threads if t.is_alive()]
                if len(threads) >= 64:
                    peer.close()
                    continue
                t = threading.Thread(target=relay, args=(peer, (a.target_host, a.target_port), stop), daemon=True)
                t.start()
                threads.append(t)
        finally:
            stop.set()
            join_deadline = time.monotonic() + 2
            for t in threads:
                t.join(timeout=max(0., join_deadline - time.monotonic()))


if __name__ == '__main__':
    main()
