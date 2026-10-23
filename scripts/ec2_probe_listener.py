"""Short-lived private-interface echo endpoint for TCP/UDP security-group checks."""
from __future__ import annotations

import argparse
import socket
import time

p = argparse.ArgumentParser()
p.add_argument('--ip', required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--udp', action='store_true')
a = p.parse_args()
deadline = time.monotonic() + 90
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM if a.udp else socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((a.ip, a.port))
    sock.settimeout(1)
    if not a.udp:
        sock.listen()
    while time.monotonic() < deadline:
        try:
            if a.udp:
                data, address = sock.recvfrom(4096)
                sock.sendto(data, address)
            else:
                conn, _ = sock.accept()
                with conn:
                    conn.settimeout(2)
                    conn.sendall(conn.recv(4096))
        except TimeoutError:
            pass
