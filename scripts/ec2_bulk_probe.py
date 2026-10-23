"""Fixed-length bulk probe: four streams, CRC32 and receiver-clock goodput, no shared control stream."""

import argparse
import concurrent.futures
import json
import socket
import threading
import time
import zlib

BLOCK = b"\0" * (1024 * 1024)
LENGTH = 1024 * 1024 * 1024
EXPECTED = 0
for _ in range(LENGTH // len(BLOCK)):
    EXPECTED = zlib.crc32(BLOCK, EXPECTED)


def line(sock):
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise EOFError("Connection closed before control record")
        data.extend(chunk)
        if len(data) > 8192:
            raise ValueError("Oversized record")
    return json.loads(data)


def receive(sock):
    count = 0
    crc = 0
    start = None
    while count < LENGTH:
        data = sock.recv(min(len(BLOCK), LENGTH - count))
        if not data:
            raise EOFError(f"Only {count} of {LENGTH} bytes received")
        if start is None:
            start = time.time_ns()
        count += len(data)
        crc = zlib.crc32(data, crc)
    end = time.time_ns()
    if crc != EXPECTED:
        raise ValueError(f"CRC32 mismatch {crc} != {EXPECTED}")
    return dict(bytes=count, crc32=crc, crc_valid=True, start_ns=start, end_ns=end)


def send(sock, rate):
    start = time.monotonic()
    count = 0
    while count < LENGTH:
        sock.sendall(BLOCK)
        count += len(BLOCK)
        delay = start + count * 8 / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    return dict(bytes=count, accepted_bps=count * 8 / (time.monotonic() - start))


def server(host, port):
    def handle(conn):
        with conn:
            conn.settimeout(65)
            try:
                request = line(conn)
                assert request["length"] == LENGTH and request["mode"] in ("GET", "PUT")
                conn.sendall(b"R")
                if request["mode"] == "PUT":
                    result = receive(conn)
                    conn.sendall(json.dumps(result).encode() + b"\n")
                else:
                    assert conn.recv(1) == b"G"
                    sent = send(conn, request["rate"])
                    result = line(conn)
                    result["sender"] = sent
                print(json.dumps(dict(trial=request["trial"], result=result)), flush=True)
            except Exception as exc:
                print(json.dumps(dict(error=repr(exc))), flush=True)

    deadline = time.monotonic() + 300
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(32)
        listener.settimeout(1)
        while time.monotonic() < deadline:
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            threading.Thread(target=handle, args=(conn,), daemon=True).start()


def client(host, port, mode, trial):
    barrier = threading.Barrier(4, timeout=15)

    def stream(index):
        with socket.create_connection((host, port), timeout=8) as conn:
            conn.settimeout(65)
            request = dict(length=LENGTH, mode=mode, rate=2_500_000_000, trial=trial, stream=index)
            conn.sendall(json.dumps(request).encode() + b"\n")
            assert conn.recv(1) == b"R"
            barrier.wait()
            if mode == "PUT":
                sent = send(conn, request["rate"])
                result = line(conn)
                result["sender"] = sent
            else:
                conn.sendall(b"G")
                result = receive(conn)
                conn.sendall(json.dumps(result).encode() + b"\n")
            return result

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results = list(pool.map(stream, range(4)))
    duration = (max(x["end_ns"] for x in results) - min(x["start_ns"] for x in results)) / 1e9
    print(
        json.dumps(
            dict(
                trial=trial,
                mode=mode,
                total_bytes=sum(x["bytes"] for x in results),
                receiver_seconds=duration,
                receiver_bps=sum(x["bytes"] for x in results) * 8 / duration,
                crc_valid=all(x["crc_valid"] for x in results),
                streams=results,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--mode", choices=["GET", "PUT"], default="PUT")
    parser.add_argument("--trial", default="bulk")
    args = parser.parse_args()
    if args.server:
        server(args.host, args.port)
    else:
        client(args.host, args.port, args.mode, args.trial)
