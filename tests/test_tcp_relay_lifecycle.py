import importlib.util
import socket
import threading
from pathlib import Path


def test_closed_tls_client_releases_relay_even_if_remote_socket_is_retained():
    spec = importlib.util.spec_from_file_location('relay', Path(__file__).parents[1] / 'scripts/tcp_evidence_relay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stop = threading.Event()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        client, downstream = socket.socketpair()
        thread = threading.Thread(target=module.relay, args=(downstream, listener.getsockname(), stop), daemon=True)
        thread.start()
        remote, _ = listener.accept()
        try:
            client.sendall(b'last completed TLS frame')
            remote.settimeout(2)
            assert remote.recv(1024) == b'last completed TLS frame'
            remote.sendall(b'completed')
            client.settimeout(2)
            assert client.recv(1024) == b'completed'
            client.close()
            thread.join(timeout=2)
            assert not thread.is_alive(), 'ended links cannot consume all relay connection slots'
        finally:
            stop.set()
            client.close()
            remote.close()
            thread.join(timeout=2)


def test_relay_preserves_bytes_across_temporary_backpressure_and_partial_sends(monkeypatch):
    spec = importlib.util.spec_from_file_location('relay', Path(__file__).parents[1] / 'scripts/tcp_evidence_relay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b'burst' * 100

    class Upstream:
        def __init__(self):
            self.data = bytearray()
            self.first = True
            self.done = threading.Event()
            self.replied = False

        def setsockopt(self, *args):
            pass

        def settimeout(self, *args):
            pass

        def send(self, data):
            if self.first:
                self.first = False
                raise TimeoutError('temporary backpressure')
            count = min(7, len(data))
            self.data.extend(data[:count])
            if len(self.data) == len(payload):
                self.done.set()
            return count

        def recv(self, size):
            assert self.done.wait(2)
            if not self.replied:
                self.replied = True
                return b'ack'
            return b''

        def shutdown(self, *args):
            pass

        def close(self):
            pass

    upstream = Upstream()
    monkeypatch.setattr(module.socket, 'create_connection', lambda *args, **kwargs: upstream)
    client, downstream = socket.socketpair()
    stop = threading.Event()
    thread = threading.Thread(target=module.relay, args=(downstream, ('ignored', 0), stop), daemon=True)
    thread.start()
    try:
        client.sendall(payload)
        client.settimeout(3)
        assert client.recv(10) == b'ack'
        thread.join(timeout=3)
        assert not thread.is_alive() and upstream.data == payload
    finally:
        stop.set()
        client.close()
