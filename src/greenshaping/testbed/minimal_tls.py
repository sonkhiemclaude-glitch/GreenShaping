"""Persistent minimal TLS RPC baseline without all-site scheduling controls.

Counters are framed plaintext bytes, not IPv4/wire bytes or measured energy.
Capture is still required for network cost. Same tool handler as shaped runs.
"""
import json
import socket
import socketserver
import ssl
import struct
from concurrent.futures import ThreadPoolExecutor

MAX_FRAME = 1024 * 1024


def read_frame(sock):
    def exactly(n):
        out = bytearray()
        while len(out) < n:
            b = sock.recv(n-len(out))
            if not b:raise EOFError('TLS connection closed mid-frame')
            out.extend(b)
        return bytes(out)
    size = struct.unpack('!I', exactly(4))[0]
    if not 0 < size <= MAX_FRAME:raise ValueError('invalid minimal TLS frame size')
    return json.loads(exactly(size)), size + 4


def write_frame(sock, value):
    b = json.dumps(value,separators=(',',':'),allow_nan=False).encode()
    if not 0 < len(b) <= MAX_FRAME:raise ValueError('oversized minimal TLS frame')
    sock.sendall(struct.pack('!I',len(b))+b)
    return len(b)+4


class MinimalServer:
    def __init__(self, cert, key, tool):
        import threading
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(cert,key)
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    self.request.settimeout(5)
                    with context.wrap_socket(self.request,server_side=True) as sock:
                        sock.settimeout(30)
                        while True:
                            request,_=read_frame(sock)
                            result,n=tool(request['family'],request['params'])
                            write_frame(sock,{'result':result,'matching_rows':n})
                except (EOFError,OSError):return
        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address=True
            daemon_threads=True
        self.server=Server(('127.0.0.1',0),Handler)
        self.address=self.server.server_address
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)


class MinimalClient:
    def __init__(self,addresses,cert):
        context=ssl.create_default_context(cafile=cert);context.minimum_version=ssl.TLSVersion.TLSv1_3
        self.sockets=[]
        try:
            for host,port in addresses:
                sock=context.wrap_socket(socket.create_connection((host,port),timeout=10),server_hostname=host)
                sock.settimeout(15);self.sockets.append(sock)
        except BaseException:
            self.close();raise
        self.tx=self.rx=0
    def call_round(self,family,calls):
        if len({c['site'] for c in calls})!=len(calls):raise ValueError('one call per site per round')
        def call(c):
            sock=self.sockets[c['site']]
            tx=write_frame(sock,dict(family=family,params=c['params']))
            result,rx=read_frame(sock)
            return result,tx,rx
        with ThreadPoolExecutor(len(calls)) as pool:rows=list(pool.map(call,calls))
        self.tx+=sum(r[1] for r in rows);self.rx+=sum(r[2] for r in rows)
        return [r[0] for r in rows]
    def close(self):
        for sock in self.sockets:
            try:sock.close()
            except OSError:pass
