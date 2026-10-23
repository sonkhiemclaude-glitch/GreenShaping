"""TLS/HMAC authenticated immutable evidence objects on a second host.

The run-specific token is independent of SSH credentials and never logged.
This is a private testbed service, bounded by disk and time at deployment.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import ssl
import tempfile
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from greenshaping.utils.evidence import atomic_json, sync_directory

MAX_OBJECT_BYTES = 4 * 1024 * 1024


def signature(token: bytes, checksum: str) -> str:
    return hmac.new(token, checksum.encode(), hashlib.sha256).hexdigest()


def make_server(host, port, root: Path, token: bytes, cert: Path, key: Path):
    root.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            checksum = self.path.removeprefix('/objects/')
            if (not self.path.startswith('/objects/') or not re.fullmatch('[0-9a-f]{64}', checksum)
                    or not hmac.compare_digest(self.headers.get('X-Evidence-Signature', ''), signature(token, checksum))):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length', '-1'))
                if not 0 <= length <= MAX_OBJECT_BYTES:
                    self.send_error(413)
                    return
                self.connection.settimeout(5)
                data = self.rfile.read(length)
                if len(data) != length or hashlib.sha256(data).hexdigest() != checksum:
                    self.send_error(400)
                    return
                if shutil.disk_usage(root).free < length + 512 * 1024**2:
                    self.send_error(507)
                    return
                path = root / checksum
                fd, name = tempfile.mkstemp(prefix='.upload-', dir=root)
                temp = Path(name)
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.link(temp, path)
                    except FileExistsError:
                        if path.read_bytes() != data:
                            self.send_error(409)
                            return
                    sync_directory(root)
                finally:
                    temp.unlink(missing_ok=True)
                response = json.dumps({'stored': True, 'sha256': checksum}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except (OSError, ValueError):
                self.send_error(500)

    class BoundedServer(HTTPServer):
        def get_request(self):
            connection, address = self.socket.accept()
            connection.settimeout(5)
            try:
                connection.do_handshake()
            except OSError:
                connection.close()
                raise
            return connection, address

    server = BoundedServer((host, port), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    server.timeout = 1
    return server


class EvidenceReplica:
    def __init__(self, endpoint: str, cafile: Path, token_file: Path):
        if not endpoint.startswith('https://'):
            raise ValueError('replica requires verified TLS')
        self.endpoint = endpoint.rstrip('/')
        self.context = ssl.create_default_context(cafile=str(cafile))
        self.context.minimum_version = ssl.TLSVersion.TLSv1_3
        self.token = token_file.read_bytes()
        if len(self.token) < 32 or token_file.stat().st_mode & 0o077:
            raise ValueError('replica token requires >=32 bytes and private permissions')

    def put(self, data: bytes) -> str:
        if len(data) > MAX_OBJECT_BYTES:
            raise ValueError('replica object too large; split before upload')
        checksum = hashlib.sha256(data).hexdigest()
        req = urllib.request.Request(self.endpoint + '/objects/' + checksum, data=data,
            headers={'X-Evidence-Signature': signature(self.token, checksum)}, method='POST')
        with urllib.request.urlopen(req, context=self.context, timeout=8) as response:
            result = json.loads(response.read(1024))
        if result != {'stored': True, 'sha256': checksum}:
            raise ValueError('replica did not acknowledge the expected checksum')
        return checksum

    def session(self, ledger: Path, session: str) -> dict:
        # An error stops further acquisition; the primary commit remains intact.
        files = {}
        for relative in ['protocol.json', f'started/{session}.json', f'committed/{session}.json']:
            path = ledger / relative
            files[relative] = self.put(path.read_bytes())
        manifest = {'session': session, 'files': files}
        checksum = self.put(json.dumps(manifest, sort_keys=True).encode())
        receipts = ledger / 'replication_receipts'
        receipts.mkdir(exist_ok=True)
        receipt = {'manifest_sha256': checksum, 'acknowledged': True}
        atomic_json(receipts / (session + '.json'), receipt, exclusive=True)
        return receipt


def restore_sessions(objects: Path, destination: Path) -> dict:
    """Recover even if the primary disk and its replica receipts are lost."""
    from greenshaping.utils.evidence import SessionLedger

    recovered = 0
    for path in sorted(objects.iterdir()):
        if not re.fullmatch('[0-9a-f]{64}', path.name):
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != path.name:
            raise ValueError('corrupt replica object')
        try:
            manifest = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(manifest, dict) or set(manifest) != {'session', 'files'}:
            continue
        session = manifest['session']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', session):
            raise ValueError('unsafe replica session')
        expected = {'protocol.json', f'started/{session}.json', f'committed/{session}.json'}
        if set(manifest['files']) != expected:
            raise ValueError('incomplete replica manifest')
        values = {}
        for relative, checksum in manifest['files'].items():
            if not re.fullmatch('[0-9a-f]{64}', checksum):
                raise ValueError('unsafe replica checksum')
            content = (objects / checksum).read_bytes()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError('corrupt replica object')
            values[relative] = json.loads(content)
        SessionLedger(destination, values['protocol.json']['protocol'])
        for relative in sorted(expected):
            target = destination / relative
            if target.exists():
                if json.loads(target.read_text()) != values[relative]:
                    raise ValueError('conflicting restored session')
            else:
                atomic_json(target, values[relative], exclusive=True)
        recovered += 1
    if not recovered:
        raise ValueError('no fully acknowledged session manifests found')
    protocol = json.loads((destination / 'protocol.json').read_text())['protocol']
    return SessionLedger(destination, protocol).checkpoint()
