import json
import secrets
import subprocess
import threading
import urllib.error

import pytest

from greenshaping.utils.evidence import SessionLedger
from greenshaping.utils.evidence_replica import EvidenceReplica, make_server, restore_sessions


def test_tls_replica_authentication_idempotence_and_loss_of_primary(tmp_path):
    cert, key = tmp_path / 'server.cert', tmp_path / 'server.key'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                    '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(cert)], capture_output=True, check=True)
    token_file = tmp_path / 'replica.token'
    token_file.write_bytes(secrets.token_bytes(32))
    token_file.chmod(0o600)
    objects = tmp_path / 'remote_objects'
    server = make_server('127.0.0.1', 0, objects, token_file.read_bytes(), cert, key)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    ledger = SessionLedger(tmp_path / 'primary', {'phase': 'TRAIN'})
    ledger.start('s0', {'id': 2})
    ledger.commit('s0', {'passed': True, 'latency': .3})
    client = EvidenceReplica(f'https://127.0.0.1:{server.server_port}', cert, token_file)
    try:
        receipt = client.session(ledger.root, 's0')
        manifest = json.loads((objects / receipt['manifest_sha256']).read_text())
        assert manifest['session'] == 's0'
        count = len(list(objects.iterdir()))
        assert client.put(b'payload') == client.put(b'payload')
        assert len(list(objects.iterdir())) == count + 1
        client.token = b'invalid token'
        with pytest.raises(urllib.error.HTTPError) as error:
            client.put(b'rejected')
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    # No reference to the original primary or its receipt is needed for recovery.
    restored = tmp_path / 'restored'
    result = restore_sessions(objects, restored)
    assert result['committed'] == result['passed'] == 1 and not result['incomplete']
    assert json.loads((restored / 'committed/s0.json').read_text())['record']['latency'] == .3
    ledger.start('s1', {})
    ledger.commit('s1', {'passed': True})
    with pytest.raises(urllib.error.URLError):
        client.session(ledger.root, 's1')
    assert ledger.audit()['committed'] == 2
    assert not (ledger.root / 'replication_receipts/s1.json').exists()
