"""Finite Mac-only regression, real TRAIN relay/replica check and restore drill."""
import argparse
import hashlib
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

from greenshaping.utils.evidence import atomic_json
from greenshaping.utils.evidence_backup import snapshot, verify
from greenshaping.utils.evidence_replica import restore_sessions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    app = Path(__file__).resolve().parents[1]
    gates = {}
    stages = []
    def run(name, argv, timeout):
        atomic_json(a.out / 'status.json', {'stage': name, 'cloud_used': False, 'test_acquired': False})
        start = time.time()
        with (a.out / (name + '.log')).open('wb') as log:
            result = subprocess.run(argv, cwd=app, stdout=log, stderr=log, timeout=timeout)
        stages.append({'stage': name, 'returncode': result.returncode, 'elapsed_s': time.time() - start})
        atomic_json(a.out / 'commands.json', stages)
        if result.returncode:
            raise RuntimeError(name + ' failed; retained logs')
        gates[name] = True

    server = None
    try:
        run('full_regression', [sys.executable, '-m', 'pytest', '-q'], 300)
        stability = ("import subprocess,sys\nfor i in range(5):\n"
                     " p=subprocess.run([sys.executable,'-m','pytest','-q',"
                     "'tests/test_protocol_v2.py::test_partial_truncation_does_not_corrupt_next_session'])\n"
                     " if p.returncode:sys.exit(p.returncode)\n")
        run('short_horizon_stability', [sys.executable, '-c', stability], 90)
        cert, key = a.out / 'replica.cert', a.out / 'replica.key'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2',
            '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
            '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
        key.chmod(0o600)
        token = a.out / 'replica.token'
        token.write_bytes(secrets.token_bytes(32))
        token.chmod(0o600)
        with (a.out / 'receiver.log').open('wb') as log:
            server = subprocess.Popen([sys.executable, str(app / 'scripts/evidence_replica_server.py'),
                '--host', '127.0.0.1', '--port', '0', '--root', str(a.out / 'replica_objects'),
                '--cert', str(cert), '--key', str(key), '--token', str(token), '--ready', str(a.out / 'receiver_ready.json'),
                '--max-seconds', '1200'], stdout=log, stderr=log)
        deadline = time.monotonic() + 15
        while not (a.out / 'receiver_ready.json').exists():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('replica receiver not ready')
            time.sleep(.1)
        port = json.loads((a.out / 'receiver_ready.json').read_text())['address'][1]
        run('higgs_two_relays_replication', [sys.executable, str(app / 'scripts/check_higgs_endpoint_roles.py'),
            '--store', str(a.data / 'store'), '--banks', str(a.data / 'banks'), '--out', str(a.out / 'live_check'),
            '--N', '20', '--relay-hops', '2', '--replica-endpoint', f'https://127.0.0.1:{port}',
            '--replica-cert', str(cert), '--replica-token', str(token)], 1000)
        summary = json.loads((a.out / 'live_check/summary.json').read_text())
        if not summary['all_passed'] or summary['replica_acknowledged_sessions'] != 60:
            raise RuntimeError('incomplete live session replication')
        restored = restore_sessions(a.out / 'replica_objects', a.out / 'restored_without_primary')
        if restored['committed'] != 60 or restored['passed'] != 60 or restored['incomplete']:
            raise RuntimeError('replica restore failed')
        atomic_json(a.out / 'restore_audit.json', restored)
        gates['restore_without_primary'] = True
        destination = a.out.parent / 'verified_snapshots'
        name = a.out.name
        result = snapshot({'live_check': a.out / 'live_check', 'restore_audit': a.out / 'restore_audit.json'}, destination, name)
        recovered = verify(destination, name, a.out / 'snapshot_restore')
        atomic_json(a.out / 'backup_audit.json', {'snapshot': result, 'restore': recovered})
        gates['snapshot_and_restore'] = result['complete'] and recovered['restore_tested']
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        source = {str(f.relative_to(app)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for folder in ['src', 'scripts'] for f in (app / folder).rglob('*.py')}
        atomic_json(a.out / 'source_sha256.json', source)
        atomic_json(a.out / 'readiness.json', {'software_resilience_verified': len(gates) == 5 and all(gates.values()),
            'gates': gates, 'cloud_used': False, 'test_acquired': False,
            'scope': 'Local crash/restore tests and TRAIN two-relay/replica integration; same host, no physical NetShaper capture',
            'ready_for_main_paper_run': False, 'second_host_backup_required': True})


if __name__ == '__main__':
    main()
