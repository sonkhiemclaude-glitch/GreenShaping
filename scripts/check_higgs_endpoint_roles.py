"""Exercise the portable server/client in separate local processes; TRAIN only."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--banks', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--N', type=int, choices=[5, 10, 20], default=20)
    p.add_argument('--relay-hops', type=int, choices=[0, 2], default=0)
    p.add_argument('--replica-endpoint')
    p.add_argument('--replica-cert', type=Path)
    p.add_argument('--replica-token', type=Path)
    p.add_argument('--modes', nargs='+', default=['none', 'pad', 'rounds', 'dp', 'combo'])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cert, key = a.out / 'server.cert', a.out / 'server.key'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2',
                    '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
    key.chmod(0o600)
    role = Path(__file__).with_name('higgs_tls_endpoint.py')
    bank = a.banks / f'n{a.N}/train.jsonl'
    reference = a.out / 'reference.json'
    subprocess.run([sys.executable, str(role), 'reference', '--N', str(a.N), '--store', str(a.store),
                    '--bank', str(bank), '--out', str(reference)], check=True, timeout=300)
    with (a.out / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, str(role), 'server', '--N', str(a.N), '--store', str(a.store),
                                   '--host', '127.0.0.1', '--port', '0', '--cert', str(cert), '--key', str(key),
                                   '--out', str(a.out / 'server'), '--max-seconds', '1200'], stdout=log, stderr=log)
    relays = []
    try:
        ready = a.out / 'server/ready.json'
        deadline = time.monotonic() + 180
        while not ready.exists():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('portable server not ready; inspect server.log')
            time.sleep(.1)
        address = json.loads(ready.read_text())['address']
        for hop in range(a.relay_hops):
            ready_relay = a.out / f'relay_{hop}.json'
            with (a.out / f'relay_{hop}.log').open('w') as log:
                relay = subprocess.Popen([sys.executable, str(role.with_name('tcp_evidence_relay.py')),
                    '--host', '127.0.0.1', '--port', '0', '--target-host', address[0], '--target-port', str(address[1]),
                    '--ready', str(ready_relay), '--max-seconds', '1200'], stdout=log, stderr=log)
            relays.append(relay)
            while not ready_relay.exists():
                if relay.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('relay not ready')
                time.sleep(.1)
            address = json.loads(ready_relay.read_text())['address']
        endpoints = a.out / 'endpoints.json'
        endpoints.write_text(json.dumps([address] * a.N))
        replica = (['--replica-endpoint', a.replica_endpoint, '--replica-cert', str(a.replica_cert),
                    '--replica-token', str(a.replica_token)] if a.replica_endpoint else [])
        subprocess.run([sys.executable, str(role), 'client', '--N', str(a.N), '--bank', str(bank),
                        '--cert', str(cert), '--endpoints', str(endpoints), '--reference', str(reference),
                        '--out', str(a.out / 'client'), '--modes', *a.modes, '--repetitions', '1',
                        '--max-seconds', '900', *replica], check=True, timeout=960)
    finally:
        for relay in relays:
            relay.terminate()
            try:
                relay.wait(timeout=5)
            except subprocess.TimeoutExpired:
                relay.kill()
                relay.wait()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    summary = json.loads((a.out / 'client/summary.json').read_text())
    summary.update(server_cleanup_returncode=server.returncode, roles_in_separate_processes=True, relay_hops=a.relay_hops)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
