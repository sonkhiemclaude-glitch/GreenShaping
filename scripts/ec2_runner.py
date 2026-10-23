"""SSH/SCP to the explicitly provisioned EC2 testbed; key remains on this Mac."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = json.loads(Path(os.environ.get('GREENSHAPING_EC2_INVENTORY',
                          str(ROOT / 'configs/ec2_netshaper_20261020.json'))).read_text())
EVIDENCE = Path(os.environ.get('GREENSHAPING_EC2_EVIDENCE', str(ROOT / 'outputs/ec2_netshaper_20261020')))


def require_available(nodes: list[str]) -> None:
    """Reject experiment launches on reserved hosts; read-only SSH remains usable."""
    path = Path(os.environ.get('GREENSHAPING_EC2_RESERVATIONS', str(ROOT / 'configs/ec2_host_reservations.json')))
    reservations = json.loads(path.read_text()) if path.exists() else {}
    blocked = [f'{node}: {reservations[node]["reason"]}' for node in nodes
               if reservations.get(node, {}).get('reserved')]
    blocked += [f'{node}: bounded evaluation window expired' for node in nodes
                if reservations.get(node, {}).get('available_until_epoch') is not None
                and time.time() >= reservations[node]['available_until_epoch']]
    if blocked:
        raise RuntimeError('Experiment launch blocked by host reservation: ' + '; '.join(blocked))


def ssh_args(node: str) -> list[str]:
    identity = Path(os.environ.get('GREENSHAPING_EC2_IDENTITY', str(ROOT / 'SonJ.pem')))
    known_hosts = Path(os.environ.get('GREENSHAPING_EC2_KNOWN_HOSTS', str(EVIDENCE / 'known_hosts')))
    return ['-i', str(identity), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            '-o', 'StrictHostKeyChecking=yes', '-o', f'UserKnownHostsFile="{known_hosts}"']


def run(node: str, command: str, data: str | None = None, timeout: float = 120.) -> subprocess.CompletedProcess:
    target = f'ubuntu@{INVENTORY["nodes"][node]["public_ip"]}'
    return subprocess.run(['ssh', *ssh_args(node), target, command], input=data, capture_output=True,
                          text=True, timeout=timeout)


def python(node: str, source: str, timeout: float = 120.) -> str:
    result = run(node, 'python3 -', data=source, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{node} failed ({result.returncode}): {result.stderr[-4000:]} {result.stdout[-4000:]}')
    return result.stdout


def copy(node: str, source: str | Path, target: str, get: bool = False) -> None:
    # Explicit SFTP avoids legacy scp -f/-t commands on older macOS clients.
    def quoted(value):
        value = str(value)
        if '\n' in value or '\r' in value:
            raise ValueError('newline in transfer path')
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    operation = 'get' if get else 'put'
    paths = [target, source] if get else [source, target]
    batch = operation + ' ' + ' '.join(quoted(p) for p in paths) + '\n'
    target_host = f'ubuntu@{INVENTORY["nodes"][node]["public_ip"]}'
    subprocess.run(['sftp', '-b', '-', *ssh_args(node), target_host], input=batch,
                   text=True, check=True, timeout=900)



def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('node', choices=list(INVENTORY['nodes']))
    sub = p.add_subparsers(dest='operation', required=True)
    ex = sub.add_parser('exec')
    ex.add_argument('command')
    ex.add_argument('--timeout', type=float, default=120.)
    for name in ('put', 'get'):
        cmd = sub.add_parser(name)
        cmd.add_argument('local')
        cmd.add_argument('remote')
    a = p.parse_args()
    if a.operation == 'exec':
        result = run(a.node, a.command, sys.stdin.read(), a.timeout)
        print(result.stdout, end='')
        print(result.stderr, end='', file=sys.stderr)
        return int(result.returncode)
    copy(a.node, a.local, a.remote, get=a.operation == 'get')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
