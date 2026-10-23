"""Install testbed tools and pin upstream middlebox images, preserving setup logs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--role', choices=['client', 'middlebox_client', 'middlebox_server', 'server'], required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
state: dict = {'pid': os.getpid(), 'state': 'running', 'role': a.role}


def save(phase: str) -> None:
    state['phase'] = phase
    (root / 'setup_status.json').write_text(json.dumps(state, indent=2) + '\n')


try:
    with (root / 'setup.log').open('wb') as log:
        def run(cmd: list[str]) -> None:
            subprocess.run(cmd, stdout=log, stderr=log, check=True, timeout=900)

        save('apt_update')
        run(['sudo', '-n', 'apt-get', '-o', 'DPkg::Lock::Timeout=120', 'update'])
        save('apt_install')
        run(['sudo', '-n', 'env', 'DEBIAN_FRONTEND=noninteractive', 'NEEDRESTART_MODE=a',
             'apt-get', '-o', 'DPkg::Lock::Timeout=120', 'install', '-y',
             'docker.io', 'tcpdump', 'iperf3', 'git', 'jq', 'curl', 'python3-venv'])
        run(['sudo', '-n', 'systemctl', 'start', 'docker'])
        images = []
        if a.role in ('middlebox_client', 'middlebox_server'):
            peer = 'peer1' if a.role == 'middlebox_client' else 'peer2'
            for mode in ('no-shaping', 'shaping'):
                tag = f'amirsabzi/netshaper:{peer}-{mode}'
                save(f'pull_{peer}_{mode}')
                run(['sudo', '-n', 'docker', 'pull', tag])
                info = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'image', 'inspect', tag], text=True))[0]
                images.append({'tag': tag, 'id': info['Id'], 'repo_digests': info.get('RepoDigests', []),
                               'architecture': info['Architecture'], 'created': info['Created']})
        (root / 'images.json').write_text(json.dumps(images, indent=2) + '\n')
    state['state'] = 'complete'
    save('complete')
except Exception as e:
    state.update(state='failed', error=str(e))
    save(state.get('phase', 'unknown'))
    raise
