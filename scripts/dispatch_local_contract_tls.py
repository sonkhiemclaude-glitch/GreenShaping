"""Deploy a small, isolated stdlib experiment over the verified Tailscale peer.

Existing SSH credentials stay in memory and travel only inside SSH stdin.
No model, environment or original project is copied or changed.
"""
import argparse
import io
import json
import shlex
import re
import socket
import subprocess
import tarfile
from pathlib import Path

import paramiko
from dotenv import dotenv_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='run05')
    parser.add_argument('--seed-offset', type=int, default=20000)
    parser.add_argument('--policies', default='envelope,periodic_6,periodic_10')
    a = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_]{1,40}', a.run_id):
        parser.error('invalid isolated run identifier')
    root = Path(__file__).resolve().parents[1]
    env = dotenv_values(root / '.env')
    ip = socket.gethostbyname(env['Server'])
    peers = json.loads(subprocess.check_output(['tailscale', 'status', '--json']))
    if not any(ip in p.get('TailscaleIPs', []) for p in peers.get('Peer', {}).values()):
        raise ValueError('configured server is not a Tailscale peer')
    files = ['scripts/benchmark_local_contract_tls.py', 'src/greenshaping/__init__.py',
             'src/greenshaping/testbed/__init__.py', 'src/greenshaping/controller/__init__.py',
             'src/greenshaping/controller/workflow_contract.py', 'src/greenshaping/testbed/workflow_cover.py',
             'src/greenshaping/testbed/paced_workflow_tls.py', 'src/greenshaping/utils/__init__.py',
             'src/greenshaping/utils/evidence.py', 'src/greenshaping/data/__init__.py',
             'src/greenshaping/data/wire_capture.py']
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in files:
            archive.add(root / name, arcname=name)
    payload = buf.getvalue()
    stage = '/Users/sonha/Projects/GreenShaping_LocalContract_20261023_' + a.run_id
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(str(root / 'outputs/remote_20261020/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    remote_args = ['--capture-sudo', '--reuse-contract',
                   '/Users/sonha/Projects/GreenShaping_LocalContract_20261023_run03/results/frozen_plan.json',
                   '--seed-offset', str(a.seed_offset), '--policies', a.policies]
    try:
        client.connect(ip, port=int(env.get('Port') or 22), username=env['Username'], password=env['Password'],
                       look_for_keys=False, allow_agent=False, timeout=20, auth_timeout=20)
        with client.open_sftp() as sftp:
            sftp.mkdir(stage)
            with sftp.open(stage + '/source.tar.gz', 'wb') as stream:
                stream.write(payload)
        code = """
import os,sys,subprocess,tarfile,json
from pathlib import Path
stage=Path(STAGE)
with tarfile.open(stage/'source.tar.gz') as archive:
    for m in archive.getmembers():
        if m.name.startswith('/') or '..' in Path(m.name).parts:raise ValueError('unsafe member')
    archive.extractall(stage)
password=sys.stdin.readline().rstrip('\\n')
probe='import ssl; c=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); c.minimum_version=ssl.TLSVersion.TLSv1_3'
runtime=None
for candidate in ['/opt/homebrew/bin/python3.12','/Users/sonha/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3','/opt/homebrew/bin/python3','/usr/bin/python3']:
    if Path(candidate).exists() and subprocess.run([candidate,'-c',probe],capture_output=True).returncode==0:
        runtime=candidate;break
if runtime is None:raise RuntimeError('No existing TLS 1.3 Python runtime; no acquisition started')
childenv=os.environ.copy();childenv['PYTHONPATH']=str(stage/'src')
log=(stage/'run.log').open('wb')
child=subprocess.Popen(['/usr/bin/caffeinate','-i',runtime,str(stage/'scripts/benchmark_local_contract_tls.py'),'--out',str(stage/'results')]+CLIARGS,cwd=stage,env=childenv,stdin=subprocess.PIPE,stdout=log,stderr=log,start_new_session=True)
child.stdin.write((password+'\\n').encode());child.stdin.close();log.close();password=None
print(json.dumps({'pid':child.pid,'stage':str(stage),'runtime':runtime,'source_bytes':(stage/'source.tar.gz').stat().st_size}))
""".replace('STAGE', repr(stage)).replace('CLIARGS', repr(remote_args))
        stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -c ' + shlex.quote(code), timeout=30)
        stdin.write(env['Password'] + '\n')
        stdin.flush()
        stdin.channel.shutdown_write()
        result = stdout.read().decode()
        error = stderr.read().decode()
        if stdout.channel.recv_exit_status():
            raise RuntimeError('isolated dispatch failed: ' + error[-1000:])
        out = root / 'outputs/local_contract_tls_20261023'
        out.mkdir(exist_ok=True)
        (out / ('dispatch_' + a.run_id + '.json')).write_text(result)
        print(result)
    finally:
        client.close()


if __name__ == '__main__':
    main()
