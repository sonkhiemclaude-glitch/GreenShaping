"""Credential-safe SSH/SFTP over a verified Tailscale peer; .env stays local.

Use a known_hosts file established during host onboarding. No credentials are
printed or placed in command arguments; this helper never provisions global services.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path


def main() -> int:
    import paramiko
    from dotenv import dotenv_values

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', default=str(Path(__file__).resolve().parents[1] / '.env'))
    p.add_argument('--known-hosts', required=True)
    sub = p.add_subparsers(dest='operation', required=True)
    ex = sub.add_parser('exec')
    ex.add_argument('command')
    ex.add_argument('--timeout', type=float, default=60.)
    for operation in ('put', 'get'):
        cmd = sub.add_parser(operation)
        cmd.add_argument('source')
        cmd.add_argument('target')
        if operation == 'get':
            cmd.add_argument('--sha256', help='expected remote object checksum; otherwise compute it on the remote host')
    a = p.parse_args()
    env = dotenv_values(a.env)
    if any(not env.get(k) for k in ('Server', 'Username', 'Password')):
        raise ValueError('local connection environment requires Server, Username and Password')
    ip = socket.gethostbyname(str(env['Server']))
    peers = json.loads(subprocess.check_output(['tailscale', 'status', '--json']))
    if not any(ip in peer.get('TailscaleIPs', []) for peer in peers.get('Peer', {}).values()):
        raise ValueError('configured endpoint is not a Tailscale peer')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(a.known_hosts)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(ip, port=int(env.get('Port') or 22), username=env['Username'], password=env['Password'],
                       look_for_keys=False, allow_agent=False, timeout=20, auth_timeout=20)
        client.get_transport().set_keepalive(15)
        if a.operation == 'exec':
            stdin, stdout, stderr = client.exec_command(a.command, timeout=a.timeout)
            data = sys.stdin.buffer.read()
            if data:
                stdin.write(data)
                stdin.flush()
            stdin.channel.shutdown_write()
            # Drain both streams to avoid an SSH channel-window deadlock.
            import threading

            def drain(stream: object, output: object) -> None:
                while chunk := stream.read(32768):  # type: ignore[attr-defined]
                    output.write(chunk)  # type: ignore[attr-defined]
                    output.flush()  # type: ignore[attr-defined]

            thread = threading.Thread(target=drain, args=(stderr, sys.stderr.buffer))
            thread.start()
            drain(stdout, sys.stdout.buffer)
            thread.join()
            return int(stdout.channel.recv_exit_status())
        with client.open_sftp() as sftp:
            if a.operation == 'put':
                sftp.put(a.source, a.target)
            else:
                from greenshaping.utils.transfer import download_verified

                expected = a.sha256
                if not expected:
                    import shlex
                    source = ("import hashlib;from pathlib import Path;h=hashlib.sha256();"
                              "f=Path(" + repr(a.source) + ").open('rb');"
                              "[h.update(b) for b in iter(lambda:f.read(1048576),b'')];print(h.hexdigest())")
                    _, stdout, stderr = client.exec_command('python3 -c ' + shlex.quote(source), timeout=180)
                    expected = stdout.read().decode().strip()
                    if stdout.channel.recv_exit_status():
                        raise RuntimeError('remote checksum failed: ' + stderr.read().decode()[-1000:])
                if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                    raise ValueError('invalid expected checksum')
                download_verified(sftp, a.source, Path(a.target), expected)
        return 0
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
