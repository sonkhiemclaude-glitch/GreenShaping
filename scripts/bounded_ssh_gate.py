"""Forced command for a run-specific Mac Studio key, with an absolute expiry.

Trusted evaluation code can execute Python or transfer evidence using SFTP;
this is not a sandbox for untrusted Python. SSH restrict disables forwarding.
Never copies or replaces the user's SonJ.pem.
"""
import argparse
import os
import signal
import subprocess
import time


def command(original):
    original = original.strip()
    if original == 'python3 -':
        return ['python3', '-']
    if original in ('internal-sftp', '/usr/lib/openssh/sftp-server'):
        return ['/usr/lib/openssh/sftp-server']
    raise ValueError('only evaluation Python/SFTP commands are permitted')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deadline-epoch', type=float, required=True)
    a = p.parse_args()
    remaining = a.deadline_epoch - time.time()
    if remaining <= 0:
        raise SystemExit(124)
    argv = command(os.environ.get('SSH_ORIGINAL_COMMAND', ''))
    child = subprocess.Popen(argv, start_new_session=True)
    try:
        raise SystemExit(child.wait(timeout=remaining))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise SystemExit(124) from None


if __name__ == '__main__':
    main()
