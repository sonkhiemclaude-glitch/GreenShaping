import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / 'scripts/bounded_ssh_gate.py'


def test_run_key_expires_and_stops_its_owned_command():
    env = dict(os.environ, SSH_ORIGINAL_COMMAND='python3 -')
    p = subprocess.run([sys.executable, str(SCRIPT), '--deadline-epoch', str(time.time() + .5)],
                       input='import time;time.sleep(20)', text=True, capture_output=True, env=env, timeout=5)
    assert p.returncode == 124
    p = subprocess.run([sys.executable, str(SCRIPT), '--deadline-epoch', str(time.time() - 1)],
                       input="raise Exception('must not execute')", text=True, capture_output=True, env=env, timeout=5)
    assert p.returncode == 124 and not p.stderr


def test_bounded_key_rejects_arbitrary_shell_and_allows_python():
    env = dict(os.environ, SSH_ORIGINAL_COMMAND='rm -rf example')
    p = subprocess.run([sys.executable, str(SCRIPT), '--deadline-epoch', str(time.time() + 5)],
                       text=True, capture_output=True, env=env, timeout=5)
    assert p.returncode != 0
    env['SSH_ORIGINAL_COMMAND'] = 'python3 -'
    p = subprocess.run([sys.executable, str(SCRIPT), '--deadline-epoch', str(time.time() + 5)],
                       input="print('verified')", text=True, capture_output=True, env=env, timeout=5)
    assert p.returncode == 0 and p.stdout.strip() == 'verified'


def test_sftp_subsystem_accepts_sshd_trailing_whitespace():
    import importlib.util
    spec = importlib.util.spec_from_file_location('gate', SCRIPT)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert gate.command('/usr/lib/openssh/sftp-server ') == ['/usr/lib/openssh/sftp-server']
