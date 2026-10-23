"""Local finite Tailscale status watcher and macOS notification; never starts EC2."""
from __future__ import annotations

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from durable_worker import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--remote-readiness', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    state_path = a.out / 'state.json'
    with (a.out / 'lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state = read(state_path, {})
        if state.get('notification_attempted'):
            return
        source = ('import json; from pathlib import Path; p=Path(' + repr(a.remote_readiness) + '); '
                  'print(p.read_text() if p.exists() else json.dumps({"status":"pending"}))')
        command = '/usr/bin/python3 -c ' + shlex.quote(source)
        helper = a.project / 'scripts/remote_runner.py'
        while True:
            state.update(last_check_epoch=time.time(), remote_readiness=a.remote_readiness, cloud_started=False)
            try:
                result = subprocess.run([sys.executable, str(helper), '--known-hosts',
                    str(a.project / 'outputs/remote_20261020/known_hosts'), 'exec', command, '--timeout', '30'],
                    capture_output=True, text=True, timeout=65)
                if result.returncode:
                    raise RuntimeError(result.stderr[-1000:])
                readiness = json.loads(result.stdout)
                state.pop('connection_error', None)
                if 'ready_for_ec2_calibration' in readiness:
                    write(a.out / 'readiness_snapshot.json', readiness)
                    passed = (readiness['ready_for_ec2_calibration'] is True and bool(readiness.get('gates')) and
                              all(value is True for value in readiness['gates'].values()) and
                              readiness.get('run_root') == str(Path(a.remote_readiness).parent))
                    message = ('Mac checks passed. Review EC2_START_NOTICE, then arrange a bounded T1-T4 calibration.' if passed else
                               'Preparation has findings. Keep EC2 stopped and review the readiness report.')
                    notice = subprocess.run(['/usr/bin/osascript', '-e', 'display notification ' + json.dumps(message) +
                                             ' with title "Green Shaping"'], capture_output=True, text=True, timeout=15)
                    state.update(notification_attempted=True, notification_command_exit=notice.returncode,
                                 delivery_confirmed=False, software_ready=passed, terminal_epoch=time.time(),
                                 notification_stderr=notice.stderr[-1000:])
                    write(state_path, state)
                    return
                state['phase'] = 'waiting_for_mac_result'
            except Exception as exc:
                state['connection_error'] = repr(exc)
            write(state_path, state)
            # Check once even on a later login after the deadline, to surface a
            # result completed while this computer was powered off.
            if a.once or time.time() >= a.deadline_epoch:
                return
            time.sleep(min(300, max(1, a.deadline_epoch - time.time())))


if __name__ == '__main__':
    main()
