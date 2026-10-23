"""Privileged, bounded capture supervisor: only the five pilot TCP ports on lo0."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--base-port', type=int, default=18500)
    p.add_argument('--daemon', action='store_true')
    p.add_argument('--duration', type=int, default=3600, help='bounded supervisor lifetime in seconds, at most three hours')
    args = p.parse_args()
    root = Path(args.run).resolve()
    if not root.is_dir() or not 1024 <= args.base_port <= 65530 or not 60 <= args.duration <= 10800:
        raise ValueError('invalid dedicated capture directory/ports')
    if args.daemon:
        child = os.fork()
        if child:
            print(json.dumps({'capture_supervisor_pid': child}))
            return
        os.setsid()
        with open(os.devnull, 'rb') as f:
            os.dup2(f.fileno(), 0)
        with (root / 'capture_supervisor.log').open('ab', buffering=0) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
    deadline = time.monotonic() + args.duration
    child = None
    try:
        while time.monotonic() < deadline:
            if (root.parent.parent / 'workflow.exit').exists():
                break
            try:
                state = json.loads((root / 'pilot_status.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(.2)
                continue
            if state['phase'] == 'awaiting_capture' and child is None:
                flt = ' or '.join(f'tcp port {x}' for x in range(args.base_port, args.base_port + 5))
                log = (root / 'tcpdump.log').open('wb')
                child = subprocess.Popen(['/usr/sbin/tcpdump', '-U', '-i', 'lo0', '-s', '128', '-w',
                                          str(root / 'traffic.pcap'), flt], stdout=log, stderr=log)
                time.sleep(.5)
                if child.poll() is not None:
                    raise RuntimeError('tcpdump could not start')
                (root / 'capture.ready').write_text(str(child.pid))
            if state['phase'] in ('awaiting_capture_stop', 'failed', 'complete'):
                break
            if child and child.poll() is not None:
                raise RuntimeError('capture exited unexpectedly')
            time.sleep(.2)
    finally:
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)
            child.wait(timeout=15)
        (root / 'capture.stopped').write_text(json.dumps({'returncode': child.returncode if child else None}))
        if (root / 'traffic.pcap').exists():
            (root / 'traffic.pcap').chmod(0o644)


if __name__ == '__main__':
    main()
