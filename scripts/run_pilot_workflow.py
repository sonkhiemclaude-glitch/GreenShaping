"""Supervise tests, TLS stress, integration, balanced acquisition and evaluations."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from greenshaping.utils.io import write_json


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    python = sys.executable
    smoke, full = root / 'runs/integration_01', root / 'runs/balanced_01'
    for d in (smoke, full):
        d.mkdir(parents=True, exist_ok=True)
    stages = [
        ('tests', [python, '-m', 'pytest', '-q']),
        ('stress', [python, 'scripts/stress_tls_transport.py', '--out', str(root / 'runs/tls_stress'), '--sessions', '150']),
        ('integration', [python, 'scripts/run_pilot_v2.py', '--out', str(smoke), '--smoke', '--summarize', '--include-rounds']),
        ('acquisition', [python, 'scripts/run_pilot_v2.py', '--out', str(full)]),
        ('evaluation', [python, 'scripts/evaluate_pilot_v2.py', '--root', str(full), '--out', str(root / 'evaluation')]),
        ('controller', [python, 'scripts/analyze_pilot_controller_v2.py', '--root', str(full / 'none'),
                        '--out', str(root / 'controller')]),
    ]
    state = {'pid': os.getpid(), 'state': 'running', 'phase': 'start', 'completed_stages': []}
    code = 1
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', LANGCHAIN_TRACING_V2='false')
    try:
        for name, command in stages:
            state['phase'] = name
            write_json(state, root / 'workflow.json')
            with (root / f'{name}.log').open('wb') as log:
                result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=log)
            if result.returncode:
                state.update(state='failed', returncode=result.returncode)
                return result.returncode
            state['completed_stages'].append(name)
        state['state'] = 'complete'
        code = 0
        return 0
    finally:
        if state['state'] != 'complete':
            code = int(state.get('returncode', 1))
        write_json(state, root / 'workflow.json')
        (root / 'workflow.exit').write_text(str(code))


if __name__ == '__main__':
    raise SystemExit(main())
