"""Sequential new-workload pilot, with attacker selection locked before final test."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from greenshaping.utils.io import write_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    if (root / 'workflow.json').exists():
        raise FileExistsError('use a new workflow directory')
    smoke, full = root / 'runs/integration_01', root / 'runs/adaptive_01'
    for d in (smoke, full):
        d.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    selection = root / 'selection/cnn_selection.json'
    stages = [
        ('tests', [python, '-m', 'pytest', '-q']),
        ('integration', [python, 'scripts/run_pilot_v2.py', '--out', str(smoke), '--adaptive', '--smoke',
                         '--include-rounds', '--base-port', '18600', '--seed', '20261023']),
        ('acquisition', [python, 'scripts/run_pilot_v2.py', '--out', str(full), '--adaptive',
                         '--include-rounds', '--base-port', '18600', '--seed', '20261022']),
        ('selection', [python, 'scripts/select_pilot_v4_attackers.py', '--root', str(full), '--out', str(root / 'selection')]),
        ('evaluation', [python, 'scripts/evaluate_pilot_v2.py', '--root', str(full), '--out', str(root / 'evaluation'),
                        '--live-only', '--modes', 'none', 'pad', 'dp', 'rounds', 'combo', '--cnn-selection', str(selection),
                        '--seeds', '20261022', '20261023', '20261024', '20261025', '20261026']),
    ]
    state: dict = {'pid': os.getpid(), 'state': 'running', 'phase': 'start', 'completed_stages': []}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', LANGCHAIN_TRACING_V2='false')
    code = 1
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
