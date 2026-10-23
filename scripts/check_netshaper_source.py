"""Read-only source pin check and tiny official-simulator smoke test, not a baseline result."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from greenshaping.utils.io import write_json
from greenshaping.utils.provenance import file_sha256

PIN = 'd25e006771e39fce1b91bf848e8c761e97caab67'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root = Path(args.source).resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN or subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
        raise ValueError('source does not match the unmodified pinned official release')
    sys.path.insert(0, str(root / 'simulator'))
    app_class = importlib.import_module('src.modules.Application').Application
    transport = importlib.import_module('src.DP_transport_dynamic').DP_transport_dynamic
    # The upstream Application slices exclude the final data column; a public zero
    # sentinel prevents accidental loss of the last real bin in this smoke test.
    data = pd.DataFrame([[1024, 2048, 0, 0, 0], [0, 512, 4096, 0, 1]], columns=[0, 1, 2, 3, 'label'])
    np.random.seed(20261021)
    app, emitted, dummy = transport(app_class(1, data, 1), 1, 'Gaussian', 1., 1024., 3., 128., 65536.)
    for a, b in zip(app.drop(columns='label').sum(axis=1), emitted.drop(columns='label').sum(axis=1), strict=True):
        if b + 1e-6 < a:
            raise RuntimeError('upstream simulator lost bytes')
    files = ['README.md', 'simulator/src/modules/Application.py', 'simulator/src/DP_transport_dynamic.py']
    write_json({'source': 'https://github.com/ubc-systopia/netshaper', 'branch': 'release/sec24', 'commit': commit,
                'sha256': {f: file_sha256(root / f) for f in files}, 'smoke_passed': True,
                'scope': 'official simulator import and tiny synthetic data execution only; no network or energy comparison',
                'upstream_observations': ['Application.generate_data excludes the final data column; zero sentinel used',
                                          'dynamic transport drains queues after application termination; differs from fixed-H truncation'],
                'hardware_blocker': 'official physical topology and Ubuntu x86 middleboxes absent'}, args.out)


if __name__ == '__main__':
    main()
