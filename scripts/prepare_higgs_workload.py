"""Prepare one projected HIGGS store and N=5/10/20 reference banks, without cloud."""
import argparse
import json
from pathlib import Path

from greenshaping.testbed.higgs_workload import build_banks, prepare_projection

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--store', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if not (a.store / 'projection.json').exists():
    prepare_projection(a.source, a.store)
report = build_banks(a.store, a.out)
print(json.dumps({'reference_preparation_pass': report['reference_preparation_pass'], 'audit': str(a.out / 'bank_audit.json')}))
if not report['reference_preparation_pass']:
    raise SystemExit(2)
