"""Finite Mac/local preparation pipeline; never starts or connects to EC2."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from greenshaping.testbed.higgs_workload import build_banks, prepare_projection, site_table
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.question_bank import load_question_bank

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
a.root.mkdir(parents=True, exist_ok=True)
started = time.time()
status = {'started_epoch': started, 'cloud_used': False, 'final_test_executed': False}
try:
    status['stage'] = 'source_download_and_schema_audit'
    (a.root / 'status.json').write_text(json.dumps(status, indent=2))
    subprocess.run([sys.executable, 'scripts/prepare_higgs_source.py', '--data', str(a.root / 'source'),
                    '--out', str(a.root / 'source_audit')], check=True, timeout=11400)
    status['stage'] = 'projection_and_bank_audit'
    (a.root / 'status.json').write_text(json.dumps(status, indent=2))
    store = a.root / 'store'
    if not (store / 'projection.json').exists():
        prepare_projection(a.root / 'source/HIGGS.csv.gz', store)
    banks = a.root / 'banks'
    report = build_banks(store, banks)
    status['reference_preparation_pass'] = report['reference_preparation_pass']
    status['stage'] = 'train_tool_timing'
    (a.root / 'status.json').write_text(json.dumps(status, indent=2))
    timings = []
    for n in (5, 10, 20):
        questions = load_question_bank(banks / f'n{n}/train.jsonl', require_min=False)
        # One real TRAIN call per family/size; diagnostic timing, not paper replicates.
        chosen = {}
        for q in questions:
            chosen.setdefault((q.family, q.metadata['requested_size']), q)
        for (family, size), q in chosen.items():
            c = q.plan[0][0]
            t = time.perf_counter()
            table = site_table(store, n, c['site'])
            load_s = time.perf_counter() - t
            t = time.perf_counter()
            result, rows = run_query(table, family, c['params'])
            timings.append({'n_sites': n, 'question_id': q.question_id, 'family': family, 'requested_size': size,
                            'rows': rows, 'site_load_s': load_s, 'tool_s': time.perf_counter() - t,
                            'result_bytes': len(json.dumps(result).encode()), 'independent_measurement_replicates': 1})
            (a.root / 'tool_timings.json').write_text(json.dumps(timings, indent=2))
    status.update(stage='completed' if report['reference_preparation_pass'] else 'completed_with_findings',
                  tool_diagnostic_calls=len(timings), ready_for_large_run=False)
except Exception as exc:
    status.update(stage='failed', error=repr(exc))
    raise
finally:
    status.update(end_epoch=time.time(), elapsed_s=time.time() - started)
    (a.root / 'status.json').write_text(json.dumps(status, indent=2))
