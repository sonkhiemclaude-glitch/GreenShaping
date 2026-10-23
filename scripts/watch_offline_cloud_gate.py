"""Finite local readiness/storage gate. Never starts or contacts EC2."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from greenshaping.utils.evidence import atomic_json


def inspect(results: Path, prepared_app: Path, evidence_exports: Path, minimum_free: int):
    path = results / 'readiness.json'
    result = json.loads(path.read_text()) if path.exists() else {}
    source_file = results / 'source_sha256.json'
    tested = json.loads(source_file.read_text()) if source_file.exists() else None
    actual = {str(p.relative_to(prepared_app)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ['src', 'scripts'] for p in (prepared_app / folder).rglob('*.py')}
    copy_checks = []
    for name, field in [('ec2_84_metrics_verified.json', 'cross_host_verified'),
                         ('mac_complete_controller_copy_verified.json', 'cross_host_copy_verified')]:
        p = evidence_exports / name
        copy_checks.append(p.exists() and json.loads(p.read_text()).get(field) is True)
    gates = {'software_resilience_verified': result.get('software_resilience_verified') is True,
             'frozen_source_matches_prepared_batch': tested is not None and tested == actual,
             'existing_metrics_on_two_hosts': all(copy_checks),
             'storage_at_least_10GiB': shutil.disk_usage(prepared_app).free >= minimum_free}
    failed = bool(result) and not gates['software_resilience_verified']
    failed |= tested is not None and not gates['frozen_source_matches_prepared_batch']
    return gates, failed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--prepared-app', type=Path, required=True)
    p.add_argument('--exports', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    while True:
        gates, failed = inspect(a.results, a.prepared_app, a.exports, 10 * 1024**3)
        ready = all(gates.values())
        state = {'gates': gates, 'cloud_started': False, 'ready_for_main_paper_run': False,
                 'run_root': str(a.out), 'epoch': time.time(), 'free_bytes': shutil.disk_usage(a.prepared_app).free}
        if ready or failed or time.time() >= a.deadline_epoch:
            state['ready_for_ec2_calibration'] = ready and not failed
            atomic_json(a.out / 'readiness.json', state)
            return
        state['status'] = 'waiting_for_offline_checks_or_storage'
        atomic_json(a.out / 'status.json', state)
        time.sleep(min(60, max(1, a.deadline_epoch - time.time())))


if __name__ == '__main__':
    main()
