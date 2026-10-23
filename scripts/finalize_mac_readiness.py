"""Recheck a completed planner repair with the shipped synthetic test fixture present.

Keep original failed test logs, reuse only hash-matched TLS/model evidence, and
publish a new readiness report. Never starts EC2 or acquires final-test cases.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from durable_worker import write
from validate_higgs_planner import source_contract

from greenshaping.testbed.readiness import cloud_calibration_gates
from greenshaping.utils.provenance import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    root, app = a.project, Path.cwd()
    prior = root / 'max_preparation_03'
    state = {'stage': 'waiting_precision_validation', 'cloud_used': False, 'started_epoch': time.time()}
    data, transports, planners, stability, code, verified = {}, {}, {}, [], None, False
    try:
        write(a.out / 'status.json', state)
        deadline = time.monotonic() + 3600
        while True:
            q = json.loads((root / 'precision_queue_01/state.json').read_text())
            if q.get('queue_status') == 'idle_complete':
                break
            if time.monotonic() > deadline:
                raise TimeoutError('precision validation dependency exceeded one hour')
            time.sleep(10)
        source = json.loads((root / 'evidence/source_audit/source_manifest.json').read_text())
        audit = json.loads((root / 'evidence/banks/bank_audit.json').read_text())
        data = {'source_sha256': source['sha256'], 'rows': source['rows'],
                'reference_preparation_pass': audit['reference_preparation_pass']}
        evidence = json.loads((prior / 'reused_transport_evidence.json').read_text())
        for name in evidence['protected_modules']:
            if file_sha256(root / 'mirror_app_04' / name) != file_sha256(app / name):
                raise ValueError('transport evidence incompatible with ' + name)
        for name, record in evidence['runs'].items():
            path = Path(record['path'])
            if file_sha256(path) != record['sha256']:
                raise ValueError('transport summary changed')
            transports[name] = json.loads(path.read_text())
        expected = source_contract()
        for phase in ('train', 'select'):
            for n in (5, 10, 20):
                for model in ('mistral', 'qwen14b'):
                    s = json.loads((prior / f'planner/{phase}/n{n}/{model}/summary.json').read_text())
                    if s['source_contract'] != expected:
                        raise ValueError('planner source hash mismatch')
                    planners[f'{phase}/n{n}/{model}'] = s
        commands = json.loads((prior / 'commands.json').read_text())
        stability = [v['returncode'] for v in commands if v['name'].startswith('transport_stability_')]
        state['stage'] = 'full_regression_with_complete_shipped_fixture'
        write(a.out / 'status.json', state)
        with (a.out / 'full_regression.log').open('wb') as log:
            code = subprocess.run([sys.executable, '-m', 'pytest', '-q'], stdout=log,
                                  stderr=subprocess.STDOUT, timeout=900).returncode
        write(a.out / 'regression_result.json', {'returncode': code, 'source_app': str(app),
              'previous_failure': 'missing shipped outputs/synthetic_demo; original log retained in max_preparation_03',
              'fixture_hashes': {str(p.relative_to(app)): file_sha256(p)
                                for p in (app / 'outputs/synthetic_demo').rglob('*') if p.is_file()}})
        state['stage'] = 'verify_complete_cloud_package'
        write(a.out / 'status.json', state)
        old_package = prior / 'cloud_package'
        previous_hashes = json.loads((old_package / 'SHA256.json').read_text())
        for name, digest in previous_hashes.items():
            if file_sha256(old_package / name) != digest:
                raise ValueError('prior cloud package hash changed')
        package = a.out / 'cloud_package'
        shutil.copytree(old_package, package, ignore=shutil.ignore_patterns('runtime.tar.gz', 'SHA256.json'))
        with tarfile.open(package / 'runtime.tar.gz', 'w:gz') as archive:
            for folder in ('src', 'scripts', 'tests', 'configs', 'outputs/synthetic_demo'):
                for path in sorted((app / folder).rglob('*')):
                    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                        archive.add(path, arcname=str(path.relative_to(app)), recursive=False)
            for name in ('pyproject.toml', 'README.md'):
                archive.add(app / name, arcname=name, recursive=False)
        hashes = {str(p.relative_to(package)): file_sha256(p) for p in package.rglob('*') if p.is_file()}
        write(package / 'SHA256.json', hashes)
        verified = all(file_sha256(package / path) == digest for path, digest in hashes.items())
        write(a.out / 'evidence_links.json', {'source_run': str(prior), 'transport': evidence,
              'source_contract': expected, 'planner_summaries': planners, 'stability_exits': stability})
        state['stage'] = 'completed'
    except Exception as exc:
        state.update(stage='completed_with_findings', error=repr(exc))
    finally:
        ready = cloud_calibration_gates(data, code, stability, transports, planners, verified)
        ready.update(run_root=str(a.out), epoch=time.time(), cloud_used=False, test_acquired=False)
        write(a.out / 'readiness.json', ready)
        if not ready['ready_for_ec2_calibration']:
            state['stage'] = 'completed_with_findings'
        state.update(finished_epoch=time.time(), ready_for_ec2_calibration=ready['ready_for_ec2_calibration'])
        write(a.out / 'status.json', state)
        text = ('Mac software preparation passed. Request T1-T4 for a bounded attended calibration; '
                'main paper acquisition, physical energy and matched NetShaper still pending.') if ready[
                    'ready_for_ec2_calibration'] else 'Keep EC2 stopped. Failed gates: ' + ', '.join(ready['failed_gates'])
        (a.out / 'EC2_START_NOTICE.txt').write_text(text + '\n')
    if not ready['ready_for_ec2_calibration']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
