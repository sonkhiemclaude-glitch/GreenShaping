"""Finite Mac-only preparation, after existing data/model jobs, with strict readiness gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from durable_worker import write
from validate_higgs_planner import source_contract

from greenshaping.testbed.readiness import cloud_calibration_gates
from greenshaping.utils.provenance import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dependency-queue', type=Path)
    p.add_argument('--reuse-transport-root', type=Path)
    p.add_argument('--reuse-source-root', type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    app = Path.cwd()
    data_root = a.project / 'evidence'
    dependency_queue = a.dependency_queue or (a.project / 'recovery_queue_03')
    state = {'stage': 'waiting_existing_queue', 'cloud_used': False, 'test_acquired': False, 'started_epoch': time.time()}
    tests_exit, stability, transports, planners, verified, data = None, [], {}, {}, False, {}
    commands = []
    work_deadline = None
    active = None

    def interrupted(sig, frame):
        state['interrupted_signal'] = sig
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def status(stage):
        state.update(stage=stage, updated_epoch=time.time())
        write(a.out / 'status.json', state)

    def run(name, argv, timeout):
        nonlocal active
        if time.monotonic() > work_deadline:
            raise TimeoutError('Mac postprocessing work window expired')
        directory = a.out / 'commands' / name
        directory.mkdir(parents=True)
        started = time.time()
        timeout = min(timeout, max(1, work_deadline - time.monotonic()))
        with (directory / 'output.log').open('wb') as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
            active = process
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Each step owns a process group, including any server children.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
            finally:
                active = None
        result = {'name': name, 'argv': argv, 'returncode': code, 'start_epoch': started, 'end_epoch': time.time()}
        commands.append(result)
        write(directory / 'result.json', result)
        write(a.out / 'commands.json', commands)
        return code

    try:
        status('waiting_existing_queue')
        wait_deadline = time.monotonic() + 14400
        while True:
            queue = json.loads((dependency_queue / 'state.json').read_text())
            if queue.get('queue_status') == 'idle_complete':
                break
            if time.monotonic() > wait_deadline:
                raise TimeoutError('existing data/model queue did not finish within four hours')
            time.sleep(15)
        source = json.loads((data_root / 'source_audit/source_manifest.json').read_text())
        audit = json.loads((data_root / 'banks/bank_audit.json').read_text())
        data = {'source_sha256': source['sha256'], 'rows': source['rows'],
                'reference_preparation_pass': audit.get('reference_preparation_pass', False)}
        if not data['reference_preparation_pass']:
            raise RuntimeError('real HIGGS contribution-size audit failed; no experiment launched')
        work_deadline = time.monotonic() + 14400
        contract = {'source_contract': source_contract(), 'models': ['mistral-small3.1:24b', 'qwen2.5:14b-instruct-q4_K_M'],
                    'frozen_epoch': time.time(), 'final_test_policy': 'no execution or tuning',
                    'bank_hashes': {f'{phase}/n{n}': file_sha256(data_root / f'banks/n{n}/{phase}.jsonl')
                                    for phase in ('train', 'select') for n in (5, 10, 20)}}
        write(a.out / 'frozen_planner_contract.json', contract)
        status('mac_full_regression')
        tests_exit = run('full_regression', [sys.executable, '-m', 'pytest', '-q'], 900)
        for i in range(3):
            stability.append(run(f'transport_stability_{i}', [sys.executable, '-m', 'pytest', '-q',
                                 'tests/test_transport_lifecycle.py',
                                 'tests/test_protocol_v2.py::test_live_composition_rounds_padding_dp'], 180))
        if a.reuse_transport_root:
            if not a.reuse_source_root:
                raise ValueError('reusing TLS evidence requires its exact frozen source root')
            old_contract = json.loads((a.reuse_transport_root / 'frozen_planner_contract.json').read_text())
            if old_contract['bank_hashes'] != contract['bank_hashes']:
                raise ValueError('cannot reuse TLS evidence from different question banks')
            protected = ['src/greenshaping/testbed/transport.py', 'src/greenshaping/testbed/queries.py',
                         'src/greenshaping/testbed/higgs_workload.py', 'src/greenshaping/testbed/higgs_gateway.py',
                         'scripts/mac_higgs_tls_check.py', 'scripts/higgs_tls_endpoint.py',
                         'scripts/check_higgs_endpoint_roles.py']
            for name in protected:
                if file_sha256(a.reuse_source_root / name) != file_sha256(app / name):
                    raise ValueError('cannot reuse TLS evidence after changing ' + name)
            evidence = {}
            for key, folder in [('separate', 'tls_separate'), ('routed', 'tls_routed'), ('portable_roles', 'portable_roles')]:
                path = a.reuse_transport_root / folder / 'summary.json'
                transports[key] = json.loads(path.read_text())
                evidence[key] = {'path': str(path), 'sha256': file_sha256(path),
                                 'all_passed': transports[key].get('all_passed')}
            write(a.out / 'reused_transport_evidence.json', {'runs': evidence, 'protected_modules': protected,
                  'bank_hashes': contract['bank_hashes'], 'reason': 'only planner number serialization and config packaging changed'})
        else:
            status('higgs_real_tls_matrix')
            for layout, reps in [('separate', 2), ('routed', 1)]:
                destination = a.out / ('tls_' + layout)
                run('tls_' + layout, [sys.executable, 'scripts/mac_higgs_tls_check.py', '--store', str(data_root / 'store'),
                     '--banks', str(data_root / 'banks'), '--out', str(destination), '--server-layout', layout,
                     '--repetitions', str(reps), '--max-seconds', '5400'], 5700)
                if (destination / 'summary.json').exists():
                    transports[layout] = json.loads((destination / 'summary.json').read_text())
            status('portable_cloud_role_validation')
            destination = a.out / 'portable_roles'
            run('portable_roles', [sys.executable, 'scripts/check_higgs_endpoint_roles.py', '--store', str(data_root / 'store'),
                 '--banks', str(data_root / 'banks'), '--out', str(destination), '--N', '20'], 1500)
            if (destination / 'summary.json').exists():
                transports['portable_roles'] = json.loads((destination / 'summary.json').read_text())
        status('frozen_planner_train_select_tools')
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=30) as response:
            (a.out / 'ollama_inventory.json').write_bytes(response.read())
        for phase in ('train', 'select'):
            for n in (5, 10, 20):
                for model, label in [('mistral-small3.1:24b', 'mistral'), ('qwen2.5:14b-instruct-q4_K_M', 'qwen14b')]:
                    directory = a.out / 'planner' / phase / f'n{n}' / label
                    run(f'{phase}_n{n}_{label}', [sys.executable, 'scripts/validate_higgs_planner.py',
                        '--bank', str(data_root / f'banks/n{n}/{phase}.jsonl'), '--store', str(data_root / 'store'),
                        '--out', str(directory), '--sites', str(n), '--model', model, '--partition', phase,
                        '--frozen-contract', str(a.out / 'frozen_planner_contract.json')], 1200)
                    if (directory / 'summary.json').exists():
                        planners[f'{phase}/n{n}/{label}'] = json.loads((directory / 'summary.json').read_text())
        status('freeze_and_verify_cloud_bundle')
        bundle = a.out / 'cloud_package'
        bundle.mkdir()
        references = bundle / 'references'
        references.mkdir()
        for n in (5, 10, 20):
            code = run(f'cloud_reference_n{n}', [sys.executable, 'scripts/higgs_tls_endpoint.py', 'reference', '--N', str(n),
                       '--store', str(data_root / 'store'), '--bank', str(data_root / f'banks/n{n}/train.jsonl'),
                       '--out', str(references / f'n{n}.json')], 300)
            if code:
                raise RuntimeError('cloud reference generation failed')
        # Verify every canonical store file before packaging it. No raw test sessions.
        projection = json.loads((data_root / 'store/projection.json').read_text())
        for name, digest in projection['files'].items():
            if file_sha256(data_root / 'store' / name) != digest:
                raise ValueError('canonical store hash changed: ' + name)
        data_files = list((data_root / 'store').glob('*.npy')) + [data_root / 'store/projection.json']
        data_files += [data_root / f'banks/n{n}/train.jsonl' for n in (5, 10, 20)]
        with tarfile.open(bundle / 'higgs_store_and_train.tar.gz', 'w:gz') as archive:
            for path in data_files:
                archive.add(path, arcname=str(path.relative_to(data_root)), recursive=False)
        with tarfile.open(bundle / 'runtime.tar.gz', 'w:gz') as archive:
            for folder in ('src', 'scripts', 'tests', 'configs', 'outputs/synthetic_demo'):
                for path in sorted((app / folder).rglob('*')):
                    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                        archive.add(path, arcname=str(path.relative_to(app)), recursive=False)
            for name in ('pyproject.toml', 'README.md'):
                archive.add(app / name, arcname=name, recursive=False)
        # Re-read compressed data members and compare to their canonical sources.
        with tarfile.open(bundle / 'higgs_store_and_train.tar.gz', 'r:gz') as archive:
            for member in archive:
                stream = archive.extractfile(member)
                digest = hashlib.sha256()
                while chunk := stream.read(4 * 1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != file_sha256(data_root / member.name):
                    raise ValueError('cloud archive member failed hash verification: ' + member.name)
        package_hashes = {str(path.relative_to(bundle)): file_sha256(path)
                          for path in bundle.rglob('*') if path.is_file()}
        plan = {'phase': 'bounded_engineering_calibration_not_main_paper_acquisition', 'auto_start_ec2': False,
                'max_window_s': 5400, 'max_four_host_window_instance_hours': 6,
                'site_counts': [5, 10, 20], 'sessions': 360, 'partition': 'train',
                'roles': {'T1': 'orchestrator', 'T2': 'NetShaper client middlebox',
                          'T3': 'NetShaper server middlebox', 'T4': 'shared endpoint with logical HIGGS sites'},
                'green_shaping_role_command': 'python scripts/higgs_tls_endpoint.py {server|client} --help',
                'address_policy': 'revalidate current addresses and host keys after user starts instances',
                'certificate_policy': 'generate fresh endpoint certificate with actual server/proxy SANs; verify CA on client',
                'netshaper': 'separate matching capture integration still required; do not label loopback as NetShaper',
                'cost_policy': 'finite processes do not stop EC2 billing; establish AWS stop mechanism or attended stop window',
                'test_policy': 'final-test bank excluded from cloud data archive', 'energy_measured': False,
                'mac_runtime_s': {name: s.get('elapsed_s') for name, s in transports.items()},
                'cloud_duration_estimate_status': 'Mac timing only; refresh after short EC2 preflight, not a cloud throughput prediction'}
        write(bundle / 'calibration_plan.json', plan)
        write(bundle / 'SHA256.json', package_hashes)
        verified = all(file_sha256(bundle / path) == digest for path, digest in package_hashes.items())
        status('completed')
    except Exception as exc:
        state['error'] = repr(exc)
        status('completed_with_findings')
    finally:
        readiness = cloud_calibration_gates(data, tests_exit, stability, transports, planners, verified)
        readiness.update(epoch=time.time(), run_root=str(a.out), cloud_used=False, test_acquired=False,
                         initial_queue=str(dependency_queue), commands=commands)
        write(a.out / 'readiness.json', readiness)
        text = ('Mac software checks passed. T1-T4 can be requested for a bounded calibration; '
                'large paper acquisition is still pending.') if readiness['ready_for_ec2_calibration'] else (
                'Keep EC2 stopped. Pending gates: ' + ', '.join(readiness['failed_gates']))
        (a.out / 'EC2_START_NOTICE.txt').write_text(text + '\n')
        state.update(finished_epoch=time.time(), readiness_file=str(a.out / 'readiness.json'),
                     ready_for_ec2_calibration=readiness['ready_for_ec2_calibration'])
        if not readiness['ready_for_ec2_calibration']:
            state['stage'] = 'completed_with_findings'
        write(a.out / 'status.json', state)
        # Notification delivery depends on the user's macOS notification settings.
        notification = 'Software checks passed; review EC2_START_NOTICE before starting EC2.' if readiness[
            'ready_for_ec2_calibration'] else 'Preparation finished with findings. Keep EC2 stopped; review readiness.json.'
        try:
            notice = subprocess.run(['/usr/bin/osascript', '-e', 'display notification ' + json.dumps(notification) +
                                     ' with title "Green Shaping"'], capture_output=True, text=True, timeout=15)
            write(a.out / 'notification_attempt.json', {'returncode': notice.returncode,
                  'stderr': notice.stderr[-1000:], 'delivery_confirmed': False})
        except Exception as exc:
            write(a.out / 'notification_attempt.json', {'error': repr(exc), 'delivery_confirmed': False})
    if not readiness['ready_for_ec2_calibration']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
