"""Bounded Mac Studio paired acquisition; stdlib only, no old bank access.

New generated integer-table workloads, a scripted feedback controller, and
three logical loopback sites. TRAIN fixes the public contract before SELECT.
An optional, narrowly filtered tcpdump records IP lengths, including lifecycle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import signal
import ssl
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from greenshaping.controller.workflow_contract import WorkflowEnvelope
from greenshaping.testbed.paced_workflow_tls import PacedTLSSite, run_paced_tls, slots
from greenshaping.utils.evidence import SessionLedger, atomic_json

PORTS = (55080, 55081, 55082)
SEED = 2026102307


def tables(seed, size):
    return [[random.Random(seed * 1000003 + i * 10007 + k).randrange(100)
             for k in range(size)] for i in range(3)]


def tool(table, request):
    count, total, groups = 0, 0, [0, 0, 0, 0]
    for value in table:
        if value >= request['threshold']:
            count += 1
            total += value
            groups[value % 4] += 1
    return {'count': count, 'sum': total, 'groups': groups}


def direct_tool(table, request):
    chosen = [v for v in table if v >= request['threshold']]
    return {'count': len(chosen), 'sum': sum(chosen),
            'groups': [sum(v % 4 == i for v in chosen) for i in range(4)]}


def plan_for(job, r, previous):
    feedback = sum(v['sum'] + v['count'] for v in previous.values())
    index = (job['seed'] + feedback) % 3
    active = range(3) if job['family'] == 'all_sites' else [index if job['family'] == 'feedback_route' else job['seed'] % 3]
    threshold = (job['seed'] + feedback + 7 * r) % 100
    return [{'threshold': threshold, 'round': r} if i in active else None for i in range(3)]


def reference(job, data):
    result, prior = [], {}
    for r in range(job['actual_rounds']):
        calls = plan_for(job, r, prior)
        prior = {i: direct_tool(data[i], call) for i, call in enumerate(calls) if call is not None}
        result.append(prior)
    return result


def profile(job, compute_bound=None, planning_bound=None):
    # Delays are controlled workload perturbations, not measured LLM delays.
    shifted = job['cohort'] == 'select_shifted'
    compute = .018 if shifted else .006 + (job['seed'] % 4) * .002
    planning = .008 if shifted else .001 + (job['seed'] % 3) * .001
    if job.get('stress') == 'compute_over':
        compute = compute_bound * 1.5
    elif job.get('stress') == 'compute_miss':
        compute = compute_bound + planning_bound + .16
    elif job.get('stress') == 'planning_over':
        planning = planning_bound * 2.5
    return compute, planning


def workload_plan(seed_offset=0, policy_names=('envelope', 'periodic')):
    families = ('single_site', 'feedback_route', 'all_sites')
    jobs = []
    for i in range(32):
        jobs.append({'job_id': 'select_%02d' % i, 'seed': 2000 + i,
                     'cohort': 'select_base' if i < 16 else 'select_shifted',
                     'family': families[i % 3], 'actual_rounds': i % 3 + 1,
                     'rows_per_site': 1024 if i < 16 else 4096})
    for stress in ('compute_over', 'compute_miss', 'planning_over', 'call_oversize', 'reply_oversize'):
        for i in range(2):
            jobs.append({'job_id': stress + '_%d' % i, 'seed': 3000 + len(jobs),
                         'cohort': 'stress', 'stress': stress, 'family': families[i],
                         'actual_rounds': 2, 'rows_per_site': 1024})
    trials = []
    for job in jobs:
        job['seed'] += seed_offset
    rng = random.Random(SEED + seed_offset)
    rng.shuffle(jobs)
    for job in jobs:
        policies = list(policy_names)
        rng.shuffle(policies)
        trials.extend({'job': job, 'policy': policy} for policy in policies)
    return trials


def calibrate():
    rows = []
    for i in range(12):
        job = {'seed': 1000 + i, 'cohort': 'train', 'family': 'all_sites', 'actual_rounds': 1}
        data = tables(job['seed'], 512)
        compute, planning = profile(job)
        began = time.monotonic()
        time.sleep(planning)
        calls = plan_for(job, 0, {})
        planning_elapsed = time.monotonic() - began
        times = []
        for table, call in zip(data, calls):
            began = time.monotonic()
            time.sleep(compute)
            assert tool(table, call) == direct_tool(table, call)
            times.append(time.monotonic() - began)
        rows.append({'seed': job['seed'], 'planning_s': planning_elapsed, 'compute_s': times})
    interval = .04
    bound = lambda maximum: max(interval, math.ceil((2 * maximum + .005) / interval) * interval)
    return rows, WorkflowEnvelope(3, interval, bound(max(max(r['compute_s']) for r in rows)),
                                  bound(max(r['planning_s'] for r in rows)), (512,) * 3, (1024,) * 3)


def start_capture(out, password):
    """Privilege is limited to tcpdump and stopping that exact child process."""
    capture = out / 'loopback.pcap'
    capture.touch(mode=0o600, exist_ok=False)
    log = (out / 'tcpdump.log').open('wb')
    command = ['/usr/sbin/tcpdump', '-U', '-n', '-i', 'lo0', '-s', '128', '-G', '300', '-W', '1',
               '-w', str(capture), 'ip and tcp and portrange 55080-55082']
    if password:
        auth = subprocess.run(['/usr/bin/sudo', '-S', '-p', '', '-v'], input=(password + '\n').encode(),
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
        if auth.returncode:
            log.close()
            return None, {'started': False, 'reason': 'sudo authentication unavailable; no retry'}
        command = ['/usr/bin/sudo', '-n'] + command
    process = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    log.close()
    time.sleep(.5)
    if process.poll() is not None:
        return None, {'started': False, 'returncode': process.returncode}
    return process, {'started': True, 'pid': process.pid, 'filter': 'ip and tcp and portrange 55080-55082',
                     'snaplen': 128, 'maximum_capture_seconds': 300,
                     'interface': 'lo0', 'physical_network_observation': False}


def stop_capture(process, privileged):
    if process is None:
        return
    if process.poll() is None:
        if privileged:
            subprocess.run(['/usr/bin/sudo', '-n', '/bin/kill', '-INT', str(process.pid)],
                           capture_output=True, timeout=10)
        else:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if privileged:
                subprocess.run(['/usr/bin/sudo', '-n', '/bin/kill', '-TERM', str(process.pid)],
                               capture_output=True, timeout=10)
            else:
                process.terminate()
            process.wait(timeout=5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--capture-sudo', action='store_true', help='receive the existing SSH credential only via stdin')
    p.add_argument('--no-capture', action='store_true')
    p.add_argument('--limit-trials', type=int, help='smaller values only for engineering smoke')
    p.add_argument('--policies', default='envelope,periodic')
    p.add_argument('--reuse-contract', type=Path, help='retain an earlier frozen public contract without recalibration')
    p.add_argument('--seed-offset', type=int, default=0, help='fresh generated jobs; does not change any public bound')
    a = p.parse_args()
    password = sys.stdin.readline().rstrip('\n') if a.capture_sudo else None
    # Refuse unsupported system LibreSSL before calibration or any dispatch.
    context_probe = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context_probe.minimum_version = ssl.TLSVersion.TLSv1_3
    a.out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(a.out).free < 15 * 2**30:
        raise OSError('minimum 15 GiB free required')
    if a.reuse_contract:
        original = json.loads(a.reuse_contract.read_text())['contract']
        original['call_caps'], original['reply_caps'] = tuple(original['call_caps']), tuple(original['reply_caps'])
        envelope = WorkflowEnvelope(**original)
        calibration = {'reused_from': str(a.reuse_contract),
                       'original_frozen_plan_sha256': hashlib.sha256(a.reuse_contract.read_bytes()).hexdigest(),
                       'recalibrated': False}
    else:
        measurements, envelope = calibrate()
        calibration = {'measurements': measurements, 'selection_rule':
                       'tau=.04; C,G=ceil((2*TRAIN max+.005)/tau)*tau, min tau; frame caps fixed 512/1024'}
    atomic_json(a.out / 'train_calibration.json', calibration)
    policy_names = a.policies.split(',')
    for name in policy_names:
        slots(envelope, name, 'up')
    trials = workload_plan(a.seed_offset, policy_names)[:a.limit_trials]
    plan = {'seed': SEED, 'contract': asdict(envelope), 'trials': trials,
            'python_version': sys.version, 'ssl_version': ssl.OPENSSL_VERSION,
            'seed_offset': a.seed_offset, 'contract_recalibrated': not bool(a.reuse_contract),
            'policy_names': policy_names,
            'physical_hosts': 1, 'logical_sites': 3, 'host': os.uname().nodename,
            'cpu_count': os.cpu_count(), 'load_before': os.getloadavg(),
            'scope': 'generated integer tables and scripted feedback, paired local TLS acquisition',
            'existing_train_select_modified': False, 'existing_final_accessed': False,
            'max_seconds': 240, 'max_artifact_bytes': 256 * 2**20, 'minimum_free_bytes': 15 * 2**30,
            'no_automatic_retry': True, 'positive_gate': 'reference + no client/site errors + all fixed frames + lag<=tau',
            'stress_gate': 'failure/violation reported + all fixed frames; semantic success alone never clears a bound violation'}
    source_root = Path(__file__).resolve().parents[1]
    source_names = ('scripts/benchmark_local_contract_tls.py',
                    'src/greenshaping/controller/workflow_contract.py',
                    'src/greenshaping/testbed/paced_workflow_tls.py',
                    'src/greenshaping/testbed/workflow_cover.py',
                    'src/greenshaping/utils/evidence.py')
    plan['source_sha256'] = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                             for name in source_names}
    atomic_json(a.out / 'frozen_plan.json', plan, exclusive=True)
    ledger = SessionLedger(a.out / 'ledger', plan, minimum_free_bytes=15 * 2**30)
    capture, capture_status = (None, {'started': False, 'reason': 'explicitly disabled'}) if a.no_capture else start_capture(a.out, password)
    password = None
    atomic_json(a.out / 'capture_status.json', capture_status)
    records = []
    deadline = time.monotonic() + 240
    try:
        with tempfile.TemporaryDirectory(prefix='gs-paired-tls-') as temp:
            cert, key = Path(temp) / 'cert.pem', Path(temp) / 'key.pem'
            subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                            '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                            '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
            key.chmod(0o600)
            for k, trial in enumerate(trials):
                if time.monotonic() > deadline:
                    raise TimeoutError('no new dispatch after bounded acquisition deadline')
                if sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file()) > 256 * 2**20:
                    raise OSError('bounded artifact budget exhausted')
                job, policy = trial['job'], trial['policy']
                sid = '%03d_%s_%s' % (k, job['job_id'], policy)
                ledger.start(sid, trial)
                row = {'trial_id': sid, **trial, 'start_epoch': time.time(), 'load': os.getloadavg(), 'passed': False}
                sites = []
                try:
                    data = tables(job['seed'], job['rows_per_site'])
                    expected = reference(job, data)
                    compute, planning = profile(job, envelope.compute_bound, envelope.planning_bound)

                    def handler(i, request):
                        time.sleep(compute)
                        value = tool(data[i], request)
                        if job.get('stress') == 'reply_oversize':
                            value['oversize'] = 'x' * 2048
                        return value

                    sites = [PacedTLSSite(envelope, policy, i, cert, key,
                                          lambda request, i=i: handler(i, request), PORTS[i]) for i in range(3)]

                    def planner(r, previous):
                        time.sleep(planning)
                        values = plan_for(job, r, previous)
                        if job.get('stress') == 'call_oversize':
                            for value in values:
                                if value is not None:
                                    value['oversize'] = 'x' * 1024
                        return values

                    result = run_paced_tls(envelope, policy, [s.address for s in sites], cert,
                                           planner, job['actual_rounds'])
                    for site in sites:
                        site.close()
                    site_errors = [x for site in sites for x in site.errors]
                    logs = result['client_logs'] + [x for site in sites for x in site.logs]
                    expected_count = 3 * (len(slots(envelope, policy, 'up')) + len(slots(envelope, policy, 'down')))
                    expected_bytes = len(slots(envelope, policy, 'up')) * sum(envelope.call_caps) + len(slots(envelope, policy, 'down')) * sum(envelope.reply_caps)
                    frames_ok = len(logs) == expected_count and result['application_bytes'] == expected_bytes
                    timing_ok = frames_ok and max(x['lag_s'] for x in logs) <= envelope.interval
                    semantic_ok = result['responses'] == expected
                    success = result['completed'] and not site_errors and semantic_ok and frames_ok and timing_ok
                    detected = bool(result['errors'] or site_errors)
                    row.update(result=result, site_logs=[s.logs for s in sites], site_errors=site_errors,
                               reference_match=semantic_ok, fixed_frames_valid=frames_ok, timing_valid=timing_ok,
                               observed_contract_success=bool(success), detected_failure=detected,
                               expected_contract_failure=job['cohort'] == 'stress',
                               passed=bool((detected and frames_ok and timing_ok) if job['cohort'] == 'stress' else success))
                except Exception as exc:
                    row['acquisition_error'] = repr(exc)
                    for site in sites:
                        try:
                            site.close()
                        except Exception:
                            pass
                row['end_epoch'] = time.time()
                ledger.commit(sid, row)
                records.append(row)
                atomic_json(a.out / 'progress.json', {'committed': len(records), 'expected': len(trials),
                            'passed_engineering_gates': sum(r['passed'] for r in records),
                            'last_trial': sid, 'last_passed': row['passed']})
                print('%d/%d %s passed=%s' % (len(records), len(trials), sid, row['passed']), flush=True)
                if 'acquisition_error' in row:
                    # Infrastructure errors must not turn into repeated acquisitions.
                    break
                time.sleep(.02)
    finally:
        stop_capture(capture, a.capture_sudo)
        ledger.checkpoint()
    summary = {'completed': len(records) == len(trials), 'trials': len(records),
               'all_engineering_gates_passed': all(r['passed'] for r in records),
               'positive_trials': sum(r['job']['cohort'] != 'stress' for r in records),
               'positive_contract_successes': sum(r.get('observed_contract_success', False) for r in records if r['job']['cohort'] != 'stress'),
               'stress_trials': sum(r['job']['cohort'] == 'stress' for r in records),
               'stress_detected_with_fixed_frames': sum(r['passed'] for r in records if r['job']['cohort'] == 'stress'),
               'physical_hosts': 1, 'logical_sites': 3, 'cloud_sessions': 0,
               'scope': plan['scope'], 'wire_privacy_certified': False, 'energy_measured': False,
               'final_bank_opened': False, 'private_tls_keys_retained': False,
               'capture_started': capture_status['started'], 'capture_returncode': None if capture is None else capture.returncode,
               'load_after': os.getloadavg(), 'ledger': ledger.audit()}
    atomic_json(a.out / 'summary.json', summary)
    manifest = {str(f.relative_to(a.out)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(a.out.rglob('*')) if f.is_file()}
    atomic_json(a.out / 'SHA256.json', manifest)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
