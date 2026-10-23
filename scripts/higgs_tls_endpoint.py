"""Portable HIGGS TLS roles for a bounded future cloud calibration.

This does not provision EC2 or start NetShaper. The same role is exercised on
Mac loopback before cloud use. A shared endpoint represents logical sites only.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from mac_higgs_tls_check import equivalent, modes

from greenshaping.testbed.higgs_gateway import RoutedTool, routed_calls
from greenshaping.testbed.higgs_workload import site_table
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.question_bank import load_question_bank
from greenshaping.testbed.transport import ShapedTransport, TLSToolServer
from greenshaping.utils.evidence import SessionLedger, atomic_json
from greenshaping.utils.evidence_replica import EvidenceReplica
from greenshaping.utils.provenance import file_sha256


def selected(bank):
    questions = load_question_bank(bank, require_min=False)
    if not questions or any(q.metadata.get('partition') != 'train' for q in questions):
        raise ValueError('only TRAIN engineering calibration allowed')
    strata = {}
    for q in questions:
        strata.setdefault((q.family, q.metadata['requested_size']), q)
    if len(strata) != 12:
        raise ValueError('need four families and three contribution-size strata')
    return list(strata.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=['server', 'client', 'reference'])
    p.add_argument('--N', type=int, choices=[5, 10, 20], required=True)
    p.add_argument('--store', type=Path)
    p.add_argument('--bank', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5555)
    p.add_argument('--cert', type=Path)
    p.add_argument('--key', type=Path)
    p.add_argument('--endpoints', type=Path)
    p.add_argument('--modes', nargs='+', choices=['none', 'pad', 'rounds', 'dp', 'combo'], default=list(modes(5)))
    p.add_argument('--repetitions', type=int, choices=[1, 2], default=2)
    p.add_argument('--max-seconds', type=int, default=5400)
    p.add_argument('--replica-endpoint', help='private HTTPS evidence receiver, separate from measured data path')
    p.add_argument('--replica-cert', type=Path)
    p.add_argument('--replica-token', type=Path)
    a = p.parse_args()
    replica_args = [a.replica_endpoint, a.replica_cert, a.replica_token]
    if any(replica_args) and not all(replica_args):
        p.error('replica requires endpoint, trusted certificate and private token file')
    if not 1 <= a.max_seconds <= 5400:
        raise ValueError('role deadline must be within 1..5400 seconds')
    if a.role == 'server':
        if not all((a.store, a.cert, a.key)):
            p.error('server needs --store --cert --key')
        a.out.mkdir(parents=True, exist_ok=False)
        tool = RoutedTool(set(range(a.N)), lambda i: site_table(a.store, a.N, i))
        # Preload before readiness so cold loading is recorded separately from sessions.
        for i in range(a.N):
            tool.tables[i] = site_table(a.store, a.N, i)
        server = TLSToolServer(a.host, a.port, str(a.cert), str(a.key), tool)
        ready = {'ready': True, 'N': a.N, 'address': server.address, 'site_model': 'logical partitions on one physical server',
                 'epoch': time.time(), 'physical_energy_available': False}
        atomic_json(a.out / 'ready.json', ready)
        print(json.dumps(ready), flush=True)
        try:
            threading.Event().wait(a.max_seconds)
        finally:
            server.stop()
        return
    if not a.bank:
        p.error('reference/client needs --bank')
    questions = selected(a.bank)
    if a.role == 'reference':
        if not a.store:
            p.error('reference needs --store')
        tables = {}
        result = {}
        for q in questions:
            rows = []
            for c in q.plan[0]:
                site = c['site']
                if site not in tables:
                    tables[site] = site_table(a.store, a.N, site)
                value, count = run_query(tables[site], q.family, c['params'])
                rows.append({'result': value, 'matching_rows': count})
            result[str(q.question_id)] = rows
        with a.out.open('x') as f:
            json.dump({'N': a.N, 'partition': 'train', 'reference': result}, f, indent=2)
        return
    if not all((a.reference, a.cert, a.endpoints)):
        p.error('client needs --reference --cert (trusted CA) --endpoints')
    reference = json.loads(a.reference.read_text())
    endpoints = json.loads(a.endpoints.read_text())
    if reference['N'] != a.N or reference['partition'] != 'train' or len(endpoints) != a.N:
        raise ValueError('deployment/reference N or partition mismatch')
    a.out.mkdir(parents=True, exist_ok=False)
    ledger = SessionLedger(a.out / 'ledger', {
        'N': a.N, 'partition': 'train', 'modes': a.modes, 'repetitions': a.repetitions,
        'endpoints': endpoints, 'bank_sha256': file_sha256(a.bank),
        'reference_sha256': file_sha256(a.reference), 'runner_sha256': file_sha256(Path(__file__)),
        'configurations': {name: modes(a.N)[name].to_dict() for name in a.modes},
    })
    replica = EvidenceReplica(a.replica_endpoint, a.replica_cert, a.replica_token) if a.replica_endpoint else None
    records = []
    started, deadline = time.time(), time.monotonic() + a.max_seconds
    summary = {'N': a.N, 'scope': 'TRAIN shared-endpoint TLS correctness; capture and energy pending',
               'expected_sessions': 12 * len(a.modes) * a.repetitions, 'final_test_executed': False,
               'privacy_certified': False, 'large_run_ready': False}
    try:
        for name in a.modes:
            cfg = modes(a.N)[name]
            tr = ShapedTransport([tuple(v) for v in endpoints], str(a.cert), cfg, start_lead_s=1.)
            try:
                for rep in range(a.repetitions):
                    for q in questions:
                        if time.monotonic() > deadline:
                            raise TimeoutError('bounded client dispatch deadline reached')
                        row = {'mode': name, 'repetition': rep, 'question_id': q.question_id,
                               'start_epoch': time.time(), 'passed': False}
                        session_id = f'session_{len(records):04d}'
                        ledger.start(session_id, dict(row))
                        began = False
                        try:
                            tr.begin(called_sites=[c['site'] for c in q.plan[0]])
                            began = True
                            response = tr.call_round(q.family, routed_calls(q.plan[0]))
                            row['tool_completed_epoch'] = time.time()
                            row['responses'] = [{k: v[k] for k in ('result', 'matching_rows')} for v in response]
                            row['reference_match'] = equivalent(row['responses'], reference['reference'][str(q.question_id)])
                        except Exception as exc:
                            row['error'] = repr(exc)
                        finally:
                            if began:
                                try:
                                    row['transport'] = tr.finish()
                                except Exception as exc:
                                    row['finish_error'] = repr(exc)
                            row['end_epoch'] = time.time()
                            row['passed'] = (row.get('reference_match', False) and
                                             row.get('transport', {}).get('measurement_valid', False) and 'error' not in row)
                            ledger.commit(session_id, row)
                            if replica:
                                replica.session(ledger.root, session_id)
                            atomic_json(a.out / (session_id + '.json'), row)
                            records.append({k: v for k, v in row.items() if k not in ('transport', 'responses')})
                            ledger.checkpoint()
                            atomic_json(a.out / 'progress.json', {'sessions': len(records),
                                  'passed': sum(r['passed'] for r in records), 'last': records[-1]})
                        if not began:
                            raise RuntimeError('connection start barrier failed; no unsafe connection reuse')
                        if not row['passed']:
                            raise RuntimeError('invalid TRAIN calibration session retained; further acquisition stopped')
            finally:
                tr.close()
        summary['all_passed'] = len(records) == summary['expected_sessions'] and all(r['passed'] for r in records)
    except Exception as exc:
        summary.update(all_passed=False, error=repr(exc))
        raise
    finally:
        summary.update(sessions=len(records), passed=sum(r['passed'] for r in records),
                       elapsed_s=time.time() - started, checks=records)
        summary['evidence_audit'] = ledger.checkpoint()
        summary['synchronous_replica_enabled'] = replica is not None
        summary['replica_acknowledged_sessions'] = len(list((ledger.root / 'replication_receipts').glob('*.json')))
        atomic_json(a.out / 'summary.json', summary)
    if not summary['all_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
