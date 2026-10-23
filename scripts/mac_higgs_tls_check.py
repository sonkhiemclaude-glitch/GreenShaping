"""Bounded HIGGS TRAIN TLS integration on loopback; no wire-privacy/energy claim."""
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from greenshaping.shaping.pipeline import DPParams, PaddingParams, RoundsParams, ShapingConfig
from greenshaping.testbed.higgs_gateway import RoutedTool, routed_calls
from greenshaping.testbed.higgs_workload import site_table
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.question_bank import load_question_bank
from greenshaping.testbed.transport import ShapedTransport, TLSToolServer


def equivalent(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            equivalent(a, b) for a, b in zip(actual, expected, strict=True))
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-10))
    return actual == expected


def modes(n):
    epsilon_session, volume, delta = 8., 8192, 16384
    dp = DPParams(eps_dir=epsilon_session / (2 * n), interval=.5, horizon=6., mode='arrivals',
                  delta=delta, volume_cap_bytes=volume, bias=8 * delta / (epsilon_session / (2 * n)))
    rounds = RoundsParams(window=.5, cover_prob=1., reply_frame=1024)
    return {'none': ShapingConfig(), 'pad': ShapingConfig(padding=PaddingParams()),
            'rounds': ShapingConfig(rounds=rounds), 'dp': ShapingConfig(dp=dp),
            'combo': ShapingConfig(padding=PaddingParams(), rounds=rounds, dp=dp)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--banks', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-sites', type=int, choices=[5, 10, 20], nargs='+', default=[5, 10, 20])
    p.add_argument('--modes', choices=['none', 'pad', 'rounds', 'dp', 'combo'], nargs='+',
                   default=['none', 'pad', 'rounds', 'dp', 'combo'])
    p.add_argument('--server-layout', choices=['separate', 'routed'], default='separate')
    p.add_argument('--repetitions', type=int, choices=[1, 2], default=2)
    p.add_argument('--max-seconds', type=int, default=5400)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + a.max_seconds
    cert, key = a.out / 'server.cert', a.out / 'server.key'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2',
                    '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
    key.chmod(0o600)
    checks = []
    summary = {'scope': 'HIGGS TRAIN one-round reference tools over loopback TLS; not distributed captures, LLM or energy',
               'server_layout': a.server_layout, 'expected_sessions': len(a.n_sites) * len(a.modes) * 12 * a.repetitions,
               'epsilon_session_dp': 8, 'physical_hosts': 1,
               'final_test_executed': False, 'privacy_certified': False, 'large_run_ready': False}
    started = time.time()
    try:
        for n in a.n_sites:
            table_started = time.monotonic()
            tables = [site_table(a.store, n, i) for i in range(n)]
            chosen = {}
            for q in load_question_bank(a.banks / f'n{n}/train.jsonl', require_min=False):
                if q.metadata.get('partition') != 'train':
                    raise ValueError('only train input allowed')
                chosen.setdefault((q.family, q.metadata['requested_size']), q)
            if len(chosen) != 12:
                raise ValueError('missing family/size diagnostic stratum')
            reference = {q.question_id: [{'result': result, 'matching_rows': count}
                         for c in q.plan[0] for result, count in [run_query(tables[c['site']], q.family, c['params'])]]
                         for q in chosen.values()}
            servers = []
            try:
                gateway = RoutedTool(set(range(n)), lambda i, tables=tables: tables[i])
                gateway.tables = dict(enumerate(tables))
                for i in range(n if a.server_layout == 'separate' else 1):
                    servers.append(TLSToolServer('127.0.0.1', 0, str(cert), str(key),
                                                 gateway if a.server_layout == 'routed' else
                                                 lambda f, p, df=tables[i]: run_query(df, f, p)))
                for name, cfg in modes(n).items():
                    if name not in a.modes:
                        continue
                    addresses = [s.address for s in servers] if a.server_layout == 'separate' else [servers[0].address] * n
                    tr = ShapedTransport(addresses, str(cert), cfg, start_lead_s=1.)
                    try:
                        for repetition in range(a.repetitions):
                            for q in chosen.values():
                                if time.monotonic() > deadline:
                                    raise TimeoutError('Mac TLS dispatch deadline reached')
                                row = {'N': n, 'mode': name, 'repetition': repetition, 'question_id': q.question_id,
                                       'family': q.family, 'size': q.metadata['requested_size'], 'start_epoch': time.time(),
                                       'reference_match': False, 'measurement_valid': False}
                                began = False
                                try:
                                    tr.begin(called_sites=[c['site'] for c in q.plan[0]])
                                    began = True
                                    calls = routed_calls(q.plan[0]) if a.server_layout == 'routed' else q.plan[0]
                                    response = tr.call_round(q.family, calls)
                                    row['tool_completed_epoch'] = time.time()
                                    row['responses'] = [{k: v[k] for k in ('result', 'matching_rows')} for v in response]
                                    row['reference_match'] = equivalent(row['responses'], reference[q.question_id])
                                except Exception as exc:
                                    row['error'] = repr(exc)
                                finally:
                                    if began:
                                        try:
                                            row['transport'] = tr.finish()
                                            row['measurement_valid'] = bool(row['transport']['measurement_valid'])
                                        except Exception as exc:
                                            row['finish_error'] = repr(exc)
                                    row['end_epoch'] = time.time()
                                    row['passed'] = row['reference_match'] and row['measurement_valid'] and 'error' not in row
                                    path = a.out / f'session_{len(checks):04d}.json'
                                    path.write_text(json.dumps(row, indent=2))
                                    checks.append({k: v for k, v in row.items() if k not in ('transport', 'responses')})
                                    (a.out / 'progress.json').write_text(json.dumps({'sessions': len(checks),
                                         'passed': sum(c['passed'] for c in checks), 'N': n, 'mode': name,
                                         'elapsed_s': time.time() - started}, indent=2))
                                    print(json.dumps(checks[-1]), flush=True)
                                if not began:
                                    raise RuntimeError('start barrier failed; retained session; connection requires review')
                    finally:
                        tr.close()
                summary.setdefault('site_table_and_group_seconds', {})[str(n)] = time.monotonic() - table_started
            finally:
                for server in servers:
                    server.stop()
            del tables, reference, servers, gateway
            gc.collect()
        summary['all_passed'] = len(checks) == summary['expected_sessions'] and all(c['passed'] for c in checks)
    except Exception as exc:
        summary.update(all_passed=False, error=repr(exc))
        raise
    finally:
        summary.update(sessions=len(checks), passed=sum(c['passed'] for c in checks), elapsed_s=time.time() - started,
                       checks=checks, configurations={str(n): {k: v.to_dict() for k, v in modes(n).items()} for n in (5, 10, 20)})
        durations = [c['end_epoch'] - c['start_epoch'] for c in checks]
        if durations:
            summary['observed_session_seconds'] = {'mean': float(np.mean(durations)), 'max': max(durations)}
        for n in a.n_sites:
            cfg = modes(n)['dp'].dp
            assert cfg
            # Arrivals-mode expected positive release with no real arrivals (not Joules).
            scale = cfg.delta / cfg.eps_dir
            expected = cfg.bias + .5 * scale * math.exp(-cfg.bias / scale)
            summary.setdefault('analytical_dummy_bytes_per_session', {})[str(n)] = 2 * n * round(cfg.horizon / cfg.interval) * expected
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    if not summary['all_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
