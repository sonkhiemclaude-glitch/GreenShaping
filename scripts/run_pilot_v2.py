"""Balanced synthetic pilot: online LangGraph and persistent TLS; no measured energy."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from greenshaping.data.adapters import SessionDirectoryAdapter, export_sessions, split_capture_by_sessions
from greenshaping.data.schemas import SessionManifest, SiteSpec
from greenshaping.shaping.pipeline import DPParams, PaddingParams, RoundsParams, ShapingConfig
from greenshaping.testbed.capture_commands import tshark_export_command
from greenshaping.testbed.orchestrator import LangGraphOrchestrator
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.transport import ShapedTransport, TLSToolServer
from greenshaping.testbed.workload_v2 import build_workload
from greenshaping.utils.io import write_json
from greenshaping.utils.provenance import file_sha256


def wait_for(path: Path, timeout: float = 180.) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(.2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--model', default='mistral-small3.1:24b')
    p.add_argument('--base-port', type=int, default=18500)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--include-rounds', action='store_true')
    p.add_argument('--adaptive', action='store_true', help='fresh v4 private-data, multi-round workload')
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'pilot_status.json').exists():
        raise FileExistsError('use a fresh pilot directory')
    status: dict[str, Any] = {'pid': os.getpid(), 'state': 'running', 'data_kind': 'rehearsal',
                              'physical_hosts': 1, 'N': 5, 'energy_measured': False}

    def update(phase: str, **kw: Any) -> None:
        status.update(phase=phase, **kw)
        write_json(status, out / 'pilot_status.json')
        print(json.dumps({'phase': phase, **kw}), flush=True)

    servers: list[TLSToolServer] = []
    transports: list[ShapedTransport] = []
    try:
        update('prepare')
        builder = build_workload
        if args.adaptive:
            from greenshaping.testbed.workload_v4 import build_workload as adaptive_builder
            builder = adaptive_builder
        questions = builder(out / 'workload', args.seed)
        cert, key = out / 'cert.pem', out / 'key.pem'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2',
                        '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                        '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
        key.chmod(0o600)
        tables = []
        for i in range(5):
            df = pd.read_csv(out / 'workload' / f'site_{i}.csv')
            tables.append(df)

            def tool(f: str, params: dict[str, Any], table: pd.DataFrame = df, site: int = i) -> tuple[dict[str, Any], int]:
                result, rows = run_query(table, f, params)
                if args.adaptive:
                    # Controlled heterogeneous service times; not measured edge hardware.
                    time.sleep((.003, .015, .035, .07, .12)[site] + min(.08, rows / 1_000_000))
                return result, rows

            servers.append(TLSToolServer('127.0.0.1', args.base_port + i, str(cert), str(key), tool))
        sites = [s.address for s in servers]
        modes = {'none': ShapingConfig(), 'pad': ShapingConfig(padding=PaddingParams()),
                 'dp': ShapingConfig(dp=DPParams(eps_dir=4, interval=.05, horizon=1., mode='arrivals',
                                                 delta=8192, volume_cap_bytes=4096, bias=16000))}
        if args.include_rounds:
            rounds = RoundsParams(window=.1, cover_prob=1., reply_frame=1024)
            modes['rounds'] = ShapingConfig(rounds=rounds)
            modes['combo'] = ShapingConfig(rounds=rounds, padding=PaddingParams(), dp=modes['dp'].dp)
        if args.adaptive:
            if not args.include_rounds:
                raise ValueError('adaptive pilot requires all five comparison modes')
            # Conservative cap and horizon are fixed before acquisition, including final holdout.
            dp = DPParams(eps_dir=4, interval=.05, horizon=4., mode='arrivals',
                          delta=65536, volume_cap_bytes=32768, bias=100000)
            rounds = RoundsParams(window=.05, cover_prob=1., reply_frame=2048)
            modes.update(dp=ShapingConfig(dp=dp), rounds=ShapingConfig(rounds=rounds),
                         combo=ShapingConfig(rounds=rounds, padding=PaddingParams(), dp=dp))
        if args.smoke:
            # One question/family/mode is integration evidence, never attacker evaluation.
            selected = ({100_000 + f * 48 + j for f in range(4) for j in (0, 4, 8)} if args.adaptive else
                        {q.question_id for q in questions if q.question_id % 48 == 0})
            order = [(mode, q) for mode in modes for q in questions if q.question_id in selected]
        else:
            # Entire immutable bank in every mode; identical questions across defences.
            order = [(mode, q) for mode in modes for q in questions]
        np.random.default_rng(args.seed).shuffle(order)
        orchs = {}
        for mode, cfg in modes.items():
            root = out / mode
            (root / 'manifests').mkdir(parents=True)
            tr = ShapedTransport(sites, str(cert), cfg)
            transports.append(tr)
            orchs[mode] = LangGraphOrchestrator(args.model, tr, str(root / 'agent'), seed=args.seed, summarize=args.summarize)
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=15) as response:
            model_info = next(m for m in json.loads(response.read())['models'] if m['name'] == args.model)
        with urllib.request.urlopen('http://127.0.0.1:11434/api/version', timeout=15) as response:
            server_info = json.loads(response.read())
        write_json({'model': model_info, 'serving': server_info, 'script_sha256': file_sha256(__file__),
                    'tls_versions': sorted({c.tls_version for tr in transports for c in tr.channels}),
                    'tcp_nodelay': True, 'start_lead_s': .5, 'dp_session_epsilon_bound': 2 * 5 * 4,
                    'summary_enabled': args.summarize}, out / 'acquisition_protocol.json')
        write_json({k: asdict(v) for k, v in modes.items()}, out / 'transport_configs.json')
        update('awaiting_capture', expected_sessions=len(order), mode_counts={m: sum(x == m for x, _ in order) for m in modes})
        wait_for(out / 'capture.ready')
        failures = []
        for sid, (mode, q) in enumerate(order):
            update('collecting', session=sid, expected_sessions=len(order), mode=mode, question=q.question_id)
            orch = orchs[mode]
            before_channels = [len(s.channels) for s in servers]
            try:
                rows = orch.run(q, sites)
                expected = {c['site']: run_query(tables[c['site']], q.family, c['params'])[1]
                            for c in q.plan[0]}
                if args.adaptive:
                    from greenshaping.testbed.workload_v4 import next_calls
                    replay = q.plan[0]
                    replay_trace = []
                    while replay:
                        responses = []
                        for call in replay:
                            result, nrows = run_query(tables[call['site']], q.family, call['params'])
                            responses.append({'result': result, 'matching_rows': nrows})
                        replay_trace.append({'calls': replay, 'responses': responses})
                        replay = next_calls(q, replay, responses, len(replay_trace))
                    actual = [{'calls': r['calls'], 'responses': [
                        {'result': v['result'], 'matching_rows': v['matching_rows']} for v in r['responses']]}
                              for r in orch.last_rounds]
                    if actual != replay_trace:
                        raise RuntimeError('adaptive results differ from independent local replay')
                if rows != expected:
                    raise RuntimeError(f'query correctness mismatch: {rows} vs {expected}')
                if not orch.last_window:
                    raise RuntimeError('missing public transport window')
                start, end = orch.last_window
                if modes[mode].dp:
                    end += .1  # public receive guard; includes measured scheduler lateness at the final tick.
                meta = q.metadata | {'acquisition_mode': 'online LangGraph; analyst-supplied typed arguments',
                                     'transport_mode': mode, 'config_id': modes[mode].config_id(), 'repetition': 0,
                                     'summary_enabled': args.summarize, 'query_correct': True,
                                     'round_count': len(orch.last_rounds)}
                manifest = SessionManifest(sid, q.question_id, q.family, start,
                                           tuple(SiteSpec(h, port, int(i in rows), rows.get(i), 'mac_studio_loopback')
                                                 for i, (h, port) in enumerate(sites)),
                                           orch.llm, 'rehearsal', end, meta)
                write_json(manifest.to_dict(), out / mode / 'manifests' / f'{sid}.json')
                # Server-side diagnostics stay local; they are not sent on the shaped link.
                server_stats = []
                for server in servers:
                    ch = server.channels[list(modes).index(mode)]
                    server_stats.append({'volume_bytes': ch.volume, 'overflow_bytes': ch.pending,
                                         'cap_exceeded': ch.cap_exceeded, 'error': ch.error, 'ticks': ch.logs})
                write_json({'upstream': orch.last_transport['upstream'], 'downstream': server_stats,
                            'channels_before': before_channels, 'privacy_certified': False},
                           out / mode / 'transport' / f'{sid}.json')
                if not orch.last_transport.get('measurement_valid', False):
                    raise RuntimeError('Transport timing/completion/volume diagnostic failure; capture retained for investigation')
            except Exception as e:
                failure = {'session': sid, 'question': q.question_id, 'mode': mode, 'error': str(e),
                           'traceback': traceback.format_exc(),
                           'client_errors': [c.error for c in orch.transport.channels],
                           'server_errors': [[c.error for c in s.channels] for s in servers]}
                failures.append(failure)
                write_json(failures, out / 'failures.json')
                raise RuntimeError('integration failure; inspect preserved evidence before another acquisition') from e
        update('awaiting_capture_stop', collected_sessions=len(order) - len(failures), failures=len(failures))
        wait_for(out / 'capture.stopped')
        command = tshark_export_command(str(out / 'traffic.pcap'))
        command[0] = shutil.which('tshark') or '/opt/homebrew/bin/tshark'
        with (out / 'all.csv').open('w') as f, (out / 'tshark.log').open('w') as log:
            subprocess.run(command, stdout=f, stderr=log, check=True)
        quality = {}
        for mode in modes:
            root = out / mode
            counts = split_capture_by_sessions(out / 'all.csv', root / 'manifests', root / 'captures')
            sessions = SessionDirectoryAdapter(root, gap_s=.005).load()
            if any(not s.messages for s in sessions):
                raise RuntimeError('empty observed session')
            export_sessions(sessions, root / 'sessions.jsonl')
            quality[mode] = {'sessions': len(sessions), 'capture_rows': sum(counts.values()),
                             'partitions': {p: sum(s.meta['partition'] == p for s in sessions) for p in ('train', 'select', 'test')},
                             'clipped_size_buckets': sum(s.meta['clipped_size_buckets'] for s in sessions)}
        write_json({'modes': quality, 'failures': failures, 'pcap_sha256': file_sha256(out / 'traffic.pcap'),
                    'data_kind': 'rehearsal', 'energy_measured': False, 'privacy_certified': False,
                    'summary_enabled': args.summarize, 'smoke': args.smoke,
                    'limitations': ['single physical host, loopback and synthetic tables',
                                    'planner and summary traffic excluded from execution windows',
                                    'fixed public start controls outside execution windows',
                                    'rounds/compositions require separate wire validation',
                                    'one acquisition per question/config; seeds are computational repeats']},
                   out / 'pilot_quality.json')
        update('complete', state='complete')
        return 0
    except Exception as e:
        update('failed', state='failed', error=str(e))
        raise
    finally:
        for tr in transports:
            tr.close()
        for server in servers:
            server.stop()


if __name__ == '__main__':
    raise SystemExit(main())
