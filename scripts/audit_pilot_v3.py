"""Audit captured payload accounting, all-link schedules and immutable question splits."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from greenshaping.data.adapters import SessionDirectoryAdapter
from greenshaping.utils.io import read_json, write_csv, write_json
from greenshaping.utils.provenance import file_sha256


def audit(root: Path, out: Path) -> dict:
    quality = read_json(root / 'pilot_quality.json')
    modes = list(quality['modes'])
    seq_file = root / 'all_seq.csv'
    if not seq_file.exists():
        command = [shutil.which('tshark') or '/opt/homebrew/bin/tshark', '-r', str(root / 'traffic.pcap'),
                   '-Y', 'tcp.len>0', '-T', 'fields', '-E', 'separator=,']
        for field in ('frame.time_epoch', 'tcp.srcport', 'tcp.dstport', 'tcp.seq_raw', 'tcp.len', 'tcp.stream'):
            command += ['-e', field]
        with seq_file.open('w') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.DEVNULL, check=True)
    with seq_file.open() as f:
        sequence_rows = [r for r in csv.reader(f) if r]
    sequence_rows.sort(key=lambda r: float(r[0]))
    times = [float(r[0]) for r in sequence_rows]

    def unique_bytes(intervals: list[tuple[int, int]]) -> int:
        total, end = 0, -1
        for a, b in sorted(intervals):
            total += max(0, b - max(a, end))
            end = max(end, b)
        return total

    accounting, schedules = [], []
    balances = {}
    for mode in modes:
        ad = SessionDirectoryAdapter(root / mode, gap_s=.005)
        sessions = ad.load()
        partition_balance = {}
        for partition in ('train', 'select', 'test'):
            subset = [s for s in sessions if s.meta.get('partition') == partition]
            partition_balance[partition] = {
                'sessions': len(subset), 'questions': len({s.question_id for s in subset}),
                'participation_per_site': np.mean([s.participation for s in subset], axis=0).tolist() if subset else [],
                'size_counts_per_site': [dict(Counter(s.size_bucket[i] for s in subset if s.participation[i]))
                                         for i in range(5)]}
        balances[mode] = partition_balance
        for session in sessions:
            manifest = read_json(root / mode / 'manifests' / f'{session.sid}.json')
            stats = read_json(root / mode / 'transport' / f'{session.sid}.json')
            ports = {int(s['port']): i for i, s in enumerate(manifest['sites'])}
            wire = Counter()
            intervals: dict[tuple[int, str, str], list[tuple[int, int]]] = {}
            left = bisect.bisect_left(times, manifest['start_time'])
            right = bisect.bisect_right(times, manifest['end_time'])
            for row in sequence_rows[left:right]:
                src, dst, seq, length, stream = map(int, row[1:])
                key = ((ports[dst], 'upstream', str(stream)) if dst in ports else
                       (ports[src], 'downstream', str(stream)) if src in ports else None)
                if key is not None:
                    end = seq + length
                    intervals.setdefault(key, []).append((seq, min(end, 2 ** 32)))
                    if end > 2 ** 32:
                        intervals[key].append((0, end - 2 ** 32))
            with (root / mode / 'captures' / f'{session.sid}.csv').open() as f:
                for row in csv.reader(f):
                    src, dst, count = int(row[3]), int(row[4]), int(row[5])
                    if dst in ports:
                        wire[(ports[dst], 'upstream')] += count
                    elif src in ports:
                        wire[(ports[src], 'downstream')] += count
            for side in ('upstream', 'downstream'):
                for site, d in enumerate(stats[side]):
                    ticks = [x for x in d['ticks'] if 'scheduled_epoch' in x]
                    if not ticks:
                        continue
                    app = sum(x['application_bytes'] for x in ticks)
                    expected_wire = sum(x['application_bytes'] + 22 * math.ceil(x['application_bytes'] / 16384)
                                        for x in ticks)
                    unique = sum(unique_bytes(v) for (i, direction, _), v in intervals.items()
                                 if i == site and direction == side)
                    accounting.append({'mode': mode, 'session': session.sid, 'question': session.question_id,
                                       'site': site, 'direction': side, 'application_bytes': app,
                                       'observed_tcp_payload_bytes': wire[(site, side)],
                                       'tls13_expected_payload_bytes': expected_wire,
                                       'wire_difference_bytes': wire[(site, side)] - expected_wire,
                                       'unique_tcp_payload_bytes': unique,
                                       'duplicate_tcp_payload_bytes': wire[(site, side)] - unique,
                                       'unique_wire_difference_bytes': unique - expected_wire,
                                       'overflow_bytes': d['overflow_bytes'], 'cap_exceeded': d['cap_exceeded'],
                                       'io_error': d['error']})
                    for tick in ticks:
                        schedules.append({'mode': mode, 'session': session.sid, 'site': site, 'direction': side,
                                          'tick': tick['tick'], 'lag_s': tick['lag_s'], 'application_bytes': tick['application_bytes']})
    out.mkdir(parents=True, exist_ok=False)
    if accounting:
        write_csv(accounting, out / 'wire_accounting.csv')
        write_csv(schedules, out / 'ticks.csv')
    lag = np.asarray([r['lag_s'] for r in schedules])
    report = {'data_kind': 'rehearsal', 'privacy_certified': False, 'energy_measured': False,
              'quality_sha256': file_sha256(root / 'pilot_quality.json'), 'balance': balances,
              'wire_link_directions': len(accounting),
              'wire_exact_accounting_fraction': float(np.mean([r['wire_difference_bytes'] == 0 for r in accounting]))
              if accounting else None,
              'wire_max_absolute_difference_bytes': max((abs(r['wire_difference_bytes']) for r in accounting), default=0),
              'unique_wire_exact_fraction': float(np.mean([r['unique_wire_difference_bytes'] == 0 for r in accounting]))
              if accounting else None,
              'unique_wire_max_difference_bytes': max((abs(r['unique_wire_difference_bytes']) for r in accounting), default=0),
              'duplicate_tcp_payload_bytes': sum(r['duplicate_tcp_payload_bytes'] for r in accounting),
              'overflow_link_directions': sum(r['overflow_bytes'] > 0 for r in accounting),
              'cap_violations': sum(r['cap_exceeded'] for r in accounting),
              'io_errors': sum(bool(r['io_error']) for r in accounting),
              'tick_count': len(schedules),
              'pre_send_tick_lag_median_s': float(np.median(lag)) if len(lag) else None,
              'pre_send_tick_lag_p99_s': float(np.percentile(lag, 99)) if len(lag) else None,
              'pre_send_tick_lag_max_s': float(np.max(lag)) if len(lag) else None,
              'assumptions': ['TLS 1.3 AEAD overhead 22 bytes/record, maximum 16384 application bytes/record',
                              'TCP payload accounting includes retransmissions; nonzero differences require inspection',
                              'capture windows exclude fixed public start controls and all planner/summary requests',
                              'exact byte accounting alone does not certify timing-channel privacy']}
    write_json(report, out / 'report.json')
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    report = audit(Path(args.root), Path(args.out))
    print(json.dumps({k: v for k, v in report.items() if k not in ('balance', 'assumptions')}, indent=2))


if __name__ == '__main__':
    main()
