"""Integrity, full-lifecycle IP accounting and paired exploratory intervals.

Analysis is post-acquisition. SELECT examples are not a confirmatory final test.
Loopback observations include TLS setup, controls, cover, ACKs and teardown.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from greenshaping.data.wire_capture import packets
from greenshaping.utils.evidence import SessionLedger, atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.input / 'SHA256.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((a.input / name).read_bytes()).hexdigest() != expected:
            raise ValueError('evidence hash mismatch: ' + name)
    plan = json.loads((a.input / 'frozen_plan.json').read_text())
    ledger = SessionLedger(a.input / 'ledger', plan)
    audit = ledger.audit()
    records = [json.loads(f.read_text())['record'] for f in sorted((a.input / 'ledger/committed').glob('*.json'))]
    flow_owner, ip_counts, packet_counts, protected, outside = {}, collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    per_site, unassigned, ip_total = collections.Counter(), [], 0
    ports = (55080, 55081, 55082)
    for packet in packets(a.input / 'loopback.pcap', ('127.0.0.1', '127.0.0.1'), fingerprints=True):
        sport, dport = packet['source_port'], packet['destination_port']
        if dport in ports:
            site, client_port, direction = dport, sport, 'up'
        elif sport in ports:
            site, client_port, direction = sport, dport, 'down'
        else:
            raise ValueError('capture contains an unexpected endpoint')
        key, epoch = (site, client_port), packet['epoch']
        if key not in flow_owner:
            owners = [r for r in records if r['start_epoch'] <= epoch <= r['end_epoch']]
            if len(owners) != 1:
                unassigned.append(packet)
                ip_total += packet['network_bytes']
                continue
            flow_owner[key] = owners[0]
        row = flow_owner[key]
        if epoch < row['start_epoch'] - .001 or epoch > row['end_epoch'] + 1.:
            raise ValueError('ambiguous port reuse or out-of-lifecycle packet')
        sid, size = row['trial_id'], packet['network_bytes']
        ip_counts[sid] += size
        packet_counts[sid] += 1
        per_site[(sid, ports.index(site), direction)] += size
        result = row.get('result', {})
        begin = result.get('start_epoch', float('inf'))
        end = begin + result.get('public_horizon_s', 0) + plan['contract']['interval']
        (protected if begin <= epoch <= end else outside)[sid] += size
        ip_total += size
    log = (a.input / 'tcpdump.log').read_text()
    captured = re.search(r'(\d+) packets captured', log)
    dropped = re.search(r'(\d+) packets dropped by kernel', log)
    reconciliation = {'pcap_packets': sum(packet_counts.values()) + len(unassigned),
                      'tcpdump_packets_captured': None if captured is None else int(captured[1]),
                      'kernel_drops': None if dropped is None else int(dropped[1]),
                      'captured_ip_bytes': ip_total, 'assigned_ip_bytes': sum(ip_counts.values()),
                      'unassigned_packets': len(unassigned), 'flows': len(flow_owner),
                      'physical_hosts': 1, 'interface': 'lo0',
                      'metric': 'captured loopback IPv4 lengths, not a physical NIC byte counter'}
    reconciliation['passed'] = (not unassigned and dropped is not None and int(dropped[1]) == 0
                                 and captured is not None and int(captured[1]) == reconciliation['pcap_packets']
                                 and len(flow_owner) == 3 * len(records))
    rows = []
    for r in records:
        result = r.get('result', {})
        all_logs = result.get('client_logs', []) + [v for site in r.get('site_logs', []) for v in site]
        row = {'trial_id': r['trial_id'], 'job_id': r['job']['job_id'], 'cohort': r['job']['cohort'],
               'stress': r['job'].get('stress', ''), 'policy': r['policy'],
               'actual_rounds': r['job']['actual_rounds'], 'family': r['job']['family'],
               'contract_success': r.get('observed_contract_success', False),
               'engineering_gate_passed': r['passed'], 'reference_match': r.get('reference_match', False),
               'fixed_frames_valid': r.get('fixed_frames_valid', False), 'timing_valid': r.get('timing_valid', False),
               'application_bytes': result.get('application_bytes'),
               'control_application_bytes': result.get('control_application_bytes'),
               'whole_lifecycle_ip_bytes': ip_counts[r['trial_id']], 'packets': packet_counts[r['trial_id']],
               'protected_window_ip_bytes': protected[r['trial_id']], 'outside_window_ip_bytes': outside[r['trial_id']],
               'tool_completion_s': result.get('tool_completion_s'),
               'session_wall_s': r['end_epoch'] - r['start_epoch'],
               'max_send_completion_lag_s': max([x['lag_s'] for x in all_logs], default=0.),
               'errors': '|'.join(result.get('errors', []) + r.get('site_errors', []) + [r.get('acquisition_error', '')])}
        rows.append(row)
    with (a.out / 'trial_metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    site_rows = [{'trial_id': sid, 'site': site, 'direction': d, 'ip_bytes': n}
                 for (sid, site, d), n in sorted(per_site.items())]
    with (a.out / 'per_logical_site_ip_bytes.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(site_rows[0]))
        w.writeheader()
        w.writerows(site_rows)
    policy_summary = {}
    policy_names = sorted({r['policy'] for r in rows})
    for policy in policy_names:
        normal = [r for r in rows if r['policy'] == policy and r['cohort'] != 'stress']
        stress = [r for r in rows if r['policy'] == policy and r['cohort'] == 'stress']
        n, s = len(normal), sum(r['contract_success'] for r in normal)
        ci = [0. if s == 0 else float(beta.ppf(.025, s, n - s + 1)),
              1. if s == n else float(beta.ppf(.975, s + 1, n - s))]
        completed = [r['tool_completion_s'] for r in normal if r['contract_success']]
        policy_summary[policy] = {'positive_n': n, 'contract_success': s, 'binomial_95_ci': ci,
                                  'stress_n': len(stress), 'stress_expected_failure_gate_passed': sum(r['engineering_gate_passed'] for r in stress),
                                  'positive_mean_ip_bytes': float(np.mean([r['whole_lifecycle_ip_bytes'] for r in normal])),
                                  'positive_tool_p50_s': None if not completed else float(np.median(completed)),
                                  'positive_tool_p95_s': None if not completed else float(np.quantile(completed, .95)),
                                  'all_trials_max_send_completion_lag_s': max(r['max_send_completion_lag_s'] for r in rows if r['policy'] == policy)}
        policy_summary[policy]['cohorts'] = {
            cohort: {'n': sum(r['cohort'] == cohort for r in normal),
                     'contract_success': sum(r['cohort'] == cohort and r['contract_success'] for r in normal)}
            for cohort in ('select_base', 'select_shifted')}
        policy_summary[policy]['stress_by_type'] = {
            kind: {'n': sum(r['stress'] == kind for r in stress),
                   'reported_with_fixed_cover': sum(r['stress'] == kind and r['engineering_gate_passed'] for r in stress)}
            for kind in sorted({r['stress'] for r in stress})}
    pairs = {}
    for row in rows:
        if row['cohort'] != 'stress':
            pairs.setdefault(row['job_id'], {})[row['policy']] = row
    if any(set(pair) != set(policy_names) for pair in pairs.values()):
        raise ValueError('incomplete paired acquisition')
    paired = list(pairs.values())
    a_bytes = np.array([v['envelope']['whole_lifecycle_ip_bytes'] for v in paired])
    rng = np.random.default_rng(2026102311)
    index = rng.integers(0, len(paired), (20000, len(paired)))
    comparisons = {}
    for baseline in policy_names:
        if baseline == 'envelope':
            continue
        b_bytes = np.array([v[baseline]['whole_lifecycle_ip_bytes'] for v in paired])
        byte_ratio_samples = a_bytes[index].mean(axis=1) / b_bytes[index].mean(axis=1)
        comparison = {'baseline': baseline, 'independent_paired_jobs': len(paired), 'bootstrap_seed': 2026102311,
                      'bootstrap_resamples': 20000, 'resampling_unit': 'paired new SELECT job',
                      'whole_lifecycle_ip_mean_ratio': float(a_bytes.mean() / b_bytes.mean()),
                      'whole_lifecycle_ip_mean_ratio_95_ci': np.quantile(byte_ratio_samples, [.025, .975]).tolist(),
                      'application_byte_ratio': paired[0]['envelope']['application_bytes'] / paired[0][baseline]['application_bytes'],
                      'both_policies_success_pairs': sum(v['envelope']['contract_success'] and v[baseline]['contract_success'] for v in paired),
                      'interval_scope': 'exploratory conditional SELECT workload; does not include contract selection uncertainty or host-to-host variation'}
        if all(v['envelope']['contract_success'] and v[baseline]['contract_success'] for v in paired):
            a_time = np.array([v['envelope']['tool_completion_s'] for v in paired])
            b_time = np.array([v[baseline]['tool_completion_s'] for v in paired])
            ratios = np.quantile(a_time[index], .95, axis=1) / np.quantile(b_time[index], .95, axis=1)
            comparison['tool_p95_ratio'] = float(np.quantile(a_time, .95) / np.quantile(b_time, .95))
            comparison['tool_p95_ratio_95_ci'] = np.quantile(ratios, [.025, .975]).tolist()
            comparison['paired_median_tool_extra_s'] = float(np.median(a_time - b_time))
        comparisons[baseline] = comparison
    summary = {'input_files_hash_verified': len(manifest), 'ledger': audit, 'capture_reconciliation': reconciliation,
               'contract': plan['contract'], 'policies': policy_summary, 'paired_comparisons': comparisons,
               'total_trials': len(rows), 'all_fixed_frames_valid': all(r['fixed_frames_valid'] for r in rows),
               'all_timing_valid': all(r['timing_valid'] for r in rows),
               'all_engineering_gates_passed': all(r['engineering_gate_passed'] for r in rows),
               'wire_privacy_certified': False, 'energy_measured': False, 'final_bank_opened': False,
               'physical_sites': 1, 'logical_loopback_sites': 3, 'cloud_sessions': 0,
               'cpu_load_before': plan['load_before'],
               'caveats': ['scripted feedback, generated tables, no LLM timing validation',
                           'one shared host under recorded contention, no physical per-site network',
                           'same public horizon; tool completion is an internal timestamp, public session continues',
                           'packet traces can still contain workload-dependent jitter; no 0-DP wire claim',
                           'schedule byte ratio is structural; observed IP overhead and failure handling are new acquisition evidence',
                           'task family and round count are coupled by this pilot design; no independent family/round effect is estimated',
                           'a small zero-failure sample cannot verify rho=0.001']}
    atomic_json(a.out / 'analysis.json', summary)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    plot_order = [v for v in policy_names if v != 'envelope'] + ['envelope']
    for i, policy in enumerate(plot_order):
        v = policy_summary[policy]
        color = '#217c91' if policy == 'envelope' else '#7d8da3'
        ax[0].bar(i, v['positive_mean_ip_bytes'] / 1024, color=color)
        ax[1].bar(i, v['positive_tool_p95_s'] or 0, color=color)
    for axis in ax:
        labels = [('Envelope' if v == 'envelope' else 'Periodic\n' + v.replace('periodic_', '') + ' ticks') for v in plot_order]
        axis.set_xticks(range(len(plot_order)), labels)
        axis.spines[['top', 'right']].set_visible(False)
    ax[0].set_ylabel('Mean captured IPv4 bytes (KiB)')
    ax[0].set_title('TLS lifecycle included')
    ax[1].set_ylabel('Internal tool-completion p95 (s)')
    ax[1].set_title('All policies keep the full public horizon')
    fig.suptitle('Mac Studio: 32 paired scripted SELECT jobs, 3 logical loopback sites', fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / 'local_tls_tradeoff.png', dpi=220)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
