"""Verify public hashes and rederive accounting without any cloud access.

Oracle labels delimit accounting windows, never attacker features. Published
packet metadata has no payload/IP addresses and a shifted clock per phase.
"""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', newline='') as stream:
        yield from csv.DictReader(stream)


def main():
    manifest = json.loads((ROOT / 'DATA_SHA256.json').read_text())
    for name, value in manifest.items():
        assert digest(ROOT / name) == value['sha256'], name
    calibration = json.loads(gzip.decompress((ROOT / 'outputs/public_mac_calibration/calibration.json.gz').read_bytes()))
    calibration_counts = {}
    for phase, count, passed in [('train', 360, 349), ('select', 216, 215)]:
        rows = calibration['phases'][phase]['sessions']
        assert len(rows) == count and sum(row['passed'] for row in rows) == passed
        calibration_counts[phase] = {'n': count, 'passed': passed}
    cloud = list(read_csv(ROOT / 'outputs/public_ec2/sessions.csv'))
    assert len(cloud) == 168 and all(row['passed'] == 'True' for row in cloud)
    metrics = {v['mode']: v for v in json.loads((ROOT / 'outputs/paper_completion_20261022/network_metrics.json').read_text())}
    cloud_result, cloud_packets = {}, 0
    for mode, metric in metrics.items():
        sessions = [row for row in cloud if row['mode'] == mode]
        for observer in ('T2', 'T3'):
            inside, total = collections.Counter(), 0
            for packet in read_csv(ROOT / ('outputs/public_ec2/packets_' + observer + '_' + mode + '.csv.gz')):
                total += int(packet['network_bytes'])
                cloud_packets += 1
                if packet['oracle_session_id']:
                    inside[packet['oracle_session_id']] += int(packet['network_bytes'])
            assert len(sessions) == 24
            for row in sessions:
                assert inside[row['session_id']] == int(row[observer + '_IP_bytes'])
            assert abs((total - sum(inside.values())) / 1e6 - metric[observer + '_outside_MB']) < 1e-8
        mean_mb = np.mean([int(row['T2_IP_bytes']) for row in sessions]) / 1e6
        assert abs(mean_mb - metric['mean_IP_MB']) < 1e-10
        cloud_result[mode] = {'sessions': 24, 'mean_T2_IP_MB': mean_mb}
    model_checks = 0
    for path in sorted((ROOT / 'outputs/public_model_checks').glob('*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == 12 and all(row['passed'] for row in rows)
        model_checks += len(rows)
    assert model_checks == 48
    local_result = {}
    for phase, positives, passed, stress in [('initial', 64, 63, 20), ('repaired', 64, 64, 20), ('frequency', 96, 96, 30)]:
        folder = ROOT / 'outputs/public_local_tls' / phase
        with gzip.open(folder / 'sessions.jsonl.gz', 'rt') as stream:
            sessions = [json.loads(line) for line in stream]
        normal = [row for row in sessions if row['job']['cohort'] != 'stress']
        invalid = [row for row in sessions if row['job']['cohort'] == 'stress']
        assert len(normal) == positives and sum(row['observed_contract_success'] for row in normal) == passed
        assert len(invalid) == stress and all(row['passed'] and row['fixed_frames_valid'] for row in invalid)
        counted, packet_count = collections.Counter(), 0
        for packet in read_csv(folder / 'packets.csv.gz'):
            counted[packet['oracle_trial_id']] += int(packet['network_bytes'])
            packet_count += 1
        derived = list(read_csv(ROOT / 'outputs/local_contract_tls_20261023' / (phase + '_analysis') / 'trial_metrics.csv'))
        assert set(counted) == {row['trial_id'] for row in derived}
        for row in derived:
            assert counted[row['trial_id']] == int(row['whole_lifecycle_ip_bytes'])
        local_result[phase] = {'positive': positives, 'positive_passed': passed, 'stress_reported': stress, 'packets': packet_count}
    final = list(read_csv(ROOT / 'outputs/local_contract_tls_20261023/frequency_analysis/trial_metrics.csv'))
    pairs = {}
    for row in final:
        if row['cohort'] != 'stress':
            pairs.setdefault(row['job_id'], {})[row['policy']] = row
    assert len(pairs) == 32
    paired = list(pairs.values())
    observed = json.loads((ROOT / 'outputs/local_contract_tls_20261023/frequency_analysis/analysis.json').read_text())
    a = np.array([float(v['envelope']['whole_lifecycle_ip_bytes']) for v in paired])
    at = np.array([float(v['envelope']['tool_completion_s']) for v in paired])
    index = np.random.default_rng(2026102311).integers(0, len(paired), (20000, len(paired)))
    comparisons = {}
    for baseline in ('periodic_6', 'periodic_10'):
        b = np.array([float(v[baseline]['whole_lifecycle_ip_bytes']) for v in paired])
        bt = np.array([float(v[baseline]['tool_completion_s']) for v in paired])
        ratio, tratio = a.mean() / b.mean(), np.quantile(at, .95) / np.quantile(bt, .95)
        ci = np.quantile(a[index].mean(axis=1) / b[index].mean(axis=1), [.025, .975])
        reference = observed['paired_comparisons'][baseline]
        assert abs(ratio - reference['whole_lifecycle_ip_mean_ratio']) < 1e-12
        assert abs(tratio - reference['tool_p95_ratio']) < 1e-12
        assert np.allclose(ci, reference['whole_lifecycle_ip_mean_ratio_95_ci'], atol=1e-12, rtol=0)
        comparisons[baseline] = {'byte_saving_percent': 100 * (1 - ratio), 'internal_tool_p95_ratio': tratio}
    print(json.dumps({'data_files_sha256_verified': len(manifest), 'mac_calibration': calibration_counts,
                      'cloud': cloud_result, 'cloud_packets_reconciled': cloud_packets,
                      'model_checks': model_checks, 'local_tls_phases': local_result,
                      'final_paired_comparisons': comparisons,
                      'new_experiment_or_final_test_execution': False}, indent=2))


if __name__ == '__main__':
    main()
