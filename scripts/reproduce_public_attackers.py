"""Recompute metadata features and the frozen retrospective RF diagnostic.

This is replication on the published acquired TRAIN subset, not a new privacy
experiment or a final-test evaluation. Oracle window labels are never features.
"""
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / 'outputs/public_ec2/sessions.csv').open() as stream:
        sessions = list(csv.DictReader(stream))
    feature_rows = json.loads((ROOT / 'outputs/evaluation_extension_20261022/features.json').read_text())
    with (ROOT / 'outputs/evaluation_extension_20261022/attacker_scores.csv').open() as stream:
        expected = list(csv.DictReader(stream))
    verified_features = verified_scores = 0
    for mode in sorted({v['mode'] for v in sessions}):
        ss = {v['session_id']: v for v in sessions if v['mode'] == mode}
        observations = {sid: [] for sid in ss}
        with gzip.open(ROOT / ('outputs/public_ec2/packets_T2_' + mode + '.csv.gz'), 'rt') as stream:
            for packet in csv.DictReader(stream):
                sid = packet['oracle_session_id']
                if sid:
                    observations[sid].append((float(packet['time_s']) - float(ss[sid]['start_s']), int(packet['network_bytes']), packet['direction']))
        features = [v for v in feature_rows if v['mode'] == mode]
        for row in features:
            sid = str(row['question_id']) + '_' + str(row['repetition'])
            obs = observations[sid]
            times = np.array([v[0] for v in obs]); sizes = np.array([v[1] for v in obs]); gaps = np.diff(times)
            bins = np.histogram(times, bins=np.arange(23) * .5, weights=sizes)[0]
            values = [sizes.sum(), len(sizes), sum(v[1] for v in obs if v[2] == 'up'), sum(v[1] for v in obs if v[2] == 'down')]
            values += [sizes.mean(), sizes.std(), *np.quantile(sizes, [.1, .5, .9]), times[-1] - times[0], gaps.mean(), gaps.std(), np.quantile(gaps, .9)] + list(bins)
            assert np.allclose(values, row['features'], rtol=1e-12, atol=1e-10), (mode, sid)
            verified_features += 1
        X = np.log1p([v['features'] for v in features])
        for target, group in [('family', 'requested_size'), ('requested_size', 'family')]:
            labels = np.array([v[target] for v in features]); groups = np.array([v[group] for v in features])
            for ablation in ['bytes_only', 'all_metadata']:
                x = X[:, :1] if ablation == 'bytes_only' else X
                for classifier in ['majority', 'rf']:
                    predicted = np.empty_like(labels)
                    for held in sorted(set(groups)):
                        test = groups == held
                        assert not ({v['question_id'] for v, t in zip(features, test) if t} & {v['question_id'] for v, t in zip(features, test) if not t})
                        model = DummyClassifier(strategy='most_frequent') if classifier == 'majority' else RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, class_weight='balanced', random_state=20261023, n_jobs=2)
                        model.fit(x[~test], labels[~test]); predicted[test] = model.predict(x[test])
                    reference = next(v for v in expected if (v['mode'], v['target'], v['ablation'], v['classifier']) == (mode, target, ablation, classifier))
                    assert abs(balanced_accuracy_score(labels, predicted) - float(reference['balanced_accuracy'])) < 1e-12
                    assert abs(accuracy_score(labels, predicted) - float(reference['accuracy'])) < 1e-12
                    verified_scores += 1
    print(json.dumps({'metadata_feature_rows_rederived': verified_features, 'classifier_score_rows_reproduced': verified_scores, 'final_test_opened': False, 'scope': 'Retrospective grouped acquired-TRAIN diagnostic replication; no empirical DP certificate'}, indent=2))


if __name__ == '__main__':
    main()
