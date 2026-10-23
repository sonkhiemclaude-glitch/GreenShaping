"""Lock attacker choices using train/select only, before any final-test scoring."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_pilot import labels_and_features
from evaluate_pilot_v2 import CNN, MODELS, SECRETS, score

from greenshaping.attack.classifiers import RandomForestAttacker, TimeSliceCNN
from greenshaping.attack.evaluation import assert_disjoint, split_by_question
from greenshaping.attack.features import build_linkaware_tables, build_tables, build_time_slice_tables
from greenshaping.data.adapters import SessionDirectoryAdapter
from greenshaping.utils.io import write_csv, write_json
from greenshaping.utils.provenance import file_sha256


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=20261022)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    candidates = [dict(CNN), dict(CNN, temporal_bins=4, log_input=True)]
    receipt: dict[str, Any] = {'state': 'running', 'seed': a.seed, 'choices': {}, 'test_metrics_computed': False,
                               'pcap_sha256': file_sha256(root / 'traffic.pcap'),
                               'candidates': candidates, 'selection_rule': 'maximum select balanced accuracy; first candidate wins ties',
                               'script_sha256': file_sha256(__file__)}
    rows = []
    for mode in ('none', 'pad', 'dp', 'rounds', 'combo'):
        sessions = sorted(SessionDirectoryAdapter(root / mode, gap_s=.005).load(), key=lambda s: s.question_id)
        train, select, test = split_by_question(sessions, seed=a.seed)
        assert_disjoint(train, select, test)
        if len(train) != 96 or len(select) != 48 or len(test) != 48:
            raise ValueError('requires the locked 96/48/48 question design')
        receipt['choices'][mode] = {}
        # Held-out test sessions are never passed to any feature builder or model.
        for model in MODELS:
            builder = (lambda ss: build_time_slice_tables(ss, .05, 120)) if model == 'cnn' else (
                build_tables if model == 'rf' else lambda ss, model=model: build_linkaware_tables(ss, identity_only=model == 'identity'))
            trtables, setables = builder(train), builder(select)
            for secret in SECRETS:
                xtr, ytr, gtr = labels_and_features(trtables, secret)
                xse, yse, gse = labels_and_features(setables, secret)
                site = np.asarray([i for s in select for i in range(s.n_sites)]) if secret != 'query_family' else None
                if secret == 'size_bucket':
                    assert site is not None
                    site = site[setables[0]['part'] == 1]
                best = None
                for candidate, params in enumerate(candidates if model == 'cnn' else [{}]):
                    clf = (TimeSliceCNN(seed=a.seed, **params) if model == 'cnn' else
                           RandomForestAttacker(n_estimators=200, seed=a.seed, n_jobs=4))
                    clf.fit(xtr, ytr, groups=gtr)
                    pred = clf.predict(xse)
                    result = score(yse, pred, gse, site, 100, a.seed)
                    rows.append({'mode': mode, 'model': model, 'secret': secret, 'candidate': candidate, **result})
                    if model == 'cnn':
                        write_json({'history': clf.history_, 'fit_questions': clf.training_groups_,
                                    'validation_questions': clf.validation_groups_, 'params': params},
                                   out / 'histories' / f'{mode}_{secret}_{candidate}.json')
                        if best is None or result['balanced_accuracy'] > best['select_balanced_accuracy']:
                            best = {'params': params, 'candidate': candidate,
                                    'select_balanced_accuracy': result['balanced_accuracy']}
                if best is not None:
                    receipt['choices'][mode][secret] = best
        write_csv(rows, out / 'select_scores.csv')
        write_json(receipt, out / 'cnn_selection.json')
        print(f'selected attackers for {mode}', flush=True)
    receipt['state'] = 'locked'
    write_json(receipt, out / 'cnn_selection.json')


if __name__ == '__main__':
    main()
