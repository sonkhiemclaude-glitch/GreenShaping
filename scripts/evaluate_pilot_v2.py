"""Five-seed diagnostics of paired live captures and offline hypothetical defences.

No measured energy, no certified privacy, no repeated physical acquisitions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_pilot import candidates, labels_and_features

from greenshaping.attack.classifiers import RandomForestAttacker, TimeSliceCNN
from greenshaping.attack.evaluation import assert_disjoint, split_by_question
from greenshaping.attack.features import build_linkaware_tables, build_tables, build_time_slice_tables
from greenshaping.data.adapters import SessionDirectoryAdapter
from greenshaping.experiments.common import PLACEHOLDER_COST_MODEL, RunContext, cost_summary, shape_all
from greenshaping.metrics.confidence_intervals import bootstrap_by_group
from greenshaping.metrics.metrics import chance_level, majority_class, normalised_advantage, roc_auc
from greenshaping.shaping.pipeline import ShapingConfig, resolve_delta
from greenshaping.utils.io import read_json, write_csv, write_json
from greenshaping.utils.provenance import file_sha256, run_metadata

MODELS = ('rf', 'rf_linkaware', 'identity', 'cnn')
SECRETS = ('participation', 'size_bucket', 'query_family')
LABEL = 'balanced synthetic single-host pilot; not a paper result; energy is modelled only'
CNN: dict[str, Any] = {'epochs': 30, 'batch_size': 64, 'patience': 8, 'val_fraction': .2, 'lr': .001}


def score(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, site: np.ndarray | None,
          n_boot: int, seed: int) -> dict[str, Any]:
    def chance(index: np.ndarray) -> float:
        if site is None:
            return chance_level(y[index])
        return sum(np.sum(site[index] == s) * chance_level(y[index][site[index] == s])
                   for s in np.unique(site[index])) / len(index)

    def advantage(index: np.ndarray) -> float:
        return normalised_advantage(float(np.mean(y[index] == pred[index])), chance(index))

    idx = np.arange(len(y))
    lo, hi = bootstrap_by_group(groups, advantage, n_boot=n_boot, seed=seed)
    acc, ch = float(np.mean(y == pred)), chance(idx)
    return {'accuracy': acc, 'balanced_accuracy': float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])),
            'chance': ch, 'global_chance': chance_level(y), 'adv_raw': advantage(idx),
            'adv_clipped': max(0., advantage(idx)), 'adv_ci_low': lo, 'adv_ci_high': hi, 'n_test': len(y),
            'n_test_questions': len(np.unique(groups)), 'chance_mode': 'site_conditioned' if site is not None else 'global'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(20261021, 20261026)))
    p.add_argument('--n-boot', type=int, default=500)
    p.add_argument('--live-only', action='store_true')
    p.add_argument('--modes', nargs='+', default=['none', 'pad', 'dp'])
    p.add_argument('--cnn-selection', help='locked train/select-only choices for each live mode and secret')
    args = p.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    live = {}
    inputs: list[str] = []
    cnn_selection = read_json(args.cnn_selection) if args.cnn_selection else None
    if cnn_selection and (cnn_selection['state'] != 'locked' or not args.live_only or
                          cnn_selection['pcap_sha256'] != file_sha256(root / 'traffic.pcap')):
        raise ValueError('CNN choices must be locked for this exact live acquisition')
    if args.modes[0] != 'none':
        raise ValueError('unshaped reference must be first')
    for mode in args.modes:
        ad = SessionDirectoryAdapter(root / mode, gap_s=.005)
        live[mode] = sorted(ad.load(), key=lambda s: s.question_id)
        inputs.extend(str(f) for f in ad.input_files())
        if len(live[mode]) != 192 or len({s.question_id for s in live[mode]}) != 192:
            raise ValueError('evaluation requires all 192 unique questions per live mode')
        if any(s.meta['data_kind'] != 'rehearsal' or not s.meta.get('query_correct') for s in live[mode]):
            raise ValueError('expected validated rehearsal sessions')
    for mode in args.modes[1:]:
        for a, b in zip(live['none'], live[mode], strict=True):
            if any(getattr(a, k) != getattr(b, k) for k in ('question_id', 'query_family', 'participation', 'size_bucket')):
                raise ValueError('live defences do not share question labels')
    status: dict[str, Any] = {'state': 'running', 'completed': 0}
    scores, costs, paired = [], [], []
    protocol = run_metadata({'seeds': args.seeds, 'n_boot': args.n_boot, 'models': MODELS, 'cnn': CNN}, 'rehearsal', inputs)
    protocol.update(result_status=LABEL, evaluator_sha256=file_sha256(__file__),
                    split='immutable explicit 24/12/12 questions per family; computational seeds do not change split',
                    seed_scope='attacker/shaping RNG only; one physical acquisition per question/configuration',
                    resolution={'slice_s': .05, 'n_slices': 120, 'gap_s': .005},
                    chance='oracle constant per site on test labels, diagnostic; never traffic features',
                    dp_warning='session epsilon upper bound 2*N*eps_dir; no end-to-end timing certification')
    if cnn_selection:
        protocol['cnn_selection'] = cnn_selection
        protocol['cnn_selection_sha256'] = file_sha256(args.cnn_selection)
    write_json(protocol, out / 'protocol.json')
    pred_file = (out / 'predictions.csv').open('w', newline='')
    writer = csv.DictWriter(pred_file, fieldnames=['seed', 'config', 'mode', 'model', 'secret', 'question_id', 'site',
                                                  'y_true', 'y_pred', 'p1'])
    writer.writeheader()
    try:
        for seed in args.seeds:
            train, select, test = split_by_question(live['none'], seed=seed)
            assert_disjoint(train, select, test)
            fit = train + select
            write_json({k: sorted({s.question_id for s in v}) for k, v in [('train', train), ('select', select), ('test', test)]},
                       out / f'split_{seed}.json')
            ctx = RunContext({}, 'rehearsal', out, seed, live['none'], train, select, test,
                             {'default': PLACEHOLDER_COST_MODEL}, [])
            jobs: list[tuple[str, str, ShapingConfig | None]] = [(f'live/{mode}', mode, None) for mode in live]
            if not args.live_only:
                jobs += [('offline/' + c.label(), 'offline', c) for c in candidates(fit)[1:]]
            baseline: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for name, mode, cfg in jobs:
                status.update(seed=seed, config=name)
                write_json(status, out / 'status.json')
                cost: dict[str, Any]
                if cfg is None:
                    tr, se, te = split_by_question(live[mode], seed=seed)
                    sh_fit, sh_test = tr + se, te
                    # Observed byte ratio/latency; Joules are unavailable on live captures.
                    before_bytes = sum(sum(m.size for m in s.messages) for s in test)
                    after_bytes = sum(sum(m.size for m in s.messages) for s in te)
                    cost = {'bandwidth_overhead': after_bytes / before_bytes - 1,
                            'observed_execution_median_s': float(np.median([s.duration() for s in te])),
                            'energy_measured_site_max_median_J': float('nan'), 'privacy_certified': False}
                    cfg_id = sh_test[0].meta['config_id']
                else:
                    cfg = resolve_delta(cfg, fit, seed)
                    sh_fit, _ = shape_all(fit, cfg, seed, 'fit')
                    sh_test, stats = shape_all(test, cfg, seed, 'test')
                    cost = cost_summary(ctx, test, sh_test, stats, PLACEHOLDER_COST_MODEL)
                    cfg_id = cfg.config_id()
                resolved_json = json.dumps(cfg.to_dict() if cfg is not None else
                                           read_json(root / 'transport_configs.json')[mode], sort_keys=True)
                costs.append({'seed': seed, 'config': name, 'mode': mode, 'config_id': cfg_id, 'resolved_config_json': resolved_json,
                              **cost, 'result_status': LABEL})
                for model in MODELS:
                    if model == 'cnn':
                        train_tables = build_time_slice_tables(sh_fit, .05, 120)
                        test_tables = build_time_slice_tables(sh_test, .05, 120)
                    else:
                        def tables(sessions: list, model: str = model) -> tuple[dict, dict]:
                            return (build_tables(sessions) if model == 'rf' else
                                    build_linkaware_tables(sessions, identity_only=model == 'identity'))
                        train_tables, test_tables = tables(sh_fit), tables(sh_test)
                    for secret in SECRETS:
                        Xtr, ytr, gtr = labels_and_features(train_tables, secret)
                        Xte, yte, gte = labels_and_features(test_tables, secret)
                        site = np.asarray([i for s in sh_test for i in range(s.n_sites)]) if secret != 'query_family' else None
                        if secret == 'size_bucket':
                            assert site is not None
                            site = site[test_tables[0]['part'] == 1]
                        clf: Any = None
                        if len(np.unique(ytr)) < 2:
                            pred = np.full(len(yte), majority_class(ytr))
                            p1 = (pred == 1).astype(float)
                        else:
                            cnn_params = (cnn_selection['choices'][mode][secret]['params'] if cnn_selection else CNN)
                            clf = (TimeSliceCNN(seed=seed, **cnn_params) if model == 'cnn' else
                                   RandomForestAttacker(n_estimators=200, seed=seed, n_jobs=4))
                            clf.fit(Xtr, ytr, groups=gtr)
                            pred = clf.predict(Xte)
                            classes = list(clf.classes_)
                            proba = clf.predict_proba(Xte)
                            p1 = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(yte))
                        result = score(yte, pred, gte, site, args.n_boot, seed)
                        result['auc'] = roc_auc(yte, p1) if secret == 'participation' else float('nan')
                        result['train_majority_accuracy'] = float(np.mean(yte == majority_class(ytr)))
                        scores.append({'seed': seed, 'config': name, 'mode': mode, 'model': model,
                                       'secret': secret, **result, 'result_status': LABEL})
                        if model == 'cnn' and clf is not None:
                            write_json({'history': clf.history_, 'fit_questions': clf.training_groups_,
                                        'validation_questions': clf.validation_groups_, 'channel_mean': clf.mean_.tolist(),
                                        'channel_std': clf.scale_.tolist()},
                                       out / 'cnn' / f'{seed}_{mode}_{cfg_id}_{secret}.json')
                        for j in range(len(yte)):
                            writer.writerow({'seed': seed, 'config': name, 'mode': mode, 'model': model, 'secret': secret,
                                             'question_id': int(gte[j]), 'site': int(site[j]) if site is not None else -1,
                                             'y_true': int(yte[j]), 'y_pred': int(pred[j]), 'p1': float(p1[j])})
                        if name == 'live/none':
                            baseline[(model, secret)] = (yte.copy(), pred.copy(), gte.copy())
                        else:
                            by, bp, bg = baseline[(model, secret)]
                            if not np.array_equal(by, yte) or not np.array_equal(bg, gte):
                                raise ValueError('paired question/label order mismatch')

                            def gain(index: np.ndarray, bp: np.ndarray = bp, pred: np.ndarray = pred,
                                     yte: np.ndarray = yte) -> float:
                                # Paired accuracy gain avoids a moving bootstrap chance denominator.
                                return float(np.mean(bp[index] == yte[index]) - np.mean(pred[index] == yte[index]))

                            lo, hi = bootstrap_by_group(gte, gain, n_boot=args.n_boot, seed=seed)
                            paired.append({'seed': seed, 'config': name, 'model': model, 'secret': secret,
                                           'accuracy_reduction': gain(np.arange(len(yte))), 'ci_low': lo, 'ci_high': hi,
                                           'result_status': LABEL})
                status['completed'] += 1
                write_csv(scores, out / 'scores.csv')
                write_csv(costs, out / 'costs.csv')
                if paired:
                    write_csv(paired, out / 'paired.csv')
                pred_file.flush()
                print(json.dumps(status), flush=True)
        status['state'] = 'complete'
    except Exception as e:
        status.update(state='failed', error=str(e))
        raise
    finally:
        pred_file.close()
        write_json(status, out / 'status.json')


if __name__ == '__main__':
    main()
