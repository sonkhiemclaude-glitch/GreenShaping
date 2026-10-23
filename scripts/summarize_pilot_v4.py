"""Derive live comparisons from completed v4 evidence without changing any model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greenshaping.data.adapters import SessionDirectoryAdapter
from greenshaping.metrics.metrics import normalised_advantage
from greenshaping.utils.io import read_json, write_json
from greenshaping.utils.provenance import file_sha256

MODES = ['none', 'pad', 'rounds', 'dp', 'combo']
SECRETS = ['participation', 'size_bucket', 'query_family']


def maximum_advantage(frames: list[pd.DataFrame], ids: np.ndarray) -> float:
    maximum = 0.
    for frame in frames:
        # Repeated question IDs preserve paired, question-level bootstrap sampling.
        groups = {q: g for q, g in frame.groupby('question_id')}
        data = pd.concat([groups[int(q)] for q in ids], ignore_index=True)
        acc = float((data.y_true == data.y_pred).mean())
        if int(data.site.iloc[0]) == -1:
            chance = float(data.y_true.value_counts(normalize=True).max())
        else:
            chance = sum(len(s) * float(s.y_true.value_counts(normalize=True).max())
                         for _, s in data.groupby('site')) / len(data)
        maximum = max(maximum, normalised_advantage(acc, chance))
    return maximum


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n-boot', type=int, default=500)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    if read_json(root / 'workflow.json')['state'] != 'complete':
        raise ValueError('workflow must be complete')
    out.mkdir(parents=True, exist_ok=False)
    scores = pd.read_csv(root / 'evaluation/scores.csv')
    costs = pd.read_csv(root / 'evaluation/costs.csv')
    selection = read_json(root / 'selection/cnn_selection.json')
    protocol = read_json(root / 'evaluation/protocol.json')
    assert protocol['cnn_selection_sha256'] == file_sha256(root / 'selection/cnn_selection.json')
    assert selection['state'] == 'locked'
    selected = pd.read_csv(root / 'selection/select_scores.csv')
    keep = [r.model != 'cnn' or int(r.candidate) == selection['choices'][r.mode][r.secret]['candidate']
            for r in selected.itertuples()]
    selected = selected[keep]
    select_max = selected.groupby('mode').adv_clipped.max().to_dict()
    live = root / 'runs/adaptive_01'
    select_bytes = {}
    manifest_rows = []
    for mode in MODES:
        sessions = SessionDirectoryAdapter(live / mode, gap_s=.005).load()
        assert len(sessions) == 192 and all(s.meta['query_correct'] for s in sessions)
        select_bytes[mode] = sum(sum(m.size for m in s.messages) for s in sessions if s.meta['partition'] == 'select')
        for s in sessions:
            manifest_rows.append({'mode': mode, 'question': s.question_id, 'family': s.query_family,
                                  'partition': s.meta['partition'], 'rounds': s.meta['round_count']})
    # Fixed tie rule: least select advantage, then least observed select bytes, then name.
    best_single = min(['pad', 'rounds', 'dp'], key=lambda m: (select_max[m], select_bytes[m], m))
    strongest = scores.groupby(['seed', 'mode', 'secret']).adv_clipped.max().reset_index()
    strongest.to_csv(out / 'strongest_by_secret.csv', index=False)
    maximum = strongest.groupby(['seed', 'mode']).adv_clipped.max().reset_index(name='adv_max')
    maximum.to_csv(out / 'strongest_by_seed.csv', index=False)
    summary = maximum.groupby('mode').adv_max.agg(['mean', 'min', 'max']).reindex(MODES)
    summary.to_csv(out / 'strongest_summary.csv')
    cost = costs.groupby('mode').first().reindex(MODES)
    cost['observed_payload_factor'] = 1 + cost.bandwidth_overhead
    cost[['observed_payload_factor', 'observed_execution_median_s']].to_csv(out / 'observed_costs.csv')
    rounds = pd.DataFrame(manifest_rows)
    rounds.to_csv(out / 'round_counts.csv', index=False)
    # Labels and deterministic query traces must match across every transport.
    assert rounds.pivot(index='question', columns='mode', values='rounds').nunique(axis=1).max() == 1
    preds = pd.read_csv(root / 'evaluation/predictions.csv')
    first_seed = int(preds.seed.min())
    frames = {}
    for mode in (best_single, 'combo'):
        data = preds[(preds.seed == first_seed) & (preds.config == f'live/{mode}')]
        frames[mode] = [g for _, g in data.groupby(['model', 'secret'])]
    ids = np.sort(preds.question_id.unique())
    rng = np.random.default_rng(20261022)
    deltas = []
    for _ in range(a.n_boot):
        resample = rng.choice(ids, len(ids), replace=True)
        deltas.append(maximum_advantage(frames[best_single], resample) - maximum_advantage(frames['combo'], resample))
    ci = np.percentile(deltas, [2.5, 97.5]).tolist()
    pivot = maximum.pivot(index='seed', columns='mode', values='adv_max')
    reductions = (pivot[best_single] - pivot.combo).to_dict()
    colors = {'rf': '#2975c5', 'rf_linkaware': '#df7b20', 'identity': '#666666', 'cnn': '#299a70'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, secret in zip(axes, SECRETS, strict=True):
        for index, (model, color) in enumerate(colors.items()):
            values = scores[(scores.secret == secret) & (scores.model == model)].groupby('mode').adv_clipped
            mean, low, high = [values.agg(v).reindex(MODES).to_numpy() for v in ('mean', 'min', 'max')]
            ax.errorbar(np.arange(5) + (index - 1.5) * .13, mean, yerr=[mean-low, high-mean],
                        color=color, fmt='o-', linewidth=1, label=model)
        ax.set(title=secret, xticks=range(5), xticklabels=MODES, ylim=(-.03, 1.05))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Normalised advantage, clipped at zero')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Adaptive synthetic TLS pilot; ranges across five attacker seeds, not acquisition repeats', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'live_advantage.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(MODES, cost.observed_payload_factor)
    axes[0].set(yscale='log', ylabel='Observed TCP payload / unshaped payload', title='Traffic cost on final-test questions')
    axes[1].bar(MODES, cost.observed_execution_median_s)
    axes[1].set(ylabel='Seconds', title='Observed window duration; includes DP cover tail')
    fig.suptitle('Single Mac Studio, loopback, synthetic data; energy not measured', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'observed_costs.png', dpi=160)
    plt.close(fig)
    result = {'best_single_from_select': best_single, 'select_max_advantage': select_max,
              'tie_rule': 'least select advantage, then least select TCP payload, then mode name',
              'test_strongest_summary': summary.reset_index().to_dict(orient='records'),
              'composition_advantage_reduction_by_seed': reductions,
              'paired_question_bootstrap_first_seed': {'seed': first_seed, 'n_boot': a.n_boot,
                                                       'reduction_ci95': ci, 'positive_means_composition_better': True,
                                                       'scope': 'max over four attackers and three secrets; paired question resampling'},
              'observed_costs': cost[['observed_payload_factor', 'observed_execution_median_s']].reset_index().to_dict(orient='records'),
              'score_rows': len(scores), 'score_sha256': file_sha256(root / 'evaluation/scores.csv'),
              'selection_sha256': file_sha256(root / 'selection/cnn_selection.json'),
              'summary_script_sha256': file_sha256(__file__), 'energy_measured': False, 'privacy_certified': False,
              'status': 'synthetic single-host pilot; no claim of paper confirmation'}
    write_json(result, out / 'summary.json')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
