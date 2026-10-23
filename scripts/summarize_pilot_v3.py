"""Tables and figures derived from completed pilot evaluation artifacts only."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from greenshaping.utils.io import read_json, write_json
from greenshaping.utils.provenance import file_sha256

SECRETS = ('participation', 'size_bucket', 'query_family')
COLORS = {'rf': '#2a78d6', 'rf_linkaware': '#eb6834', 'identity': '#52514e', 'cnn': '#1baf7a'}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root, out = Path(args.evaluation), Path(args.out)
    if read_json(root / 'status.json')['state'] != 'complete':
        raise ValueError('evaluation is incomplete')
    out.mkdir(parents=True, exist_ok=False)
    scores, costs = pd.read_csv(root / 'scores.csv'), pd.read_csv(root / 'costs.csv')
    grouped = scores.groupby(['config', 'mode', 'model', 'secret'])
    summary = grouped[['accuracy', 'balanced_accuracy', 'adv_raw', 'adv_clipped', 'auc']].agg(['mean', 'min', 'max'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / 'attacker_summary.csv', index=False)
    strongest = scores.groupby(['seed', 'config', 'mode', 'secret']).adv_clipped.max().reset_index()
    worst = strongest.groupby(['seed', 'config', 'mode']).adv_clipped.max().reset_index(name='adv_max')
    worst.to_csv(out / 'strongest_by_seed.csv', index=False)
    worst_summary = worst.groupby(['config', 'mode']).adv_max.agg(['mean', 'min', 'max']).reset_index()
    worst_summary.to_csv(out / 'strongest_summary.csv', index=False)
    preds = pd.read_csv(root / 'predictions.csv')
    seed = int(preds.seed.min())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    modes = ['live/none', 'live/pad', 'live/dp']
    for ax, secret in zip(axes, SECRETS, strict=True):
        for index, model in enumerate(COLORS):
            data = scores[(scores.secret == secret) & (scores.model == model)]
            means, low, high = [], [], []
            for mode in modes:
                values = data[data.config == mode].adv_clipped
                means.append(values.mean())
                low.append(values.mean() - values.min())
                high.append(values.max() - values.mean())
            x = np.arange(3) + (index - 1.5) * .15
            ax.errorbar(x, means, yerr=[low, high], fmt='o-', color=COLORS[model], label=model, linewidth=1)
        ax.set_xticks(range(3), ['none', 'padding', 'DP'])
        ax.set_ylim(-.03, 1.05)
        ax.set_title(secret)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Normalised advantage, clipped at zero')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Live synthetic single-host pilot; ranges across computational seeds (not independent acquisitions)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'live_advantage.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in modes:
        for model in ('rf_linkaware', 'cnn'):
            data = preds[(preds.seed == seed) & (preds.config == mode) & (preds.model == model)
                         & (preds.secret == 'participation')]
            fpr, tpr, _ = roc_curve(data.y_true, data.p1)
            ax.plot(fpr, tpr, label=f'{mode.removeprefix("live/")} / {model}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.7)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'Participation ROC; seed {seed}; rehearsal')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'participation_roc.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, mode in zip(axes, modes, strict=True):
        data = preds[(preds.seed == seed) & (preds.config == mode) & (preds.model == 'rf_linkaware')
                     & (preds.secret == 'query_family')]
        matrix = confusion_matrix(data.y_true, data.y_pred, labels=range(4))
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=12)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
        ax.set(title=mode, xlabel='Predicted family', ylabel='True family', xticks=range(4), yticks=range(4))
    fig.suptitle('Query family confusion; link-aware RF; 48 final-test questions; rehearsal', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'query_confusion.png', dpi=160)
    plt.close(fig)
    live_costs = costs[costs['mode'] != 'offline'].groupby('config').first().reset_index()
    live_costs.to_csv(out / 'live_costs.csv', index=False)
    write_json({'score_sha256': file_sha256(root / 'scores.csv'), 'prediction_sha256': file_sha256(root / 'predictions.csv'),
                'strongest': worst_summary.to_dict(orient='records'),
                'live_costs': live_costs[['config', 'bandwidth_overhead', 'observed_execution_median_s']].to_dict(orient='records'),
                'seed_count': int(scores.seed.nunique()), 'score_rows': len(scores),
                'scope': 'balanced synthetic single-host rehearsal; no measured energy or end-to-end DP certification'},
               out / 'summary.json')


if __name__ == '__main__':
    main()
