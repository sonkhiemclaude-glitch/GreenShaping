"""Compact offline E1/E2/E3/E5 controller rehearsal from balanced unshaped captures."""
from __future__ import annotations

import argparse
from pathlib import Path

from greenshaping.experiments.plot_results import make_all_plots
from greenshaping.experiments.runner import run_from_config
from greenshaping.utils.io import dump_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError('choose a fresh controller output directory')
    cfg = {'run': {'id': 'balanced-pilot-controller-v3', 'deployment_id': 'mac-studio-single-host',
                   'seed': 20261021, 'data_kind': 'rehearsal', 'output_dir': str(out)},
           'data': {'source': 'directory', 'root': str(Path(args.root).resolve()), 'gap_s': .005},
           'attack': {'models': ['rf', 'rf_linkaware', 'identity', 'cnn'], 'n_estimators': 200, 'n_boot': 500,
                      'chance_mode': 'site_conditioned',
                      'cnn': {'epochs': 30, 'batch_size': 64, 'patience': 8, 'val_fraction': .2, 'slice_s': .05, 'n_slices': 120}},
           'energy': {'cost_models': {}, 'measured_energy_csv': None},
           'grid': {'padding': {'ratio': [2.], 'base': [512]},
                    'rounds': {'window': [.1], 'cover_prob': [.5, 1.], 'reply_frame': [1024]},
                    'dp': {'mode': ['queue', 'arrivals'], 'interval': [.1], 'eps_dir': [4.],
                           'delta_percentile': [100.], 'horizon': [2.]}},
           'budgets': {'site_J': []}, 'baselines': {'netshaper_root': None},
           'experiments': {'e5': {'alpha': .6, 'query_mixes': [['aggregate', 'group_by', 'train_round', 'record_count'],
                                                            ['aggregate', 'record_count']]}}}
    dump_yaml(cfg, out.parent / 'controller_config.yaml')
    for stage in ('e1', 'e2', 'e3', 'e5'):
        run_from_config(cfg, stage)
    make_all_plots(out, out / 'plots')


if __name__ == '__main__':
    main()
