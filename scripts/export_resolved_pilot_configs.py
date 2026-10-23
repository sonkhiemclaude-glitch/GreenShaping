"""Recover resolved parameters for the frozen v3 pilot and verify every recorded config ID."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluate_pilot import candidates

from greenshaping.attack.evaluation import split_by_question
from greenshaping.data.adapters import SessionDirectoryAdapter
from greenshaping.shaping.pipeline import ShapingConfig, resolve_delta
from greenshaping.utils.io import read_csv, read_json, write_json
from greenshaping.utils.provenance import file_sha256


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    args = p.parse_args()
    root = Path(args.project)
    capture = root / 'runs/balanced_01'
    sessions = sorted(SessionDirectoryAdapter(capture / 'none', gap_s=.005).load(), key=lambda s: s.question_id)
    costs = read_csv(root / 'evaluation/costs.csv')
    live = read_json(capture / 'transport_configs.json')
    output = []
    for seed in sorted({int(r['seed']) for r in costs}):
        train, select, _ = split_by_question(sessions, seed=seed)
        fit = train + select
        mapping = {f'live/{mode}': ShapingConfig.from_dict(cfg) for mode, cfg in live.items()}
        mapping.update({'offline/' + cfg.label(): resolve_delta(cfg, fit, seed) for cfg in candidates(fit)[1:]})
        for row in costs:
            if int(row['seed']) != seed:
                continue
            cfg = mapping[row['config']]
            if cfg.config_id() != row['config_id']:
                raise ValueError('resolved parameters do not match the configuration evaluated')
            output.append({'seed': seed, 'name': row['config'], 'config_id': cfg.config_id(), 'parameters': cfg.to_dict()})
    write_json({'configs': output, 'source_sha256': file_sha256(__file__),
                'costs_sha256': file_sha256(root / 'evaluation/costs.csv'),
                'scope': 'post-run deterministic reconstruction; all 60 recorded IDs must match'},
               root / 'evaluation/resolved_configs.json')
    print(f'verified {len(output)} resolved configs using {sys.version.split()[0]}')


if __name__ == '__main__':
    main()
