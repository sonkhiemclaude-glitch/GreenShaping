"""Balanced, synthetic-only workload; immutable 24/12/12 question split per family."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.schemas import QUERY_FAMILIES
from ..utils.io import write_json, write_jsonl
from ..utils.provenance import file_sha256
from .question_bank import Question


def build_workload(root: str | Path, seed: int = 20261021, n_sites: int = 5) -> list[Question]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    tables = []
    for site in range(n_sites):
        rows = 200_000
        x1, x2 = rng.normal(size=(2, rows))
        df = pd.DataFrame({'cohort': np.arange(rows) % 4, 'x1': x1, 'x2': x2,
                           'value': 100 + 15 * x1 + rng.normal(size=rows),
                           'outcome': (x1 + x2 > 0).astype(int)})
        p = root / f'site_{site}.csv'
        df.to_csv(p, index=False)
        tables.append({'site': site, 'path': p.name, 'rows': rows, 'sha256': file_sha256(p)})
    questions: list[Question] = []
    for family in QUERY_FAMILIES:
        for j in range(48):
            partition = 'train' if j < 24 else 'select' if j < 36 else 'test'
            k = j % 12
            c, phase = k % 4, k // 4
            chosen = [i for i in range(n_sites) if (i % 2 == 0 if c == 0 else i % 2 == 1 if c == 1
                                                    else i < (n_sites + 1) // 2 if c == 2 else i >= (n_sites + 1) // 2)]
            calls = []
            for i in chosen:
                params: dict[str, Any] = ({'columns': ['value', 'x1']} if family == 'aggregate' else
                          {'by': 'cohort', 'column': 'value'} if family == 'group_by' else
                          {'features': ['x1', 'x2'], 'label': 'outcome', 'weights': [0., 0., 0.]}
                          if family == 'train_round' else {})
                params.update(limit_rows=(2000, 20000, 120000)[(phase + i) % 3] + 31 * (j % 4), row_offset=137 * j)
                calls.append({'site': i, 'params': params})
            request = {'family': family, 'requests': calls}
            text = ('Analyze the synthetic table windows specified below. Issue exactly one tool call per request. '
                    'The row window is applied after filtering. Return the combined findings. ' + json.dumps(request, sort_keys=True))
            questions.append(Question(len(questions), family, text, [calls],
                                      {'partition': partition, 'workload': 'balanced-v2', 'synthetic': True}))
    write_jsonl([asdict(q) for q in questions], root / 'questions.jsonl')
    write_json({'seed': seed, 'n_sites': n_sites, 'questions_per_family': 48, 'tables': tables,
                'question_sha256': file_sha256(root / 'questions.jsonl'),
                'split': {'train': 24, 'select': 12, 'test': 12},
                'purpose': 'synthetic single-host pilot; no clinical or measured-energy claims'}, root / 'workload.json')
    return questions
