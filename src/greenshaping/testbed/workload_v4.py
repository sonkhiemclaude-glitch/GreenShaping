"""Fresh synthetic private-data workload with causal, bounded adaptive rounds.

This is a controlled stress workload, not a natural-language autonomous agent or
an independent clinical dataset. No response is padded by the workload itself.
"""
from __future__ import annotations

import copy
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


def build_workload(root: str | Path, seed: int = 20261022, n_sites: int = 5) -> list[Question]:
    if n_sites != 5:
        raise ValueError('v4 balanced design is specified for five sites')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    tables = []
    for site in range(n_sites):
        n = 165_000
        cohort = rng.choice(3, n, p=[.025, .22, .755])
        x1, x2 = rng.normal(size=(2, n))
        segment = rng.integers(0, 12, n)
        df = pd.DataFrame({'cohort': cohort, 'x1': x1, 'x2': x2,
                           'value': 100 + 4 * cohort + 15 * x1 + rng.normal(size=n),
                           'outcome': (x1 + x2 + .1 * site > 0).astype(int), 'marker': rng.uniform(size=n),
                           'flag': (x1 > 0).astype(int), 'segment_coarse': segment % 2,
                           'segment_medium': segment % 6, 'segment_fine': segment})
        p = root / f'site_{site}.csv'
        df.to_csv(p, index=False)
        tables.append({'site': site, 'rows': n, 'sha256': file_sha256(p),
                       'cohort_counts': df.cohort.value_counts().sort_index().to_dict()})
    questions: list[Question] = []
    for family in QUERY_FAMILIES:
        for j in range(48):
            partition = 'train' if j < 24 else 'select' if j < 36 else 'test'
            c, phase = j % 4, (j % 12) // 4
            chosen = [i for i in range(n_sites) if (i % 2 == 0 if c == 0 else i % 2 == 1 if c == 1
                                                   else i < 3 if c == 2 else i >= 3)]
            calls = []
            for i in chosen:
                params: dict[str, Any] = {'where': {'cohort': (phase + i) % 3,
                                                  'marker': {'min': float(rng.uniform(0, .05)),
                                                             'max': float(rng.uniform(.95, 1))}}}
                if family == 'aggregate':
                    params['columns'] = ['value']
                elif family == 'group_by':
                    params.update(by='segment_coarse', column='value')
                elif family == 'train_round':
                    params.update(features=['x1', 'x2'], label='outcome', weights=[0., 0., 0.], lr=.2)
                calls.append({'site': i, 'params': params})
            # Public stopping tolerances vary independently of family, site and partition.
            tolerance = (.015, .04, .12)[(j // 4) % 3]
            policy = {'min_rounds': 2, 'max_rounds': 4, 'tolerance': tolerance,
                      'target_count': (5000, 40000, 140000)[(j // 4) % 3],
                      'target_mean': 104. + float(rng.uniform(-2, 2))}
            text = ('Select the initial request sites for this synthetic iterative analysis. '
                    'The controller will apply the declared bounded stopping rule to actual tool outputs. '
                    + json.dumps({'family': family, 'requests': calls, 'policy': policy}, sort_keys=True))
            questions.append(Question(100_000 + len(questions), family, text, [calls],
                                      {'partition': partition, 'workload': 'adaptive-v4', 'synthetic': True,
                                       'adaptive_policy': policy}))
    write_jsonl([asdict(q) for q in questions], root / 'questions.jsonl')
    write_json({'seed': seed, 'n_sites': n_sites, 'tables': tables, 'questions_per_family': 48,
                'question_sha256': file_sha256(root / 'questions.jsonl'), 'split': {'train': 24, 'select': 12, 'test': 12},
                'new_question_id_namespace': 100_000,
                'limitations': ['synthetic single-host data', 'typed analyst inputs; bounded controller, not autonomous reasoning',
                                'repeated queries may overlap rows; contribution label is maximum per-call matching rows',
                                'new questions share a synthetic generator and site tables; not external generalization']},
               root / 'workload.json')
    return questions


def next_calls(question: Question, calls: list[dict[str, Any]], responses: list[dict[str, Any]],
               completed_rounds: int) -> list[dict[str, Any]]:
    """Compute the next round solely from actual previous outputs and public policy."""
    p = question.metadata['adaptive_policy']
    if len(calls) != len(responses) or not calls:
        raise ValueError('one actual response is required per call')
    if completed_rounds >= p['max_rounds']:
        return []
    results = [r['result'] for r in responses]
    family = question.family
    updated = copy.deepcopy(calls)
    if family == 'train_round':
        new_w = np.average([r['weights'] for r in results], axis=0,
                           weights=[max(1, r['n']) for r in results])
        delta = float(np.linalg.norm(new_w - np.asarray(calls[0]['params']['weights'])))
        stop = delta <= p['tolerance']
        for call in updated:
            call['params']['weights'] = new_w.tolist()
    elif family == 'group_by':
        stop = all(max(v['count'] for v in r.values()) / sum(v['count'] for v in r.values()) <= p['tolerance'] + .05
                   for r in results)
        for call in updated:
            call['params']['by'] = ('segment_medium', 'segment_fine', 'segment_fine')[min(completed_rounds - 1, 2)]
    elif family == 'aggregate':
        means = [r['value']['sum'] / max(1, r['value']['count']) for r in results]
        stop = all(abs(m - p['target_mean']) <= 10 * p['tolerance'] for m in means)
        for call, mean in zip(updated, means, strict=True):
            call['params']['columns'] = ['value', 'x1'] if mean >= p['target_mean'] else ['value', 'x2']
            call['params']['where']['flag'] = int(mean >= p['target_mean'])
    elif family == 'record_count':
        stop = all(r['count'] >= p['target_count'] for r in results)
        for call, result in zip(updated, results, strict=True):
            call['params']['where']['flag'] = int(result['count'] % 2)
    else:
        raise ValueError(family)
    return [] if stop and completed_rounds >= p['min_rounds'] else updated
