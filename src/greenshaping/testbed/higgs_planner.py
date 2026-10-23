"""Compact HIGGS one-round planner contract; no reference-plan fallback."""
from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

from .natural_planner import validate_requested_sites

FIELDS = {'record_count': set(), 'aggregate': {'columns'}, 'group_by': {'by', 'column'},
          'train_round': {'features', 'label', 'weights', 'lr'}}
COLUMNS = ['label', 'feature_01', 'feature_02']


def planner_schema(n_sites: int, include_feature_range: bool | None = None) -> dict:
    def obj(props):
        return {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}
    number = {'type': 'number'}
    bounds = obj({'min': number, 'max': number})
    where = obj({'cohort': bounds})
    decimal = {'type': 'string', 'maxLength': 32, 'pattern': r'^-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$'}
    if include_feature_range is not False:
        where['properties']['feature_01'] = obj({'min': decimal, 'max': decimal})
        if include_feature_range:
            where['required'].append('feature_01')
    col = {'type': 'string', 'enum': COLUMNS}
    cols = {'type': 'array', 'minItems': 1, 'maxItems': 3, 'items': col}
    params = {'record_count': {}, 'aggregate': {'columns': cols}, 'group_by': {'by': col, 'column': col},
              'train_round': {'features': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'uniqueItems': True,
                                        'items': {'type': 'string', 'enum': ['feature_01', 'feature_02']}},
                              'label': {'type': 'string', 'enum': ['label']},
                              'weights': {'type': 'array', 'minItems': 3, 'maxItems': 3, 'items': number}, 'lr': number}}
    return {'oneOf': [obj({'family': {'type': 'string', 'enum': [f]},
                          'sites': {'type': 'array', 'minItems': 1, 'maxItems': n_sites,
                                    'items': {'type': 'integer', 'minimum': 0, 'maximum': n_sites - 1}},
                          'params': obj({'where': where, **p})}) for f, p in params.items()]}


def question_schema(text: str, n_sites: int) -> dict:
    """Allow only range fields explicitly present in this public benchmark request.

    Predictor feature lists do not imply filtering. No reference values or plans
    are read; the model must still generate every argument and site itself.
    """
    return planner_schema(n_sites, include_feature_range='feature_01 between' in text)


def validate_plan(value: Any, n_sites: int, text: str) -> dict:
    if not isinstance(value, dict) or set(value) != {'family', 'sites', 'params'} or value['family'] not in FIELDS:
        raise ValueError('expected family, sites and shared params')
    value = copy.deepcopy(value)
    sites, p, family = value['sites'], value['params'], value['family']
    if (not isinstance(sites, list) or not sites or len(sites) > n_sites or
            any(type(s) is not int or not 0 <= s < n_sites for s in sites) or len(set(sites)) != len(sites)):
        raise ValueError('invalid or duplicate sites')
    if not isinstance(p, dict) or set(p) != {'where'} | FIELDS[family]:
        raise ValueError('invalid tool fields')
    where = p['where']
    if not isinstance(where, dict) or set(where) not in ({'cohort'}, {'cohort', 'feature_01'}):
        raise ValueError('invalid predicate columns')
    for name, bounds in where.items():
        if not isinstance(bounds, dict) or set(bounds) != {'min', 'max'}:
            raise ValueError('invalid inclusive range')
        if name == 'feature_01':
            for key in ('min', 'max'):
                if isinstance(bounds[key], str):
                    if len(bounds[key]) > 32 or not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', bounds[key]):
                        raise ValueError('feature bounds require finite decimal strings')
                    bounds[key] = float(bounds[key])
        low, high = bounds['min'], bounds['max']
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (low, high)) or low > high:
            raise ValueError('invalid finite bounds')
        if name == 'cohort' and (low != int(low) or high != int(high) or not 0 <= low <= high <= 4):
            raise ValueError('invalid cohort')
    for field in ('columns', 'features'):
        if field in p and (not isinstance(p[field], list) or not p[field] or
                           any(not isinstance(c, str) or c not in COLUMNS for c in p[field]) or
                           len(p[field]) != len(set(p[field]))):
            raise ValueError('invalid columns')
    for field in ('by', 'column'):
        if field in p and (not isinstance(p[field], str) or p[field] not in COLUMNS):
            raise ValueError('invalid column')
    if family == 'train_round':
        if not isinstance(p['weights'], list) or len(p['weights']) != len(p['features']) + 1:
            raise ValueError('weights must include the intercept plus one weight per feature: '
                             f'expected {len(p["features"]) + 1} weights')
        if (p['label'] != 'label' or not isinstance(p['weights'], list) or
                len(p['weights']) != len(p['features']) + 1 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in [*p['weights'], p['lr']]) or
                not 0 < p['lr'] <= 1):
            raise ValueError('invalid logistic parameters')
    plan = {'family': family, 'calls': [{'site': s, 'params': p} for s in sorted(sites)]}
    validate_requested_sites(plan, text)
    return plan


def planner_prompt(text: str, n_sites: int) -> str:
    return ('Translate the analyst request into one round. Return JSON with family, sites (unique integer IDs) and '
            'params (shared tool arguments, applied independently at every named site). No answers or tool execution. '
            'Use exactly the named sites. Filters are inclusive. where always includes cohort min/max; '
            'If no feature_01 range is requested, where MUST contain only cohort. '
            'Predictor features are not filter requirements. Never invent infinite bounds. '
            'Encode feature_01 bounds as QUOTED DECIMAL STRINGS, '
            'copying every digit exactly from the question without rounding. Cohort bounds remain numeric. '
            'Available columns: label, feature_01, feature_02. '
            'aggregate returns count/sum/sum of squares and uses columns; group_by returns grouped count/sum using '
            'by and column; train_round uses features, label, weights including intercept and lr. '
            'Its two predictors require THREE weights: intercept first, then the two feature weights; '
            'record_count returns only a count. Copy all bounds and arguments accurately from the analyst request. '
            f'Site IDs are 0 through {n_sites - 1}. Schema: ' + json.dumps(question_schema(text, n_sites)) + '\nAnalyst: ' + text)
