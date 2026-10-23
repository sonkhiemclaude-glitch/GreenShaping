"""Convert paired raw integrals to extra-over-unshaped energy without mixing idle baselines."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .measurements import IDENTITY

PAIR = (*IDENTITY, 'node', 'node_class', 'session_id', 'repetition')


def pair_energy_records(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in raw:
        required = (*PAIR, 'condition', 'energy_joules', 'idle_joules', 'sensor', 'scope', 'acquisition_id')
        if any(k not in r or str(r[k]) == '' for k in required):
            raise ValueError(f'raw energy row requires {required}')
        condition = r['condition']
        if condition not in ('shaped', 'unshaped'):
            raise ValueError('condition must be shaped or unshaped')
        if any(not math.isfinite(float(r[k])) or float(r[k]) < 0 for k in ('energy_joules', 'idle_joules')):
            raise ValueError('raw and idle integrals must be finite and nonnegative')
        key = tuple(str(r[k]) for k in PAIR)
        if condition in grouped[key]:
            raise ValueError('duplicate condition in an energy pair')
        grouped[key][condition] = r
    out = []
    for pair in grouped.values():
        if set(pair) != {'shaped', 'unshaped'}:
            raise ValueError('missing shaped or unshaped measurement in energy pair')
        a, b = pair['shaped'], pair['unshaped']
        if any(a[k] != b[k] for k in ('sensor', 'scope')):
            raise ValueError('paired measurements use different sensors or measurement scopes')
        extra = (float(a['energy_joules']) - float(a['idle_joules'])) - (float(b['energy_joules']) - float(b['idle_joules']))
        out.append({k: a[k] for k in PAIR} | {'extra_joules': extra, 'reference': 'paired_unshaped',
                    'sensor': a['sensor'], 'scope': a['scope'], 'shaped_acquisition_id': a['acquisition_id'],
                    'unshaped_acquisition_id': b['acquisition_id']})
    return out
