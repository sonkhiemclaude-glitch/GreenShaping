"""Strict paired-unshaped energy records; never pool deployments or repetitions."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np

IDENTITY = ('run_id', 'deployment_id', 'N', 'llm', 'query_mix', 'config_id')
REQUIRED = (*IDENTITY, 'node', 'node_class', 'session_id', 'repetition', 'extra_joules', 'reference')


def attach_measurements(rows: list[dict[str, Any]], records: list[dict[str, Any]], strict: bool = False) -> None:
    seen = set()
    for d in records:
        if any(k not in d or str(d[k]) == '' for k in REQUIRED):
            raise ValueError(f'energy record requires columns {REQUIRED}')
        if d['reference'] != 'paired_unshaped':
            raise ValueError('extra energy must reference a paired unshaped session')
        if not math.isfinite(float(d['extra_joules'])):
            raise ValueError('nonfinite measured energy')
        key = tuple(str(d[k]) for k in (*IDENTITY, 'node', 'session_id', 'repetition'))
        if key in seen:
            raise ValueError('duplicate measured energy record')
        seen.add(key)
    for r in rows:
        if any(k not in r for k in IDENTITY):
            raise ValueError('result row lacks energy join identity')
        matches = [d for d in records if all(str(d[k]) == str(r[k]) for k in IDENTITY)]
        classes = r.get('node_classes', {})
        if isinstance(classes, str):
            classes = json.loads(classes)
        by = defaultdict(list)
        for d in matches:
            expected = classes.get(d['node']) if r.get('node_class') == 'per-manifest' else r.get('node_class')
            if d['node'] != 'orchestrator' and expected and d['node_class'] != expected:
                continue
            by[(str(d['repetition']), d['node'])].append(float(d['extra_joules']))
        # Each repeat contributes one session mean per node; each repeat has equal weight.
        repeats = sorted({k[0] for k in by})
        expected_nodes = {'orchestrator', *(f'site{i}' for i in range(int(r['N'])))}
        complete = bool(repeats) and all({n for rr, n in by if rr == rep} == expected_nodes for rep in repeats)
        r['energy_measurements_complete'] = complete
        if strict and not complete:
            raise ValueError('paired energy measurements must include every site and orchestrator in every repetition')
        if strict:
            for rep in repeats:
                node_sessions = [{str(d['session_id']) for d in matches if str(d['repetition']) == rep and d['node'] == node}
                                 for node in sorted(expected_nodes)]
                if any(x != node_sessions[0] for x in node_sessions):
                    raise ValueError('paired energy nodes do not share the same session set')
        orch = [float(np.mean(by[(rep, 'orchestrator')])) for rep in repeats if (rep, 'orchestrator') in by]
        site_max = [max(float(np.mean(v)) for (rr, n), v in by.items() if rr == rep and n.startswith('site'))
                    for rep in repeats if any(rr == rep and n.startswith('site') for rr, n in by)]
        for kind, values in (('orchestrator', orch), ('site_max', site_max)):
            for suffix, pct in (('median', 50), ('q25', 25), ('q75', 75)):
                r[f'energy_measured_{kind}_{suffix}_J'] = float(np.percentile(values, pct)) if values else float('nan')
        r['energy_measured_runs'] = len(orch)
        r['energy_measured_site_runs'] = len(site_max)
        r['energy_measured_sessions'] = len(matches)
        r['energy_measurement_schema'] = 'paired-v2'
