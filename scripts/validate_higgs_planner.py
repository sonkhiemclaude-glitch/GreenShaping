"""Bounded train/select HIGGS LLM/tool checks; select requires a frozen source contract."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import greenshaping.testbed.higgs_planner as contract_module
import greenshaping.testbed.higgs_workload as workload_module
import greenshaping.testbed.natural_planner as site_validator_module
import greenshaping.testbed.queries as query_module
from greenshaping.testbed.higgs_planner import planner_prompt, question_schema, validate_plan
from greenshaping.testbed.higgs_workload import site_table
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.question_bank import load_question_bank
from greenshaping.utils.provenance import file_sha256


def source_contract():
    return {"runner_sha256": file_sha256(Path(__file__)),
            "planner_sha256": file_sha256(Path(contract_module.__file__)),
            "tool_sha256": file_sha256(Path(query_module.__file__)),
            "workload_sha256": file_sha256(Path(workload_module.__file__)),
            "site_validator_sha256": file_sha256(Path(site_validator_module.__file__))}


def guard_partition(questions, partition, frozen_contract):
    if partition not in ('train', 'select') or not questions or any(
            q.metadata.get('partition') != partition for q in questions):
        raise ValueError('only the explicitly requested train/select partition is allowed; never test')
    if partition == 'select' and (not frozen_contract or frozen_contract.get('source_contract') != source_contract()):
        raise ValueError('select requires a matching frozen planner/runner source contract')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sites', type=int, choices=[5, 10, 20], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--partition', choices=['train', 'select'], default='train')
    p.add_argument('--frozen-contract', type=Path)
    p.add_argument('--endpoint', default='http://127.0.0.1:11434')
    a = p.parse_args()
    questions = load_question_bank(a.bank, require_min=False)
    frozen = json.loads(a.frozen_contract.read_text()) if a.frozen_contract else None
    guard_partition(questions, a.partition, frozen)
    chosen = {}
    for q in questions:
        chosen.setdefault((q.family, q.metadata['requested_size']), q)
    if len(chosen) != 12:
        raise ValueError('need all four families and three sizes')
    a.out.mkdir(parents=True, exist_ok=False)
    records, tables = [], {}
    deadline = time.monotonic() + 3600
    for q in chosen.values():
        if time.monotonic() > deadline:
            raise TimeoutError('bounded planner smoke exceeded one-hour dispatch deadline')
        request = {'model': a.model, 'stream': False, 'format': question_schema(q.text, a.sites),
                   'messages': [{'role': 'user', 'content': planner_prompt(q.text, a.sites)}],
                   'options': {'temperature': 0, 'seed': 20261027, 'num_predict': 1024, 'num_ctx': 8192}}
        record = {'question_id': q.question_id, 'attempts': [], 'start_epoch': time.time(),
                  'valid_plan': False, 'reference_plan_match': False, 'tool_results_match': False}
        for _ in range(3):
            attempt = {'request': json.loads(json.dumps(request)), 'start_epoch': time.time()}
            try:
                req = urllib.request.Request(a.endpoint.rstrip('/') + '/api/chat', data=json.dumps(request).encode(),
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=180) as response:
                    attempt['response'] = json.loads(response.read())
                plan = validate_plan(json.loads(attempt['response']['message']['content']), a.sites, q.text)
                attempt['valid'] = True
                record.update(valid_plan=True, reference_plan_match=plan == {'family': q.family, 'calls': q.plan[0]})
            except Exception as exc:
                attempt.update(valid=False, error=repr(exc))
                request['messages'].append({'role': 'user', 'content': 'Validation error: ' + str(exc) +
                                            '. Recheck the analyst request and return corrected JSON.'})
            attempt['end_epoch'] = time.time()
            record['attempts'].append(attempt)
            if attempt['valid']:
                # Execute the actual accepted model plan; compare references only afterward.
                actual, expected = {}, {}
                for c in plan['calls']:
                    site = c['site']
                    if site not in tables:
                        tables[site] = site_table(a.store, a.sites, site)
                    result, count = run_query(tables[site], plan['family'], c['params'])
                    actual[site] = {'result': result, 'matching_rows': count}
                for c in q.plan[0]:
                    result, count = run_query(tables[c['site']], q.family, c['params'])
                    expected[c['site']] = {'result': result, 'matching_rows': count}
                record.update(actual_plan=plan, actual_tools=actual, tool_results_match=actual == expected)
                break
        record['end_epoch'] = time.time()
        records.append(record)
        (a.out / 'records.json').write_text(json.dumps(records, indent=2))
    summary = {'partition': a.partition, 'source_contract': source_contract(), 'model': a.model,
               'bank_sha256': file_sha256(a.bank), 'n_sites': a.sites, 'questions': len(records),
               'valid_plans': sum(r['valid_plan'] for r in records),
               'reference_plan_matches': sum(r['reference_plan_match'] for r in records),
               'tool_result_matches': sum(r['tool_results_match'] for r in records), 'fallback_used': False,
               'scope': '12-question one-round planning/tool smoke on declared partition; no free-answer agent or final-test utility',
               'test_executed': False}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    if summary['reference_plan_matches'] != 12 or summary['tool_result_matches'] != 12:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
