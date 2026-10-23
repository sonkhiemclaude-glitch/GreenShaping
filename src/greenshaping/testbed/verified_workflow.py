"""Explicit derived TRAIN/select workflows; no held-out-test execution.

Three training rounds are a declared extension of the original one-step bank.
The orchestration rule is deterministic, not a claim of autonomous LLM reasoning.
"""
import copy
import math


def merge(family, results, steps=1):
    if not results or any(r['matching_rows'] < 0 for r in results):
        raise ValueError('nonempty valid tool results required')
    values = [r['result'] for r in results]
    if family == 'record_count':
        return {'count': sum(v['count'] for v in values)}
    if family in ('aggregate', 'group_by'):
        keys = sorted(set().union(*(v.keys() for v in values)))
        answer = {}
        for key in keys:
            fields = sorted(set().union(*(v.get(key, {}).keys() for v in values)))
            answer[key] = {f: sum(v.get(key, {}).get(f, 0) for v in values) for f in fields}
        return answer
    if family == 'train_round':
        n = sum(v['n'] for v in values)
        width = len(values[0]['weights'])
        if n <= 0 or any(len(v['weights']) != width for v in values):
            raise ValueError('training requires nonempty compatible contributions')
        weights = [sum(v['n'] * v['weights'][i] for v in values) / n for i in range(width)]
        if not all(math.isfinite(w) for w in weights):
            raise ValueError('nonfinite model update')
        return {'weights': weights, 'n': n, 'steps': steps}
    raise ValueError('unsupported family')


def execute(question, call_round, training_steps=3):
    if question.metadata.get('partition') not in ('train', 'select'):
        raise ValueError('only train/select; final test is locked')
    if type(training_steps) is not int or not 1 <= training_steps <= 3 or len(question.plan) != 1:
        raise ValueError('declared one-round source and one to three training steps required')
    calls = copy.deepcopy(question.plan[0])
    rounds = []
    count = training_steps if question.family == 'train_round' else 1
    for i in range(count):
        response = call_round(question.family, calls)
        if len(response) != len(calls):
            raise ValueError('one response required per called site')
        answer = merge(question.family, response, i + 1)
        rounds.append({'calls': copy.deepcopy(calls), 'responses': response, 'answer': answer})
        if question.family == 'train_round':
            for c in calls:
                c['params']['weights'] = answer['weights'][:]
    return {'answer': answer, 'rounds': rounds, 'workflow_variant': 'higgs-derived-training-3steps-v1' if count==3 else 'higgs-one-step-v1'}
