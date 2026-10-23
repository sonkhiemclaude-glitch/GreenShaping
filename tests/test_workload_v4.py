from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenshaping.testbed.queries import run_query
from greenshaping.testbed.question_bank import Question
from greenshaping.testbed.workload_v4 import next_calls


def question(family: str) -> Question:
    return Question(100000, family, 'test', metadata={'adaptive_policy': {
        'min_rounds': 2, 'max_rounds': 4, 'tolerance': .04, 'target_count': 40000, 'target_mean': 104.}})


def test_federated_weights_use_actual_weighted_outputs_and_stop() -> None:
    q = question('train_round')
    calls = [{'site': i, 'params': {'weights': [0., 0., 0.]}} for i in (0, 1)]
    responses = [{'result': {'weights': [v, 0., 0.], 'n': n}} for v, n in ((.1, 100), (.3, 300))]
    following = next_calls(q, calls, responses, 1)
    np.testing.assert_allclose(following[0]['params']['weights'], [.25, 0, 0])
    assert calls[0]['params']['weights'] == [0., 0., 0.]
    stationary = [{'result': {'weights': [.25, 0., 0.], 'n': 100}}] * 2
    assert next_calls(q, following, stationary, 2) == []
    assert next_calls(q, calls, responses, 4) == []


def test_private_outputs_change_following_query() -> None:
    q = question('aggregate')
    calls = [{'site': 0, 'params': {'where': {'cohort': 1}, 'columns': ['value']}}]
    low = next_calls(q, calls, [{'result': {'value': {'sum': 1000., 'count': 10}}}], 1)
    high = next_calls(q, calls, [{'result': {'value': {'sum': 1080., 'count': 10}}}], 1)
    assert low[0]['params']['where']['flag'] == 0
    assert high[0]['params']['where']['flag'] == 1
    assert low != high
    assert 'flag' not in calls[0]['params']['where']


def test_private_range_filter_and_invalid_bounds() -> None:
    df = pd.DataFrame({'marker': [.1, .4, .8], 'cohort': [0, 0, 1]})
    result, rows = run_query(df, 'record_count', {'where': {'marker': {'min': .2, 'max': .9}, 'cohort': 0}})
    assert rows == result['count'] == 1
    with pytest.raises(ValueError, match='finite min'):
        run_query(df, 'record_count', {'where': {'marker': {'min': float('nan'), 'max': .9}}})
