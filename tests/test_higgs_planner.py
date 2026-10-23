import copy

import pytest

from greenshaping.testbed.higgs_planner import planner_prompt, validate_plan


@pytest.fixture
def plan():
    return {'family': 'aggregate', 'sites': [19, 0], 'params': {'where': {'cohort': {'min': 0, 'max': 2},
            'feature_01': {'min': -1000000, 'max': .25}}, 'columns': ['feature_02']}}


def test_compact_expansion_and_prompt(plan):
    result = validate_plan(plan, 20, 'At site 0, site 19, return sum.')
    assert [c['site'] for c in result['calls']] == [0, 19]
    assert all(c['params'] == plan['params'] for c in result['calls'])
    assert 'secret oracle' not in planner_prompt('Count records', 20)


@pytest.mark.parametrize('change', ['site', 'column', 'bounds', 'extra', 'nan', 'cohort'])
def test_reject_unsafe_arguments(plan, change):
    bad = copy.deepcopy(plan)
    if change == 'site':
        bad['sites'] = [True, 19]
    elif change == 'column':
        bad['params']['columns'] = ['source_row_id']
    elif change == 'bounds':
        bad['params']['where']['feature_01'] = {'min': 2, 'max': 1}
    elif change == 'extra':
        bad['params']['limit_rows'] = 100
    elif change == 'cohort':
        bad['params']['where']['cohort']['min'] = .5
    else:
        bad['params']['where']['feature_01']['max'] = float('nan')
    with pytest.raises(ValueError):
        validate_plan(bad, 20, 'At site 0, site 19')


def test_missing_requested_site(plan):
    with pytest.raises(ValueError):
        validate_plan(plan, 20, 'At site 0, site 1, site 19')


def test_decimal_strings_preserve_boundary_without_oracle_correction(plan):
    literal = '0.28366410732269287'
    plan['params']['where']['feature_01'] = {'min': '-1000000.0', 'max': literal}
    result = validate_plan(plan, 20, 'site 0, site 19')
    assert result['calls'][0]['params']['where']['feature_01']['max'] == float(literal)
    assert plan['params']['where']['feature_01']['max'] == literal
    shortened = copy.deepcopy(plan)
    shortened['params']['where']['feature_01']['max'] = '0.2836641073226928'
    actual = validate_plan(shortened, 20, 'site 0, site 19')
    assert actual != result  # Wrong model digits are never snapped to a reference.


@pytest.mark.parametrize('bad', ['NaN', 'Infinity', '1e999', '0.1 + 1'])
def test_reject_bad_decimal_string(plan, bad):
    plan['params']['where']['feature_01']['max'] = bad
    with pytest.raises(ValueError):
        validate_plan(plan, 20, 'site 0, site 19')


def test_public_question_schema_does_not_turn_predictors_into_filters():
    from greenshaping.testbed.higgs_planner import question_schema
    text = 'site 0; retain cohort values from 3 through 3; predict label from feature_01 and feature_02'
    for branch in question_schema(text, 20)['oneOf']:
        where = branch['properties']['params']['properties']['where']
        assert set(where['properties']) == {'cohort'}
        assert where['additionalProperties'] is False
    ranged = question_schema('feature_01 between -1000000.0 and 0.28366410732269287 inclusive', 20)
    for branch in ranged['oneOf']:
        where = branch['properties']['params']['properties']['where']
        assert set(where['required']) == {'cohort', 'feature_01'}
        assert where['properties']['feature_01']['properties']['max']['type'] == 'string'


def test_training_schema_and_feedback_include_intercept_dimension():
    from greenshaping.testbed.higgs_planner import planner_schema
    branch = next(b for b in planner_schema(20)['oneOf'] if b['properties']['family']['enum'] == ['train_round'])
    props = branch['properties']['params']['properties']
    assert props['features']['minItems'] == props['features']['maxItems'] == 2
    assert props['weights']['minItems'] == props['weights']['maxItems'] == 3
    bad = {'family': 'train_round', 'sites': [0], 'params': {'where': {'cohort': {'min': 3, 'max': 3}},
           'features': ['feature_01', 'feature_02'], 'label': 'label', 'weights': [0, 0], 'lr': .1}}
    with pytest.raises(ValueError, match='expected 3 weights'):
        validate_plan(bad, 20, 'site 0')
    assert bad['params']['weights'] == [0, 0]  # Never fill a missing weight automatically.
