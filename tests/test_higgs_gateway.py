import pandas as pd
import pytest

from greenshaping.testbed.higgs_gateway import ROUTE_KEY, RoutedTool, routed_calls


def test_routes_actual_site_and_keeps_reference_unchanged():
    tables = {0: pd.DataFrame({'x': [1, 2]}), 1: pd.DataFrame({'x': [3, 4, 5]})}
    gateway = RoutedTool(set(tables), tables.__getitem__)
    calls = [{'site': 1, 'params': {'where': {'x': {'min': 4, 'max': 5}}}}]
    routed = routed_calls(calls)
    assert gateway('aggregate', {**routed[0]['params'], 'columns': ['x']}) == (
        {'x': {'count': 2, 'sum': 9., 'sumsq': 41.}}, 2)
    assert ROUTE_KEY not in calls[0]['params']
    assert set(gateway.tables) == {1}


@pytest.mark.parametrize('site', [None, True, -1, 20, '0'])
def test_rejects_invalid_routes(site):
    gateway = RoutedTool({0}, lambda _: pd.DataFrame({'x': [1]}))
    with pytest.raises(ValueError):
        gateway('record_count', {ROUTE_KEY: site})


def test_forbids_overriding_route_in_model_arguments():
    with pytest.raises(ValueError):
        routed_calls([{'site': 0, 'params': {ROUTE_KEY: 1}}])
