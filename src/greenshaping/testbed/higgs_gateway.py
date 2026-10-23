"""Explicit logical-site routing for a physical TLS endpoint shared by sites.

The adapter adds a public site ID inside each encrypted call. This does not
create independent physical hosts or physical-site energy observations.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .queries import run_query

ROUTE_KEY = '__logical_site'


def routed_calls(calls: list[dict]) -> list[dict]:
    out = []
    for call in calls:
        if ROUTE_KEY in call['params']:
            raise ValueError('reference/model arguments must not contain the reserved routing key')
        out.append({'site': call['site'], 'params': {**call['params'], ROUTE_KEY: call['site']}})
    return out


class RoutedTool:
    def __init__(self, allowed_sites: set[int], load: Callable[[int], pd.DataFrame]):
        self.allowed_sites, self.load = allowed_sites, load
        self.tables: dict[int, pd.DataFrame] = {}

    def __call__(self, family: str, params: dict[str, Any]) -> tuple[dict[str, Any], int]:
        params = dict(params)
        site = params.pop(ROUTE_KEY, None)
        if type(site) is not int or site not in self.allowed_sites:
            raise ValueError('logical site is missing or outside this physical endpoint assignment')
        if site not in self.tables:
            self.tables[site] = self.load(site)
        return run_query(self.tables[site], family, params)
