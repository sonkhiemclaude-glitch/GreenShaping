import importlib.util
from pathlib import Path

import pytest

from greenshaping.testbed.question_bank import Question

spec = importlib.util.spec_from_file_location('higgs_runner', Path(__file__).parents[1] / 'scripts/validate_higgs_planner.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def bank(split):
    return [Question(1, 'record_count', 'site 0', metadata={'partition': split})]


def test_never_acquires_final_test():
    with pytest.raises(ValueError, match='never test'):
        module.guard_partition(bank('test'), 'test', {'source_contract': module.source_contract()})
    with pytest.raises(ValueError):
        module.guard_partition(bank('test'), 'train', None)


def test_select_requires_unchanged_contract():
    module.guard_partition(bank('train'), 'train', None)
    with pytest.raises(ValueError, match='frozen'):
        module.guard_partition(bank('select'), 'select', None)
    with pytest.raises(ValueError, match='frozen'):
        module.guard_partition(bank('select'), 'select', {'source_contract': {'planner_sha256': 'changed'}})
    module.guard_partition(bank('select'), 'select', {'source_contract': module.source_contract()})
