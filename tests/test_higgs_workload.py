from pathlib import Path

import numpy as np
import pandas as pd

from greenshaping.testbed.higgs_workload import make_questions, prepare_projection, site_table
from greenshaping.testbed.queries import run_query


def test_global_cohorts_and_unique_sites(tmp_path: Path):
    source = tmp_path / 'fixture.csv.gz'
    rng = np.random.default_rng(4)
    raw = np.c_[rng.integers(0, 2, 2000), rng.random((2000, 2))]
    pd.DataFrame(raw).to_csv(source, header=False, index=False, compression='gzip')
    root = tmp_path / 'store'
    prepare_projection(source, root, rows=2000)
    global_cohorts = []
    for n in (5, 10, 20):
        tables = [site_table(root, n, s) for s in range(n)]
        all_rows = pd.concat(tables)
        assert all_rows.source_row_id.nunique() == 2000
        assert len(all_rows) == 2000
        global_cohorts.append(all_rows.set_index('source_row_id').cohort.sort_index())
        assert all(len(t) == 2000 // n for t in tables)
    for cohorts in global_cohorts[1:]:
        pd.testing.assert_series_equal(global_cohorts[0], cohorts)
    questions = make_questions(root, 20)
    assert len(questions) == 480
    assert len({q.question_id for q in questions}) == 480
    for q in questions:
        assert not {'limit_rows', 'row_offset'} & q.plan[0][0]['params'].keys()
        if q.metadata['partition'] == 'test':
            continue
        for c in q.plan[0][:1]:
            result, rows = run_query(site_table(root, 20, c['site']), q.family, c['params'])
            assert isinstance(result, dict) and 0 <= rows <= 100
    train = set(pd.concat([site_table(root, 5, i) for i in range(5)]).query('cohort < 3').source_row_id)
    select = set(pd.concat([site_table(root, 5, i) for i in range(5)]).query('cohort == 3').source_row_id)
    assert train.isdisjoint(select)
