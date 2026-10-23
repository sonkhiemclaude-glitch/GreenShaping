import numpy as np
import pandas as pd
import pytest
from greenshaping.testbed.question_bank import Question
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.verified_workflow import execute, merge


def test_three_weighted_rounds_match_training_on_union_and_preserve_source():
    tables=[pd.DataFrame({'x':[.1,.7], 'y':[0,1]}),pd.DataFrame({'x':[1.,.4,-.2], 'y':[1,0,0]})]
    params=dict(features=['x'],label='y',weights=[0.,0.],lr=.1)
    q=Question(1,'train_round','declared derived 3 steps',[[{'site':s,'params':dict(params)} for s in range(2)]],{'partition':'train'})
    seen=[]
    def tools(f,calls):
        seen.append([c['params']['weights'][:] for c in calls])
        return [dict(zip(('result','matching_rows'),run_query(tables[c['site']],f,c['params']))) for c in calls]
    result=execute(q,tools)
    expected=params.copy()
    for _ in range(3):expected['weights']=run_query(pd.concat(tables),'train_round',expected)[0]['weights']
    assert np.allclose(result['answer']['weights'],expected['weights'])
    assert len(result['rounds'])==3 and seen[0]!=seen[1] and seen[1]!=seen[2]
    assert q.plan[0][0]['params']['weights']==[0.,0.]
    q.metadata['partition']='test'
    with pytest.raises(ValueError,match='locked'):execute(q,tools)


def test_merge_keeps_all_groups_and_empty_site_counts():
    rows=[{'matching_rows':2,'result':{'a':{'count':2,'sum':3.}}},
          {'matching_rows':3,'result':{'b':{'count':3,'sum':7.}}}]
    assert merge('group_by',rows)=={'a':{'count':2,'sum':3.},'b':{'count':3,'sum':7.}}
    with pytest.raises(ValueError):merge('record_count',[])
