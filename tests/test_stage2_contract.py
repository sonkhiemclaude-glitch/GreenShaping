import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from mac_stage2_calibration import configurations, input_contract


def test_fixed_session_budget_cap_and_common_multiround_horizon():
    for n in [5,10,20]:
        configs=configurations(n)
        assert configs['fixed_1024'].fixed_unit==1024 and configs['minimal_tls'] is None
        for c in configs.values():
            if c and c.dp:
                assert c.dp.eps_dir*2*n==8
                assert c.dp.delta>=2*c.dp.volume_cap_bytes
                assert c.dp.horizon==10 and c.dp.horizon/c.dp.interval==20


def test_input_contract_locks_train_select_and_does_not_open_test(tmp_path):
    (tmp_path/'banks/n20').mkdir(parents=True);(tmp_path/'store').mkdir()
    for name in ['train','select']:(tmp_path/'banks/n20'/(name+'.jsonl')).write_text(name)
    (tmp_path/'store/projection.json').write_text('{}')
    first=input_contract(tmp_path,20)
    # A test bank need not even exist; its contents are not read for selection.
    (tmp_path/'banks/n20/test.jsonl').write_text('must remain unused')
    assert first==input_contract(tmp_path,20)
    (tmp_path/'banks/n20/select.jsonl').write_text('changed')
    assert first!=input_contract(tmp_path,20)
