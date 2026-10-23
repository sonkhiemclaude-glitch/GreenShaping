import importlib.util
import json
import sys
from pathlib import Path


def test_matched_plan_covers_all_modes_at_common_tcp_udp_observer_without_ssh(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('matched', scripts / 'run_ec2_matched_capture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args, **kwargs):
        raise AssertionError('offline planning must not contact EC2')
    for name in ['python', 'copy', 'run']:
        monkeypatch.setattr(module.r, name, forbidden)
    monkeypatch.setattr(sys, 'argv', ['matched', '--plan-only', '--out', str(tmp_path / 'plan')])
    module.main()
    plan = json.loads((tmp_path / 'plan/plan.json').read_text())
    assert len(set(plan['modes'])) == 7 and plan['planned_sessions'] == 168
    assert 'tcp or udp' in plan['observer_filter'] and '172.31.38.41' in plan['observer_filter']
    assert plan['physical_data_sites'] == 1 and not plan['final_test_executed']
    cfg = module.configs('netshaper_shaping')
    assert cfg['T2']['maxClients'] == cfg['T3']['maxStreamsPerPeer'] == 20
