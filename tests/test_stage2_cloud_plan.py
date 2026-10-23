import importlib.util
from pathlib import Path
import sys


def load(name):
    folder=Path(__file__).parents[1]/'scripts'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location(name,folder/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_cloud_plan_keeps_workflow_and_selected_configuration_contract():
    client=load('higgs_stage2_tls_endpoint');runner=load('run_ec2_stage2_capture')
    c=client.modes(20)
    assert c['dp'].dp.horizon==c['combo'].dp.horizon==10
    assert c['combo'].dp.bias==2*c['dp'].dp.bias
    p=runner.plan()
    assert p['workflow_steps']==3 and p['planned_sessions']==168
    assert p['physical_data_sites']==1 and not p['final_test_executed']
    assert c['dp'].dp.eps_dir==.2
