from greenshaping.testbed.higgs_workload import SOURCE_HASH
from greenshaping.testbed.readiness import cloud_calibration_gates


def inputs():
    data = {'source_sha256': SOURCE_HASH, 'rows': 11000000, 'reference_preparation_pass': True}
    transports = {name: {'all_passed': True, 'sessions': count, 'passed': count,
                         'final_test_executed': False, 'checks': [{'N': n} for n in (5, 10, 20)]}
                  for name, count in [('separate', 360), ('routed', 180), ('portable_roles', 60)]}
    transports['portable_roles'].update(N=20, roles_in_separate_processes=True)
    planners = {f'{phase}/n{n}/{m}': {'partition': phase, 'n_sites': n, 'questions': 12, 'valid_plans': 12,
                                    'reference_plan_matches': 12, 'tool_result_matches': 12,
                                    'fallback_used': False, 'test_executed': False}
                for phase in ('train', 'select') for n in (5, 10, 20) for m in ('mistral', 'qwen14b')}
    return data, transports, planners


def test_software_success_is_never_main_paper_readiness():
    data, transports, planners = inputs()
    result = cloud_calibration_gates(data, 0, [0, 0, 0], transports, planners, True)
    assert result['ready_for_ec2_calibration']
    assert not result['ready_for_main_paper_run']


def test_missing_or_partial_results_never_request_restart():
    assert not cloud_calibration_gates({}, None, [], {}, {}, False)['ready_for_ec2_calibration']
    data, transports, planners = inputs()
    planners.pop('select/n20/qwen14b')
    result = cloud_calibration_gates(data, 0, [0, 0, 0], transports, planners, True)
    assert not result['ready_for_ec2_calibration']
    assert 'planner/select/n20/qwen14b' in result['failed_gates']


def test_any_invalid_trial_or_failed_stability_blocks_restart():
    data, transports, planners = inputs()
    transports['separate']['passed'] -= 1
    result = cloud_calibration_gates(data, 0, [0, 1, 0], transports, planners, True)
    assert set(result['failed_gates']) == {'tls_separate', 'repeated_transport_stability'}
