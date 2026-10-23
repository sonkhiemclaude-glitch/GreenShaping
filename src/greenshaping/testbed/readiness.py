"""Fail-closed software gates for requesting a bounded cloud calibration only."""
from __future__ import annotations

from .higgs_workload import SOURCE_HASH


def cloud_calibration_gates(data, test_exit, stability_exits, transports, planners, artifacts_verified):
    gates = {
        'source_verified': data.get('source_sha256') == SOURCE_HASH and data.get('rows') == 11000000,
        'all_N_train_select_sizes': data.get('reference_preparation_pass') is True,
        'full_mac_regression': test_exit == 0,
        'repeated_transport_stability': len(stability_exits) == 3 and all(v == 0 for v in stability_exits),
        'cloud_bundle_verified': artifacts_verified is True,
    }
    for name, count in [('separate', 360), ('routed', 180), ('portable_roles', 60)]:
        summary = transports.get(name, {})
        gates['tls_' + name] = (summary.get('all_passed') is True and summary.get('sessions') == count and
                               summary.get('passed') == count and summary.get('final_test_executed') is False)
        if name != 'portable_roles':
            gates['tls_' + name] &= {v.get('N') for v in summary.get('checks', [])} == {5, 10, 20}
        else:
            gates['tls_' + name] &= summary.get('N') == 20 and summary.get('roles_in_separate_processes') is True
    for phase in ('train', 'select'):
        for n in (5, 10, 20):
            for model in ('mistral', 'qwen14b'):
                key = f'{phase}/n{n}/{model}'
                s = planners.get(key, {})
                gates['planner/' + key] = (s.get('partition') == phase and s.get('n_sites') == n and
                                           s.get('questions') == s.get('valid_plans') ==
                                           s.get('reference_plan_matches') == s.get('tool_result_matches') == 12 and
                                           s.get('fallback_used') is False and s.get('test_executed') is False)
    failed = [k for k, v in gates.items() if not v]
    return {'ready_for_ec2_calibration': not failed, 'ready_for_main_paper_run': False, 'gates': gates,
            'failed_gates': failed, 'next_action': 'user_start_T1_T4_for_bounded_calibration_only' if not failed else
            'keep_EC2_stopped_and_resolve_findings',
            'scientific_pending': ['matched physical NetShaper captures', 'physical per-node paired energy',
                                   'full agent utility and frozen controller/attackers', 'fresh final-test acquisition']}
