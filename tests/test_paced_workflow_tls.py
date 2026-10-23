import subprocess
import time

import pytest

from greenshaping.controller.workflow_contract import WorkflowEnvelope
from greenshaping.testbed.paced_workflow_tls import PacedTLSSite, run_paced_tls, slots


@pytest.mark.parametrize('policy', ['envelope', 'periodic', 'periodic_6'])
def test_matched_feedback_workflows_and_cover_survive_private_finish(tmp_path, policy):
    cert, key = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                    '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
    e = WorkflowEnvelope(3, .04, .08, .04, (128,) * 3, (256,) * 3)
    sites = [PacedTLSSite(e, policy, i, cert, key, lambda v: v * 2) for i in range(3)]

    def planner(r, prior):
        return [2, None, None] if r == 0 else [None, prior[0], None]

    result = run_paced_tls(e, policy, [s.address for s in sites], cert, planner, 2)
    for site in sites:
        site.close()
    assert result['completed'] and not any(s.errors for s in sites)
    assert result['responses'] == [{0: 4}, {1: 8}]
    count = len(slots(e, policy, 'up'))
    assert len(result['client_logs']) == 3 * count
    assert sum(len(s.logs) for s in sites) == 3 * count
    assert result['application_bytes'] == count * 3 * (128 + 256)
    assert result['tool_completion_s'] < e.horizon + .04


@pytest.mark.parametrize('policy', ['envelope', 'periodic'])
def test_bound_failure_keeps_all_frames_and_never_counts_as_success(tmp_path, policy):
    cert, key = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                    '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
    e = WorkflowEnvelope(1, .04, .04, .04, (128,), (256,))

    def slow(value):
        time.sleep(.3)
        return value

    site = PacedTLSSite(e, policy, 0, cert, key, slow)
    result = run_paced_tls(e, policy, [site.address], cert, lambda r, p: [2], 1)
    site.close()
    assert not result['completed']
    assert 'compute_bound' in site.errors
    assert len(site.logs) == len(slots(e, policy, 'down'))
    assert len(result['client_logs']) == len(slots(e, policy, 'up'))
