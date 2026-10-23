import subprocess
import time
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from greenshaping.attack.evaluation import evaluate_attacker, split_by_question
from greenshaping.data.schemas import QUERY_FAMILIES, Session
from greenshaping.energy.measurements import attach_measurements
from greenshaping.shaping.pipeline import DPParams, ShapingConfig, apply_shaping, resolve_delta
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.transport import RecordDecoder, ShapedTransport, TLSToolServer, encode_record
from greenshaping.testbed.workload_v2 import build_workload


def test_framing_every_split_and_padding():
    expected = [{'blob': '\u0000\u0001hello'}, {'n': 17}]
    data = encode_record(expected[0], 200) + encode_record(expected[1], 300)
    for size in range(1, 75):
        decoder = RecordDecoder()
        got = []
        for start in range(0, len(data), size):
            got.extend(decoder.feed(data[start:start + size]))
        assert got == expected


def test_pairwise_volume_cap_failure(session_factory):
    s = session_factory([100, 0], [400, 0])
    cfg = resolve_delta(ShapingConfig(dp=DPParams(volume_cap_bytes=200)), [s])
    assert cfg.dp.delta == 400
    _, stats = apply_shaping(s, cfg, np.random.default_rng(1))
    assert stats.privacy_failure  # volume 400 is inside Delta but outside V!
    with pytest.raises(ValueError):
        resolve_delta(ShapingConfig(dp=DPParams(volume_cap_bytes=300, delta=400)), [s])
    assert ShapingConfig(dp=DPParams(delta=400)).config_id() != ShapingConfig(dp=DPParams(delta=600)).config_id()


def test_energy_join_does_not_pool_n_or_repeats():
    identity = {'run_id': 'r', 'deployment_id': 'd', 'N': 5, 'llm': 'm', 'query_mix': 'all', 'config_id': 'c'}
    row = identity | {'node_class': 'per-manifest', 'node_classes': {'site0': 'arm'}}
    records = [identity | {'node': 'site0', 'node_class': 'arm', 'session_id': i, 'repetition': rep,
                           'extra_joules': value, 'reference': 'paired_unshaped'}
               for rep, i, value in [(0, 1, 1), (0, 2, 3), (1, 3, 10)]]
    records += [records[0] | {'N': 10, 'extra_joules': 999}]
    attach_measurements([row], records)
    assert row['energy_measured_site_max_median_J'] == 6  # median(mean(1,3), mean(10))
    assert row['energy_measured_site_runs'] == 2
    with pytest.raises(ValueError, match='duplicate'):
        attach_measurements([row], records + [records[0]])


def test_balanced_question_bank_and_explicit_split(tmp_path):
    questions = build_workload(tmp_path / 'workload')
    assert len(questions) == 192
    sessions = []
    for q in questions:
        calls = {c['site']: c for c in q.plan[0]}
        part = [int(i in calls) for i in range(5)]
        buckets = [int(np.digitize(calls[i]['params']['limit_rows'], [1e4, 1e5])) if i in calls else -1 for i in range(5)]
        sessions.append(Session(q.question_id, q.question_id, 5, QUERY_FAMILIES.index(q.family), part, buckets,
                                meta=q.metadata))
    parts = split_by_question(sessions, seed=999)
    assert [len(x) for x in parts] == [96, 48, 48]
    for group in parts:
        for fam in range(4):
            subset = [s for s in group if s.query_family == fam]
            for site in range(5):
                assert np.mean([s.participation[site] for s in subset]) == .5
                counts = Counter(s.size_bucket[site] for s in subset if s.participation[site])
                assert len(set(counts.values())) == 1 and len(counts) == 3
    scores = evaluate_attacker(parts[0] + parts[1], parts[2], models=['identity'], chance_mode='site_conditioned')
    assert scores['participation']['adv'] == 0
    assert scores['size_bucket']['adv'] == 0
    df = pd.DataFrame({'x': range(20)})
    assert run_query(df, 'record_count', {'limit_rows': 5, 'row_offset': 3})[1] == 5


@pytest.fixture
def tls_sites(tmp_path):
    cert, key = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                    '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
    df = pd.DataFrame({'x': range(20)})
    servers = [TLSToolServer('127.0.0.1', 0, str(cert), str(key), lambda f, p: run_query(df, f, p)) for _ in range(2)]
    yield str(cert), servers
    for s in servers:
        s.stop()


def test_closed_tcp_probe_does_not_disable_tls_listener(tls_sites):
    import socket

    cert, servers = tls_sites
    for server in servers:
        with socket.create_connection(server.address, timeout=2):
            pass
    tr = ShapedTransport([s.address for s in servers], cert)
    try:
        tr.begin()
        replies = tr.call_round('record_count', [{'site': 0, 'params': {'limit_rows': 8}}])
        assert replies[0]['matching_rows'] == 8
        assert tr.finish()['measurement_valid']
        assert all(server.thread.is_alive() for server in servers)
    finally:
        tr.close()


def test_live_tls_fixed_ticks_all_links_and_reuse(tls_sites):
    cert, servers = tls_sites
    cfg = ShapingConfig(dp=DPParams(eps_dir=4, interval=.05, horizon=.5, delta=4096,
                                    volume_cap_bytes=2048, bias=10000, mode="arrivals"))
    tr = ShapedTransport([s.address for s in servers], cert, cfg)
    try:
        for _ in range(2):
            tr.begin()
            replies = tr.call_round('record_count', [{'site': 0, 'params': {'limit_rows': 8}}])
            assert replies[0]['matching_rows'] == 8
            stats = tr.finish()
            for side in stats['upstream']:
                assert len(side['ticks']) == 10
                assert side['overflow_bytes'] == 0
            # Uncalled site also emits every downstream tick.
            for server in servers:
                assert len(server.channels[0].logs) == 10
                assert not server.channels[0].error
            assert len(servers[0].channels) == 1  # connection reused, no per-session handshake
    finally:
        tr.close()


def test_live_volume_cap_never_flushes(tls_sites):
    cert, servers = tls_sites
    cfg = ShapingConfig(dp=DPParams(interval=.05, horizon=.15, delta=80, volume_cap_bytes=40))
    tr = ShapedTransport([s.address for s in servers], cert, cfg)
    try:
        tr.begin()
        with pytest.raises(RuntimeError, match='cap'):
            tr.call_round('record_count', [{'site': 0, 'params': {}}])
        stats = tr.finish()
        assert stats['upstream'][0]['cap_exceeded']
        for c in tr.channels:
            emitted = len(c.logs)
            time.sleep(.1)
            assert len(c.logs) == emitted == 3
    finally:
        tr.close()


def test_cnn_group_validation_no_question_leak():
    pytest.importorskip('torch')
    from greenshaping.attack.classifiers import TimeSliceCNN
    X = np.random.default_rng(3).normal(size=(24, 4, 12)).astype(np.float32)
    groups = np.repeat(np.arange(8), 3)
    y = np.tile([0, 1, 0], 8)
    clf = TimeSliceCNN(epochs=2, batch_size=8).fit(X, y, groups=groups)
    assert not set(clf.training_groups_) & set(clf.validation_groups_)
    tr = ~np.isin(groups, clf.validation_groups_)
    np.testing.assert_allclose(clf.mean_, X[tr].mean(axis=(0, 2), keepdims=True), rtol=1e-5, atol=1e-5)
    assert len(clf.history_) == 2
    assert clf.predict_proba(X).shape == (24, 2)


def test_online_graph_executes_actual_feedback(tls_sites, tmp_path):
    pytest.importorskip('langgraph')
    from greenshaping.testbed.orchestrator import LangGraphOrchestrator
    from greenshaping.testbed.question_bank import Question
    cert, servers = tls_sites
    tr = ShapedTransport([s.address for s in servers], cert)

    class FakeModel(LangGraphOrchestrator):
        def _chat(self, prompt, json_mode):
            if json_mode:
                return '{"sites":[0]}'
            assert 'matching_rows' in prompt and '8' in prompt
            return 'Eight matching rows.'

    orch = FakeModel('unit-test', tr, str(tmp_path / 'evidence'))
    try:
        q = Question(1, 'record_count', 'Count first eight rows at site 0', [[{'site': 0, 'params': {'limit_rows': 8}}]])
        assert orch.run(q, [s.address for s in servers]) == {0: 8}
        assert orch.last_window[1] >= orch.last_window[0]
        assert list((tmp_path / 'evidence').glob('*.json'))
    finally:
        tr.close()


def test_live_rounds_equal_active_frames_and_continuation(tls_sites):
    from greenshaping.shaping.pipeline import RoundsParams
    cert, servers = tls_sites
    cfg = ShapingConfig(rounds=RoundsParams(window=.03, cover_prob=1., reply_frame=512))
    tr = ShapedTransport([s.address for s in servers], cert, cfg)
    try:
        tr.begin(called_sites=[0])
        response = tr.call_round('group_by', [{'site': 0, 'params': {'by': 'x', 'column': 'x'}}])[0]
        assert response['matching_rows'] == 20 and len(response['result']) == 20
        # A second causal round uses the actual previous result.
        second = tr.call_round('record_count', [{'site': 0, 'params': {'limit_rows': response['matching_rows'] // 2}}])
        assert second[0]['matching_rows'] == 10
        tr.finish()
        upstream = [[(v['application_bytes'], v['kind']) for v in c.logs] for c in tr.channels]
        downstream = [[(v['application_bytes'], v['kind']) for v in s.channels[0].logs] for s in servers]
        assert upstream[0] == upstream[1]
        assert downstream[0] == downstream[1]
        assert any(kind == 'round_poll' for _, kind in upstream[0])
        assert all(size == 512 for size, _ in downstream[0])
    finally:
        tr.close()


def test_live_composition_rounds_padding_dp(tls_sites):
    from greenshaping.shaping.pipeline import PaddingParams, RoundsParams
    cert, servers = tls_sites
    cfg = ShapingConfig(rounds=RoundsParams(window=.05, cover_prob=1., reply_frame=512), padding=PaddingParams(),
                        dp=DPParams(mode='arrivals', interval=.05, horizon=.8, volume_cap_bytes=4096,
                                    delta=8192, bias=30000))
    tr = ShapedTransport([s.address for s in servers], cert, cfg)
    try:
        tr.begin(called_sites=[0])
        assert tr.call_round('record_count', [{'site': 0, 'params': {'limit_rows': 7}}])[0]['matching_rows'] == 7
        stats = tr.finish()
        assert all(len(c['ticks']) == 16 and not c['cap_exceeded'] for c in stats['upstream'])
        assert all(not s.channels[0].cap_exceeded and not s.channels[0].pending for s in servers)
    finally:
        tr.close()


def test_output_identity_guard_precedes_mutation(tmp_path, tiny_config):
    from greenshaping.experiments.common import context_from_config, write_outputs
    ctx = context_from_config(tiny_config, tmp_path / 'out')
    path = write_outputs(ctx, 'evidence', [{'value': 1}])
    original = path.read_bytes()
    tiny_config['run']['seed'] += 1
    with pytest.raises(ValueError, match='different code/config/inputs'):
        context_from_config(tiny_config, tmp_path / 'out')
    assert path.read_bytes() == original


def test_cost_model_requires_identifiable_benchmarks_and_heldout_runs():
    from greenshaping.energy.cost_model import CostModel, union_length
    from greenshaping.energy.fit import Benchmark, fit_cost_model, validate_cost_model
    with pytest.raises(ValueError, match='cannot identify'):
        fit_cost_model('x', [Benchmark(100, 100, (0.,), 1.) for _ in range(20)])
    rng = np.random.default_rng(4)
    true = CostModel('x', 3e-8, 2e-8, 4., .05)
    rows = []
    for run in range(20):
        tx, rx = map(int, rng.integers(1e4, 1e7, size=2))
        times = tuple(sorted(rng.uniform(0, 10, size=int(rng.integers(1, 60)))))
        energy = true.e_tx * tx + true.e_rx * rx + true.p_aw * union_length(list(times), true.t_tail)
        rows.append(Benchmark(tx, rx, times, energy, str(run)))
    validation = validate_cost_model(rows, 'x')
    assert not set(validation['training_runs']) & set(validation['validation_runs'])
    assert validation['heldout_relative_rmse'] < 1e-5 and validation['design_rank'] == 3


def test_e5_infeasible_alpha_is_not_a_floor(tmp_path, tiny_config):
    from greenshaping.experiments.common import context_from_config
    from greenshaping.experiments.e5_scaling import run_e5
    tiny_config.setdefault('experiments', {})['e5'] = {'alpha': .1}
    ctx = context_from_config(tiny_config, tmp_path)
    rows = run_e5(ctx)
    assert all(not r['target_alpha_valid'] and not r['target_alpha_met'] for r in rows)
    assert all(np.isnan(r['prop2_network_floor_modelled_J']) for r in rows)
    assert all(np.isnan(r['e_min_cover_modelled_J']) for r in rows)


def test_partial_truncation_does_not_corrupt_next_session(tls_sites, monkeypatch):
    import greenshaping.testbed.transport as transport_module
    cert, servers = tls_sites

    class NegativeNoise:
        def laplace(self, *_):
            return -100.

    cfg = ShapingConfig(dp=DPParams(mode='queue', interval=.05, horizon=.15, delta=4096, volume_cap_bytes=2048))
    tr = ShapedTransport([s.address for s in servers], cert, cfg)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(transport_module.np.random, 'default_rng', lambda: NegativeNoise())
            tr.begin()
            with pytest.raises(TimeoutError, match='horizon'):
                tr.call_round('record_count', [{'site': 0, 'params': {'limit_rows': 8, 'unused': 'x' * 120}}])
            assert tr.finish()['upstream'][0]['overflow_bytes'] > 0
        tr.cfg = ShapingConfig(dp=DPParams(mode='arrivals', interval=.05, horizon=.3, delta=4096,
                                           volume_cap_bytes=2048, bias=100000))
        tr.begin()
        assert tr.call_round('record_count', [{'site': 0, 'params': {'limit_rows': 7}}])[0]['matching_rows'] == 7
        tr.finish()
        assert all(not c.error for c in tr.channels)
    finally:
        tr.close()


def test_paired_energy_subtracts_matching_idle_then_unshaped():
    from greenshaping.energy.paired import pair_energy_records
    shared = {'run_id': 'r', 'deployment_id': 'd', 'N': 5, 'llm': 'm', 'query_mix': 'all', 'config_id': 'c',
              'node': 'site0', 'node_class': 'arm', 'session_id': 'logical-trial-1', 'repetition': 0,
              'sensor': 'meter-1', 'scope': 'whole-device'}
    raw = [shared | {'condition': 'shaped', 'energy_joules': 20., 'idle_joules': 8., 'acquisition_id': 's'},
           shared | {'condition': 'unshaped', 'energy_joules': 13., 'idle_joules': 6., 'acquisition_id': 'u'}]
    assert pair_energy_records(raw)[0]['extra_joules'] == 5.
    with pytest.raises(ValueError, match='missing'):
        pair_energy_records(raw[:1])


def test_plain_finish_waits_for_server_write_accounting(tls_sites, monkeypatch):
    import threading

    from greenshaping.testbed.transport import WIRE_DATA

    cert, servers = tls_sites
    tr = ShapedTransport([s.address for s in servers], cert)
    release = threading.Event()
    try:
        tr.begin()
        channel = servers[0].channels[0]
        send = channel._send_wire

        def delayed(data, **kwargs):
            result = send(data, **kwargs)
            if kwargs.get('kind', WIRE_DATA) == WIRE_DATA:
                release.wait(timeout=5)  # Peer has the reply before the application send log is committed.
            return result

        monkeypatch.setattr(channel, '_send_wire', delayed)
        result = tr.call_round('record_count', [{'site': 0, 'params': {'limit_rows': 8}}])
        assert result[0]['matching_rows'] == 8
        assert channel.plain_sends == 1
        threading.Timer(.1, release.set).start()
        stats = tr.finish()
        assert stats['completion_ok'] and stats['measurement_valid']
        assert channel.plain_sends == 0
        assert len(channel.logs) == 1
        assert set(stats['downstream']) == {0, 1}
    finally:
        release.set()
        tr.close()
