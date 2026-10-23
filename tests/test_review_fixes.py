"""Regression tests for the independent code review (audits/result_integrity_audit.md, "Review fixes")."""
from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pytest
from conftest import make_session

from greenshaping.attack.evaluation import evaluate_attacker, split_by_question
from greenshaping.attack.features import build_time_slice_tables
from greenshaping.controller.selection import Profile, is_feasible, mean_node_energy, profile_configuration
from greenshaping.data.adapters import data_kind_for_llm, export_sessions
from greenshaping.data.schemas import DOWN, UP, Message, Session
from greenshaping.data.synthetic import SyntheticConfig, generate_sessions
from greenshaping.energy.cost_model import CostModel
from greenshaping.experiments.common import load_sessions, merge_measured_energy
from greenshaping.shaping.dp_release import dp_release
from greenshaping.shaping.exchanges import pair_replies
from greenshaping.shaping.pipeline import DPParams, RoundsParams, ShapingConfig, calibrate_horizon
from greenshaping.shaping.rounds import synchronised_rounds
from greenshaping.testbed.framing import HEADER

EXACT = {"eps_dir": 1e12, "sensitivity": 10**6}   # vanishing noise: b_k = Q_k


def _one_site(msgs: list[Message]) -> Session:
    return Session(0, 0, 1, 0, [1], [0], msgs, 5.0)


# ---- B1: causality of the DP release across rounds ---------------------------------
def test_dp_release_keeps_dependent_call_after_delivered_reply():
    s = _one_site([Message(0.1, 0, UP, 200), Message(0.2, 0, DOWN, 200_000), Message(0.3, 0, UP, 300)])
    sh, st = dp_release(s, interval=1.0, horizon=6.0, rng=np.random.default_rng(0), **EXACT)
    up_real = [m.t for m in sh.messages if m.direction == UP and m.real]
    down_real = [m.t for m in sh.messages if m.direction == DOWN and m.real]
    assert down_real == [2.0]              # call delivered at tick 1, reply enters at 1.1 -> tick 2
    assert up_real == [1.0, 3.0]           # call 2 waits for the reply: 0.3 + (2.0 - 0.2) = 2.1 -> tick 3
    assert st.overflow_bytes == 0 and st.real_done == 2.0


def test_dp_release_no_up_byte_of_next_round_before_previous_round_delivered():
    sess = make_session([800, 600, 0], [30_000, 50_000, 0], n_sites=3, rounds=3)
    sh, st = dp_release(sess, interval=0.5, horizon=30.0, rng=np.random.default_rng(1), **EXACT)
    up_ticks = sorted({m.t for m in sh.messages if m.direction == UP and m.real})
    down_ticks = sorted({m.t for m in sh.messages if m.direction == DOWN and m.real})
    assert st.overflow_bytes == 0
    # three rounds: every real UP release after the first comes after some real DOWN release
    assert len(up_ticks) >= 3
    for t in up_ticks[1:]:
        assert any(d < t for d in down_ticks)
    assert up_ticks[1] > max(d for d in down_ticks if d < up_ticks[1])


def test_dp_overflow_and_truncated_latency():
    s = _one_site([Message(0.1, 0, UP, 200), Message(25.0, 0, DOWN, 900)])
    _, st = dp_release(s, interval=1.0, horizon=20.0, rng=np.random.default_rng(0), **EXACT)
    assert st.overflow_bytes == 900 and st.privacy_failure and st.real_truncated and st.real_done == 20.0


# ---- M1: replies paired by time ----------------------------------------------------
def test_pairing_by_time_and_unpaired_bytes():
    msgs = [Message(0.0, 0, DOWN, 7), Message(0.1, 0, UP, 300), Message(0.2, 0, DOWN, 40_000),
            Message(0.3, 0, DOWN, 40_000), Message(1.0, 0, UP, 300), Message(1.1, 0, DOWN, 500)]
    parent, unpaired = pair_replies(msgs)
    assert parent == {0: None, 2: 1, 3: 1, 5: 4} and unpaired == 7


def test_rounds_keep_split_reply_bytes_in_their_round():
    msgs = [Message(0.1, 0, UP, 300), Message(0.2, 0, DOWN, 40_000), Message(0.35, 0, DOWN, 40_000),
            Message(1.0, 0, UP, 300), Message(1.1, 0, DOWN, 500)]
    sh, st = synchronised_rounds(_one_site(msgs), 0.5, 1.0, 16384, rng=np.random.default_rng(0))
    payload = 16384 - HEADER.size
    carrying = [m for m in sh.messages if m.direction == DOWN and m.real]
    calls = sorted(m.t for m in sh.messages if m.direction == UP and m.real)
    assert len(calls) == 2
    first = [m for m in carrying if m.t < calls[1]]
    second = [m for m in carrying if m.t > calls[1]]
    assert len(first) == math.ceil(80_000 / payload) and len(second) == 1   # nothing dropped or shifted
    assert st.unpaired_bytes == 0


def test_rounds_accept_calls_without_reply():
    msgs = [Message(0.1, 0, UP, 300), Message(0.12, 0, UP, 300), Message(0.2, 0, DOWN, 900)]
    sh, _ = synchronised_rounds(_one_site(msgs), 0.5, 1.0, 4096, rng=np.random.default_rng(0))
    assert any(m.real and m.direction == DOWN for m in sh.messages)


def test_round_frame_not_before_last_grouped_call():
    msgs = [Message(0.45, 0, UP, 300), Message(0.52, 0, UP, 300), Message(0.6, 0, DOWN, 900)]
    sh, _ = synchronised_rounds(_one_site(msgs), 0.5, 1.0, 4096, rng=np.random.default_rng(0))
    assert min(m.t for m in sh.messages if m.direction == UP) >= 0.52


# ---- B2: one deployment size per Algorithm 1 run -------------------------------------
def test_mixed_n_is_refused_and_node_means_use_only_present_nodes():
    a = generate_sessions(SyntheticConfig(n_sessions=30, n_sites=4, n_questions=12, seed=1))
    b = generate_sessions(SyntheticConfig(n_sessions=30, n_sites=6, n_questions=12, seed=2))
    for s in b:
        s.question_id += 100
    with pytest.raises(ValueError, match="one N"):
        tr = [s for s in a + b if s.question_id % 100 < 6]
        se = [s for s in a + b if s.question_id % 100 >= 6]
        profile_configuration(ShapingConfig(), tr, se, CostModel("x", 1e-8, 1e-8, 1.0, 0.1))
    m = mean_node_energy([{"site0": 1.0, "site4": 3.0}, {"site0": 2.0}])
    assert m == {"site0": 1.5, "site4": 3.0}
    p = Profile(ShapingConfig(), False, 0.0, {"site0": 1.0, "orchestrator": 1.0})
    with pytest.raises(KeyError):
        is_feasible(p, {"site0": 2.0, "site5": 2.0})


# ---- M7: data kinds ------------------------------------------------------------------
def test_data_kind_labels_and_mixed_jsonl_refused(tmp_path):
    assert data_kind_for_llm("none (scripted plan; pipeline validation only)") == "rehearsal"
    assert data_kind_for_llm("llama-3.1-8b-instruct") == "real"
    ss = generate_sessions(SyntheticConfig(n_sessions=6, n_sites=3, n_questions=6, seed=0))
    ss[0].meta["data_kind"] = "real"
    export_sessions(ss, tmp_path / "s.jsonl")
    with pytest.raises(ValueError, match="data_kind"):
        load_sessions({"data": {"source": "jsonl", "sessions_jsonl": str(tmp_path / "s.jsonl")}})


# ---- M6: CNN wiring ----------------------------------------------------------------------
def test_time_slice_tables_and_cnn_selection():
    ss = generate_sessions(SyntheticConfig(n_sessions=12, n_sites=3, n_questions=6, seed=0))
    L, S = build_time_slice_tables(ss, 0.25, 40)
    assert L["X"].shape == (36, 4, 40) and S["X"].shape == (12, 4, 40)
    tr, te = [s for s in ss if s.question_id < 3], [s for s in ss if s.question_id >= 3]
    with pytest.raises(ValueError):
        evaluate_attacker(tr, te, models=("svm",))
    res = evaluate_attacker(tr, te, n_estimators=5, models=("rf",))
    assert res["participation"]["attacker"] == "rf" and "adv_raw_rf" in res["participation"]
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(ImportError):
            evaluate_attacker(tr, te, models=("rf", "cnn"))


# ---- minors ------------------------------------------------------------------------------
def test_split_is_stratified_by_family():
    ss = generate_sessions(SyntheticConfig(n_sessions=200, n_sites=3, n_questions=40, seed=3))
    parts = split_by_question(ss, (0.4, 0.3, 0.3), 0)
    fams = {s.query_family for s in ss}
    for p in parts:
        assert {s.query_family for s in p} == fams


def test_labels_distinguish_all_parameters_and_horizon_calibration():
    a = ShapingConfig(rounds=RoundsParams(poll_size=0)).label()
    b = ShapingConfig(rounds=RoundsParams()).label()
    c = ShapingConfig(dp=DPParams(bias=100.0)).label()
    d = ShapingConfig(dp=DPParams()).label()
    assert a != b and c != d
    ss = generate_sessions(SyntheticConfig(n_sessions=20, n_sites=3, n_questions=10, seed=0))
    h = calibrate_horizon(ss, ShapingConfig(dp=DPParams(interval=0.5)), 99.0)
    assert h > 0 and abs(h / 0.5 - round(h / 0.5)) < 1e-9


def test_measured_energy_respects_node_class(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("config_label,node,run,extra_joules,node_class\n"
                 "none,site0,0,1.0,x86_server\nnone,site0,1,1.0,x86_server\n"
                 "none,site0,0,9.0,raspberry_pi_5\nnone,orchestrator,0,0.5,x86_server\n")
    rows = [{"config": "none", "node_class": "x86_server"}, {"config": "none", "node_class": "raspberry_pi_5"}]
    merge_measured_energy(rows, str(p))
    assert rows[0]["energy_measured_site_max_median_J"] == 1.0
    assert rows[1]["energy_measured_site_max_median_J"] == 9.0
    assert json.dumps(rows, default=str)
