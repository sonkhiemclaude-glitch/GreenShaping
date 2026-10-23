import json

import numpy as np
import pandas as pd
import pytest

from greenshaping.data.adapters import CaptureCsvAdapter, SessionDirectoryAdapter
from greenshaping.data.schemas import DOWN, UP, SessionManifest, SiteSpec
from greenshaping.testbed import framing
from greenshaping.testbed.capture_commands import tcpdump_command, tshark_export_command
from greenshaping.testbed.driver import run_session
from greenshaping.testbed.orchestrator import LangGraphOrchestrator, ScriptedOrchestrator
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.question_bank import Question, load_question_bank
from greenshaping.testbed.site_server import SiteServer
from greenshaping.testbed.transport import ShapedTransport


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"age": rng.integers(20, 80, 2000), "sex": rng.integers(0, 2, 2000),
                         "bmi": rng.normal(27, 4, 2000), "y": rng.integers(0, 2, 2000)})


def test_queries(table):
    r, n = run_query(table, "aggregate", {"columns": ["bmi"]})
    assert n == 2000 and r["bmi"]["count"] == 2000
    r, n = run_query(table, "group_by", {"by": "sex", "column": "bmi"})
    assert set(r) == {"0", "1"}
    r, n = run_query(table, "train_round", {"features": ["age", "bmi"], "label": "y", "weights": [0, 0, 0]})
    assert len(r["weights"]) == 3
    r, n = run_query(table, "record_count", {"where": {"sex": 1}})
    assert n == r["count"]
    with pytest.raises(ValueError):
        run_query(table, "nope", {})


def test_site_server_scripted_orchestrator_and_manifest(table, tmp_path):
    servers = [SiteServer(table).start(), SiteServer(table.iloc[:500]).start()]
    try:
        sites = [s.address for s in servers]
        q = Question(1, "aggregate", "mean bmi", [[{"site": 0, "params": {"columns": ["bmi"]}}]])
        man = run_session(ScriptedOrchestrator(), q, 7, sites, ["x86_server", "raspberry_pi_5"], tmp_path, "none")
    finally:
        for s in servers:
            s.stop()
    assert man.sites[0].participates == 1 and man.sites[0].matching_rows == 2000
    assert man.sites[1].participates == 0
    saved = SessionManifest.from_dict(json.loads((tmp_path / "manifests" / "7.json").read_text()))
    assert saved.sites[0].bucket() == 0


def test_capture_adapter_merges_segments(tmp_path):
    man = SessionManifest(3, 4, "group_by", 1000.0, (SiteSpec("10.0.0.2", 8000, 1, 50_000),
                                                      SiteSpec("10.0.0.3", 8000, 0)))
    rows = ["1000.10,10.0.0.1,10.0.0.2,50000,8000,600",
            "1000.20,10.0.0.2,10.0.0.1,8000,50000,1400", "1000.205,10.0.0.2,10.0.0.1,8000,50000,1400",
            "1000.90,10.0.0.2,10.0.0.1,8000,50000,100", "1000.95,10.0.0.9,10.0.0.1,9,9,77"]
    (tmp_path / "manifests").mkdir()
    (tmp_path / "captures").mkdir()
    (tmp_path / "manifests" / "3.json").write_text(json.dumps(man.to_dict()))
    (tmp_path / "captures" / "3.csv").write_text("\n".join(rows) + "\n")
    s = SessionDirectoryAdapter(tmp_path).load()[0]
    assert [(round(m.t, 3), m.direction, m.size) for m in s.messages] == [(0.1, UP, 600), (0.2, DOWN, 2800), (0.9, DOWN, 100)]
    assert s.size_bucket == [1, -1] and s.meta["data_kind"] == "real"
    assert CaptureCsvAdapter(1.0).load(tmp_path / "captures" / "3.csv", man).messages[-1].size == 2900


def test_directory_adapter_reports_missing(tmp_path):
    assert SessionDirectoryAdapter(tmp_path).check_layout()
    with pytest.raises(FileNotFoundError):
        SessionDirectoryAdapter(tmp_path).load()


def test_framing_roundtrip_and_split():
    f = framing.Frame(framing.REPLY, b"abc", complete=True)
    wire = framing.encode(f, 64)
    assert len(wire) == 64 and framing.decode(wire) == f
    frames = framing.split_reply(b"x" * 100, 40)
    assert [len(x.payload) for x in frames] == [34, 34, 32] and frames[-1].complete and not frames[0].complete
    with pytest.raises(ValueError):
        framing.encode(framing.Frame(framing.CALL, b"x" * 100), 50)


def test_question_bank_minimum(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"question_id": 1, "family": "aggregate", "text": "t"}) + "\n")
    with pytest.raises(ValueError):
        load_question_bank(p)
    assert len(load_question_bank(p, require_min=False)) == 1


def test_explicit_configuration_required_and_commands_do_not_execute():
    with pytest.raises(TypeError):
        LangGraphOrchestrator()
    with pytest.raises(TypeError):
        ShapedTransport()
    assert tcpdump_command("en0", "a.pcap", [8000])[0] == "tcpdump"
    assert "tcp.len" in tshark_export_command("a.pcap")


def test_split_capture_by_sessions(tmp_path):
    from greenshaping.data.adapters import split_capture_by_sessions
    (tmp_path / "manifests").mkdir()
    for sid, st in ((0, 100.0), (1, 200.0)):
        (tmp_path / "manifests" / f"{sid}.json").write_text(json.dumps({"session_id": sid, "start_time": st}))
    cap = tmp_path / "all.csv"
    cap.write_text("50,a,b,1,2,3\n150,a,b,1,2,3\n199,a,b,1,2,3\n250,a,b,1,2,3\n")
    counts = split_capture_by_sessions(cap, tmp_path / "manifests", tmp_path / "captures")
    assert counts == {0: 2, 1: 1}
