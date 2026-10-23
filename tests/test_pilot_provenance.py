import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

from greenshaping.data.adapters import CaptureCsvAdapter, split_capture_by_sessions
from greenshaping.data.schemas import SessionManifest, SiteSpec


def test_llm_rehearsal_manifest_preserves_explicit_kind_and_window(tmp_path):
    man = SessionManifest(0, 0, "record_count", 100.0, (SiteSpec("127.0.0.1", 18400, 1, 2000),),
                          "Ollama local model", "rehearsal", 101.0)
    man = SessionManifest.from_dict(man.to_dict())
    p = tmp_path / "capture.csv"
    with p.open("w") as f:
        csv.writer(f).writerows([[t, "127.0.0.1", "127.0.0.1", 50000, 18400, 200]
                                for t in (99.0, 100.1, 102.0)])
    session = CaptureCsvAdapter().load(p, man)
    assert session.meta["data_kind"] == "rehearsal"
    assert session.meta["llm"] == "Ollama local model"
    assert len(session.messages) == 1 and session.messages[0].size == 200


def test_capture_split_excludes_background_after_session_end(tmp_path):
    mans = tmp_path / "manifests"
    mans.mkdir()
    for sid, start, end in ((0, 100.0, 101.0), (1, 103.0, 104.0)):
        (mans / f"{sid}.json").write_text(json.dumps({"session_id": sid, "start_time": start, "end_time": end}))
    capture = tmp_path / "all.csv"
    capture.write_text("100.5,a,b,1,2,10\n102.0,a,b,1,2,10\n103.5,a,b,1,2,10\n105.0,a,b,1,2,10\n")
    assert split_capture_by_sessions(capture, mans, tmp_path / "captures") == {0: 1, 1: 1}


def test_manifest_rejects_bad_provenance_or_end_time():
    with pytest.raises(ValueError, match="data_kind"):
        SessionManifest(0, 0, "aggregate", 100.0, (), data_kind="paper")
    with pytest.raises(ValueError, match="end_time"):
        SessionManifest(0, 0, "aggregate", 100.0, (), end_time=99.0)


def _pilot_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_mac_studio_pilot.py"
    spec = importlib.util.spec_from_file_location("mac_studio_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_planner_repair_preserves_rejected_attempt(monkeypatch, tmp_path):
    pilot = _pilot_module()
    calls = []

    def respond(req, timeout):
        body = json.loads(req.data)
        prompt = body["messages"][0]["content"]
        info = json.loads(prompt.split("Input: ", 1)[1].split(". Analyst question:", 1)[0])
        sites = [x["site_id"] for x in info["catalog"] if x["region_id"] in info["requested_regions"]]
        if not calls:
            sites = [2]  # reject the first plan, then require a genuine second model response
        calls.append(body)
        return io.BytesIO(json.dumps({"message": {"content": json.dumps({"sites": sites})}}).encode())

    monkeypatch.setattr(pilot.urllib.request, "urlopen", respond)
    questions = pilot.make_questions("fixture-model", "http://unused", tmp_path, 3, 0)
    assert len(questions) == 12 and len(calls) == 13
    evidence = json.loads((tmp_path / "llm_plans.jsonl").read_text().splitlines()[0])
    assert evidence["attempts"][0]["selection_correct"] is False
    assert evidence["attempts"][1]["selection_correct"] is True
    assert len(evidence["attempts"][1]["request"]["messages"]) == 3


def test_planner_never_substitutes_scripted_answer(monkeypatch, tmp_path):
    pilot = _pilot_module()
    monkeypatch.setattr(pilot.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.BytesIO(b'{"message":{"content":"{\\"sites\\":[2]}"}}'))
    with pytest.raises(ValueError, match="after 3 attempts; no fallback"):
        pilot.make_questions("fixture-model", "http://unused", tmp_path, 3, 0)
    evidence = json.loads((tmp_path / "llm_plans.jsonl").read_text().splitlines()[0])
    assert len(evidence["attempts"]) == 3 and evidence["selection_correct"] is False
