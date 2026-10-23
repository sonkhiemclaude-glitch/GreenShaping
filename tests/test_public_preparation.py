from __future__ import annotations

import copy
import importlib.util
import io
import json
import runpy
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from greenshaping.testbed.natural_planner import planner_prompt, validate_plan, validate_requested_sites
from greenshaping.testbed.public_workload import make_questions

_spec = importlib.util.spec_from_file_location("ec2_readiness", Path(__file__).parents[1] / "scripts/ec2_readiness.py")
assert _spec and _spec.loader
# The existing SSH helper is deliberately loaded from scripts, never executed here.
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
assess = _module.assess


def test_public_bank_has_disjoint_templates_predicates_and_fresh_ids():
    qs = make_questions(3, 20261027)
    assert len(qs) == len({q.question_id for q in qs}) == 480
    assert min(q.question_id for q in qs) >= 200000
    assert Counter(q.metadata["partition"] for q in qs) == {"train": 240, "select": 80, "test": 160}
    assert qs == make_questions(3, 20261027)
    groups = {p: {q.metadata["template_group"] for q in qs if q.metadata["partition"] == p}
              for p in ("train", "select", "test")}
    assert not groups["train"] & groups["select"]
    assert not groups["train"] & groups["test"]
    assert not groups["select"] & groups["test"]
    for q in qs:
        assert len(q.plan) == 1 and q.plan[0]
        low, high = q.metadata["predicate_group"]
        for call in q.plan[0]:
            assert 0 <= call["site"] < 3
            bounds = call["params"]["where"]["aspect"]
            assert low <= bounds["min"] <= bounds["max"] <= high
            assert "limit_rows" not in call["params"]
    with pytest.raises(ValueError):
        make_questions(1, 0)


def idle():
    return {"cpu_count": 8, "containers": {"returncode": 0, "stdout": ""}, "energy_paths": [], "samples": [
        {"cpu": [0, 0, 0, 100, 0, 0, 0, 0], "load": [0, 0, 0],
         "memory_available_bytes": 90, "memory_total_bytes": 100},
        {"cpu": [0, 0, 0, 200, 0, 0, 0, 0], "load": [0, 0, 0],
         "memory_available_bytes": 90, "memory_total_bytes": 100}]}


def test_idle_does_not_override_other_workload_reservation():
    assert assess(idle())["timing_candidate"]
    result = assess(idle(), reserved=True)
    assert not result["timing_candidate"]
    assert not result["exclusive_use_verified"]
    assert not assess({"error": "SSH timeout"})["timing_candidate"]


def test_readiness_rejects_cpu_contention_steal_and_unreviewed_containers():
    data = idle()
    data["samples"][1]["cpu"][0] = 100
    assert not assess(data)["timing_candidate"]
    data = idle()
    data["samples"][1]["cpu"][7] = 3
    assert "CPU steal exceeds 1%" in assess(data)["blockers"]
    data = idle()
    data["containers"]["stdout"] = "someone-elses-container Up 1 hour"
    assert not assess(data)["timing_candidate"]


def test_reserved_host_guard_blocks_four_host_runs_but_allows_preparation(tmp_path, monkeypatch):
    import ec2_runner

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/ec2_host_reservations.json").write_text(json.dumps({"T4": {"reserved": True, "reason": "other job"}}))
    monkeypatch.setattr(ec2_runner, "ROOT", tmp_path)
    ec2_runner.require_available(["T1", "T2", "T3"])
    with pytest.raises(RuntimeError, match="T4"):
        ec2_runner.require_available(["T1", "T2", "T3", "T4"])


def test_reference_plans_fit_natural_tool_schema_without_exposing_oracle_to_prompt():
    for q in make_questions(3, 20261027):
        reference = {"family": q.family, "calls": q.plan[0]}
        assert validate_plan(reference, 3) == reference
        validate_requested_sites(reference, q.text)
        prompt = planner_prompt(q.text, 3)
        assert q.text in prompt
        assert str(q.question_id) not in prompt
        assert json.dumps(q.plan) not in prompt


@pytest.mark.parametrize("mutation", ["duplicate", "site_bool", "extra_argument", "nan", "column"])
def test_natural_planner_rejects_invalid_model_calls(mutation):
    q = make_questions(3, 1)[0]
    value = copy.deepcopy({"family": q.family, "calls": q.plan[0]})
    if mutation == "duplicate":
        value["calls"].append(value["calls"][0])
    elif mutation == "site_bool":
        value["calls"][0]["site"] = True
    elif mutation == "extra_argument":
        value["calls"][0]["params"]["limit_rows"] = 1000
    elif mutation == "nan":
        value["calls"][0]["params"]["where"]["aspect"]["min"] = float("nan")
    else:
        value["calls"][0]["params"]["columns"] = ["private_file"]
    with pytest.raises(ValueError):
        validate_plan(value, 3)


def test_public_site_allowlist_rejects_hallucinated_but_schema_valid_site():
    q = make_questions(3, 1)[0]
    value = copy.deepcopy({"family": q.family, "calls": q.plan[0]})
    unrequested = next(i for i in range(3) if i not in {c["site"] for c in value["calls"]})
    value["calls"].append({"site": unrequested, "params": value["calls"][0]["params"]})
    value = validate_plan(value, 3)
    with pytest.raises(ValueError, match="explicitly named"):
        validate_requested_sites(value, q.text)


@pytest.mark.parametrize("recover", [True, False])
def test_planner_retries_are_recorded_and_never_replace_model_output_with_oracle(tmp_path, monkeypatch, recover):
    q = make_questions(3, 1)[0]
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps(asdict(q)) + "\n")
    good = {"family": q.family, "calls": q.plan[0]}
    requests = []

    def response(req, timeout):
        requests.append(json.loads(req.data))
        value = good if recover and len(requests) == 2 else {"family": "invented", "calls": []}
        return io.BytesIO(json.dumps({"message": {"content": json.dumps(value)}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", response)
    monkeypatch.setattr(sys, "argv", ["validate", "--bank", str(bank), "--out", str(tmp_path / "result"),
                                      "--model", "mock", "--attempts", "2"])
    if recover:
        runpy.run_path(str(Path(__file__).parents[1] / "scripts/validate_natural_planner.py"), run_name="__main__")
    else:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(Path(__file__).parents[1] / "scripts/validate_natural_planner.py"), run_name="__main__")
        assert exc.value.code == 2
    records = json.loads((tmp_path / "result/attempts.json").read_text())
    assert len(requests) == len(records[0]["attempts"]) == 2
    assert not records[0]["attempts"][0]["valid"]
    assert records[0]["reference_match"] == recover
    assert "invented" not in requests[0]["messages"][0]["content"]
    assert "Your previous response failed validation" in requests[1]["messages"][-1]["content"]
