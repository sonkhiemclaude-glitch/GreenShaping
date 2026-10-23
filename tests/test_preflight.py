import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts" / "preflight_experiment.py")
assert spec and spec.loader
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_synthetic_preflight_does_not_write_results(tiny_config, tmp_path):
    output = tmp_path / "not_created"
    tiny_config["run"]["output_dir"] = str(output)
    report = preflight.check(tiny_config)
    assert report["ok"] and report["data_kind"] == "synthetic"
    assert not report["paper_ready"] and not output.exists()


def test_paper_preflight_rejects_demo_and_known_stubs(tiny_config):
    report = preflight.check(tiny_config, paper=True)
    assert not report["ok"]
    assert any("ShapedTransport" in e for e in report["errors"])
    assert any("real LLM" in e for e in report["errors"])


def test_preflight_reports_missing_real_data(tiny_config, tmp_path):
    tiny_config["data"] = {"source": "directory", "root": str(tmp_path / "absent")}
    report = preflight.check(tiny_config)
    assert not report["ok"]
    assert any("missing directory" in e for e in report["errors"])
