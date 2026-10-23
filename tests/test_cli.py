import json
import subprocess
import sys
from pathlib import Path

import yaml

from greenshaping import SYNTHETIC_LABEL
from greenshaping.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_help_runs():
    r = subprocess.run([sys.executable, "-m", "greenshaping.cli", "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "No paper-level experimental results" in r.stdout


def test_bounds_command(capsys):
    assert main(["bounds", "--horizon", "20", "--interval", "1", "--delta", "65536", "--eps-dir", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["K"] == 20 and abs(out["eq4_expected_padding_bytes_per_linkdir"] - 13107200) < 1


def test_demo_end_to_end_on_tiny_synthetic_config(tmp_path, tiny_config):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump(tiny_config))
    out = tmp_path / "demo"
    assert main(["--log-level", "WARNING", "demo", "--config", str(cfg), "--output", str(out)]) == 0
    for name in ("e1_leakage", "e2_primitives", "e3_profiles", "e3_budgeted", "e3_baselines", "e4_platform", "e5_scaling"):
        text = (out / f"{name}.csv").read_text()
        assert SYNTHETIC_LABEL in text
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["data_kind"] == "synthetic" and meta["result_status"] == SYNTHETIC_LABEL
    assert (out / "plots" / "e1_leakage.png").is_file()


def test_run_stage_calibrate(tmp_path, tiny_config):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump(tiny_config))
    assert main(["--log-level", "WARNING", "run", "--config", str(cfg), "--stage", "calibrate", "--output", str(tmp_path / "c")]) == 0
    assert (tmp_path / "c" / "e3_selection.csv").is_file()


def test_real_data_refuses_placeholder_cost_model(tmp_path, tiny_config):
    import pytest

    from greenshaping.experiments.common import load_cost_models
    with pytest.raises(ValueError):
        load_cost_models({"energy": {"cost_models": {}}}, "real")
