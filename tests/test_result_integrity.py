"""No fake results: every shipped result file is labelled synthetic and has provenance."""
import csv
import json
from pathlib import Path

from greenshaping import SYNTHETIC_LABEL

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_outputs_are_labelled_synthetic():
    # Only the shipped demo; real runs written under outputs/ later carry their own label (minor 22).
    out = ROOT / "outputs" / "synthetic_demo"
    assert out.is_dir()
    csvs = list(out.rglob("*.csv"))
    for p in csvs:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows, f"{p} is empty"
        assert all(r.get("result_status") == SYNTHETIC_LABEL for r in rows), f"{p} has unlabelled rows"
    for meta in out.rglob("run_metadata.json"):
        d = json.loads(meta.read_text())
        assert d["data_kind"] == "synthetic" and d["result_status"] == SYNTHETIC_LABEL


def test_no_result_tables_inside_the_package():
    # Result tables live only in outputs/ (examples/ and audits/ hold templates and the
    # traceability matrix). Manuscript audit CSVs beside this repository are not checked.
    for d in ("src", "tests", "configs", "scripts", "docs"):
        found = list((ROOT / d).rglob("*.csv"))
        assert not found, f"unexpected result-like CSV {found}"


def test_examples_contain_no_metrics():
    for p in (ROOT / "examples").rglob("*.csv"):
        header = p.read_text().splitlines()[0]
        for word in ("accuracy", "advantage", "auc", "energy"):
            assert word not in header.lower(), f"{p} looks like a result table"
