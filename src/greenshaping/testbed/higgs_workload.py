"""Memory-bounded HIGGS views: unique records, fixed global cohorts across N.

Only label and the first two source features are projected, without rounding.
Reference generation never executes final-test queries. Sites are simulated.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.schemas import QUERY_FAMILIES
from ..utils.io import write_json, write_jsonl
from ..utils.provenance import file_sha256
from .question_bank import Question

SOURCE_HASH = "ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698"
SPLITS = {"train": (0, 1, 2), "select": (3,), "test": (4,)}
TEMPLATES = (
    "At {sites}, {task}. {predicate}",
    "Use only {sites} to {task}; {predicate}",
    "For {sites}, {predicate} Then {task}.",
    "Contact {sites}. {predicate} Please {task}.",
    "Please {task} using {sites}. {predicate}",
    "On {sites}, {task}, subject to: {predicate}",
    "Restrict the analysis to {sites}: {predicate} Then {task}.",
    "Ask {sites} to {task}. Apply this filter: {predicate}",
    "Query {sites}. {predicate} Your task is to {task}.",
    "After filtering at {sites}, {task}. The filter is: {predicate}",
    "From {sites}, {task}; retain records as follows: {predicate}",
    "For this analysis contact only {sites}; {task}. {predicate}",
)
TASKS = {
    "aggregate": "return count, sum and sum of squares of feature_02",
    "group_by": "return count and sum of feature_02 grouped by label",
    "record_count": "count the matching records",
    "train_round": "take one logistic regression gradient step predicting label from feature_01 and feature_02, "
                   "with intercept and both feature weights initially zero and learning rate 0.1",
}


def prepare_projection(source: Path, output: Path, seed: int = 20261027, rows: int = 11000000) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if rows == 11000000 and file_sha256(source) != SOURCE_HASH:
        raise ValueError("HIGGS archive SHA256 differs from pinned audited source")
    output.mkdir(parents=True)
    target = output / "projection.npy"
    array = np.lib.format.open_memmap(target, mode="w+", dtype="float64", shape=(rows, 3))
    offset = 0
    for chunk in pd.read_csv(source, header=None, usecols=[0, 1, 2], chunksize=250000):
        values = chunk.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.isin(values[:, 0], [0, 1]).all() or offset + len(values) > rows:
            raise ValueError("invalid projection")
        array[offset:offset + len(values)] = values
        offset += len(values)
    if offset != rows or rows % 100:
        raise ValueError("unexpected row count or unbalanced N=5/10/20 cohorts")
    array.flush()
    order = np.random.default_rng(seed).permutation(rows)
    np.save(output / "order.npy", order, allow_pickle=False)
    # The threshold distribution uses train records only, shared across all N.
    ranks = np.arange(rows).reshape(-1, 5)[:, :3].ravel()
    train_feature = np.sort(array[order[ranks], 1])
    np.save(output / "train_feature_sorted.npy", train_feature, allow_pickle=False)
    manifest = {"source_sha256": file_sha256(source), "rows": rows, "seed": seed,
                "projection_columns": ["label", "feature_01", "feature_02"], "dtype": "float64",
                "cohort_assignment": "seeded permutation rank % 5: train=0,1,2; select=3; test=4",
                "site_assignment": "(permutation rank // 5) % N", "physical_sites": False,
                "global_cohort_independent_of_N": True, "test_queries_executed": False,
                "files": {p.name: file_sha256(p) for p in output.glob("*.npy")}}
    write_json(manifest, output / "projection.json")
    return manifest


def site_table(root: Path, n_sites: int, site: int) -> pd.DataFrame:
    if n_sites not in (5, 10, 20) or not 0 <= site < n_sites:
        raise ValueError("invalid federation")
    data = np.load(root / "projection.npy", mmap_mode="r")
    order = np.load(root / "order.npy", mmap_mode="r")
    ranks = (5 * np.arange(site, len(order) // 5, n_sites)[:, None] + np.arange(5)).ravel()
    ids = order[ranks]
    values = data[ids]
    return pd.DataFrame({"source_row_id": ids, "cohort": ranks % 5, "label": values[:, 0].astype(int),
                         "feature_01": values[:, 1], "feature_02": values[:, 2]})


def make_questions(root: Path, n_sites: int, seed: int = 20261027) -> list[Question]:
    if n_sites not in (5, 10, 20):
        raise ValueError("expected N=5/10/20")
    rows = len(np.load(root / "order.npy", mmap_mode="r"))
    train = np.load(root / "train_feature_sorted.npy", mmap_mode="r")
    rng = np.random.default_rng(seed + n_sites)
    result = []
    for family in QUERY_FAMILIES:
        for template_id, template in enumerate(TEMPLATES):
            partition = "train" if template_id < 6 else ("select" if template_id < 8 else "test")
            cohorts = SPLITS[partition]
            per_site = rows * len(cohorts) // (5 * n_sites)
            for j in range(10):
                size = ("small", "medium", "large")[j % 3]
                target = (5000, 40000, 150000)[j % 3] * float(rng.uniform(.9, 1.1))
                where = {"cohort": {"min": min(cohorts), "max": max(cohorts)}}
                predicate = f"Retain cohort values from {min(cohorts)} through {max(cohorts)} inclusive."
                if target < per_site:
                    threshold = float(train[min(len(train) - 1, int(target / per_site * len(train)))])
                    where["feature_01"] = {"min": -1000000., "max": threshold}
                    predicate += f" Also retain feature_01 between -1000000.0 and {threshold!r} inclusive."
                chosen = np.flatnonzero(rng.random(n_sites) < .65).tolist()
                if not chosen:
                    chosen = [int(rng.integers(n_sites))]
                params = {"where": where}
                if family == "aggregate":
                    params["columns"] = ["feature_02"]
                elif family == "group_by":
                    params.update(by="label", column="feature_02")
                elif family == "train_round":
                    params.update(features=["feature_01", "feature_02"], label="label", weights=[0., 0., 0.], lr=.1)
                text = template.format(sites=", ".join(f"site {s}" for s in chosen), task=TASKS[family], predicate=predicate)
                result.append(Question(300000 + n_sites * 10000 + len(result), family, text,
                                       [[{"site": s, "params": params} for s in chosen]],
                                       {"partition": partition, "template_group": template_id, "requested_size": size,
                                        "workload": "higgs-projection-v1", "plan_source": "reference; not LLM acquisition"}))
    return result


def build_banks(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {"test_queries_executed": False, "ready_for_large_run": False, "federations": {}}
    for n in (5, 10, 20):
        directory = output / f"n{n}"
        directory.mkdir()
        questions = make_questions(root, n)
        for split in SPLITS:
            write_jsonl([asdict(q) for q in questions if q.metadata["partition"] == split], directory / f"{split}.jsonl")
        # Count real train/select predicate matches efficiently; do not execute test plans.
        started = time.monotonic()
        failures, counts = [], []
        participation = {}
        for site in range(n):
            table = site_table(root, n, site)
            for split in ("train", "select"):
                feature = np.sort(table.loc[table.cohort.isin(SPLITS[split]), "feature_01"].to_numpy())
                for family in QUERY_FAMILIES:
                    relevant = [q for q in questions if q.family == family and q.metadata["partition"] == split]
                    positives = sum(any(c["site"] == site for c in q.plan[0]) for q in relevant)
                    participation[f"{site}/{split}/{family}"] = {"positive": positives, "negative": len(relevant) - positives}
                    if positives == 0 or positives == len(relevant):
                        failures.append(f"participation:{site}/{split}/{family}")
                for q in questions:
                    if q.metadata["partition"] != split or not any(c["site"] == site for c in q.plan[0]):
                        continue
                    bounds = q.plan[0][0]["params"]["where"].get("feature_01")
                    count = len(feature) if bounds is None else int(np.searchsorted(feature, bounds["max"], side="right") -
                                                                  np.searchsorted(feature, bounds["min"], side="left"))
                    size = "small" if 1000 <= count < 10000 else ("medium" if 10000 <= count < 100000 else (
                        "large" if 100000 <= count < 1000000 else "outside"))
                    counts.append({"question_id": q.question_id, "site": site, "split": split, "family": q.family,
                                   "rows": count, "observed_size": size, "requested_size": q.metadata["requested_size"]})
                    if size != q.metadata["requested_size"]:
                        failures.append(f"size:{q.question_id}/{site}:{count}")
        coverage = {}
        for split in ("train", "select"):
            for family in QUERY_FAMILIES:
                for size in ("small", "medium", "large"):
                    key = f"{split}/{family}/{size}"
                    coverage[key] = sum(c["split"] == split and c["family"] == family and c["observed_size"] == size for c in counts)
                    if not coverage[key]:
                        failures.append("missing:" + key)
        write_jsonl(counts, directory / "train_select_counts.jsonl")
        report["federations"][str(n)] = {"questions": len(questions), "split_questions": {"train": 240, "select": 80, "test": 160},
                                         "coverage": coverage, "participation": participation, "failures": failures,
                                         "size_support_pass": not failures, "audit_seconds": time.monotonic() - started,
                                         "bank_hashes": {s: file_sha256(directory / f"{s}.jsonl") for s in SPLITS}}
    report["reference_preparation_pass"] = all(v["size_support_pass"] for v in report["federations"].values())
    write_json(report, output / "bank_audit.json")
    return report
