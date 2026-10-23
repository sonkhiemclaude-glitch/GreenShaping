"""Train/select feasibility and smoke reference generation; never executes final test."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from greenshaping.data.schemas import size_bucket, size_bucket_clipped
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.question_bank import load_question_bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.workload / "workload.json").read_text())
    tables = [pd.read_csv(a.workload / f"site_{i}.csv") for i in range(manifest["n_sites"])]
    ids = pd.concat([x.source_row_id for x in tables])
    if len(ids) != manifest["source_rows"] or ids.nunique() != len(ids):
        raise ValueError("source rows lost or duplicated between sites")
    rows, selected = [], []
    for split in ["train", "select"]:
        for q in load_question_bank(a.workload / f"{split}.jsonl", require_min=False):
            if split == "train" and sum(x.family == q.family for x in selected) < 2:
                selected.append(q)
            for c in q.plan[0]:
                result, n = run_query(tables[c["site"]], q.family, c["params"])
                rows.append({"partition": split, "family": q.family, "question_id": q.question_id,
                             "site": c["site"], "matching_rows": n, "size_bucket": size_bucket(n),
                             "clipped": size_bucket_clipped(n),
                             "reply_bytes": len(json.dumps({"result": result, "matching_rows": n}).encode())})
    df = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "train_select_feasibility.csv", index=False)
    support = []
    for (split, family), group in df.groupby(["partition", "family"]):
        support.append({"partition": split, "family": family,
                        "size_bucket_counts": {str(k): int(v) for k, v in group.size_bucket.value_counts().items()},
                        "all_three_buckets": set(group.size_bucket) == {0, 1, 2},
                        "clipped_calls": int(group.clipped.sum())})
    summary = {"unique_source_rows": int(ids.nunique()), "source_records_repeated_between_sites": False,
               "test_executed": False, "calls": len(rows), "matching_rows_min": int(df.matching_rows.min()),
               "matching_rows_max": int(df.matching_rows.max()), "clipped_calls": int(df.clipped.sum()),
               "max_reply_bytes": int(df.reply_bytes.max()), "support_per_split_family": support,
               "full_manuscript_size_support": all(v["all_three_buckets"] and not v["clipped_calls"] for v in support)}
    (a.out / "workload_feasibility.json").write_text(json.dumps(summary, indent=2))
    (a.out / "smoke_train.jsonl").write_text("".join(json.dumps(asdict(q)) + "\n" for q in selected))
    reference = {str(q.question_id): [{"result": result, "matching_rows": n} for c in q.plan[0]
                                     for result, n in [run_query(tables[c["site"]], q.family, c["params"])]] for q in selected}
    (a.out / "smoke_reference.json").write_text(json.dumps(reference, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
