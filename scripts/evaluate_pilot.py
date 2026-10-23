"""Exploratory robustness evaluation of a fixed rehearsal capture; no acquisition.

Uses fixed candidate configurations, five computational seeds, original RF/CNN and
a stronger link-aware RF. Never presents placeholder joules as measured energy.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from greenshaping.attack.classifiers import RandomForestAttacker, TimeSliceCNN
from greenshaping.attack.evaluation import assert_disjoint, split_by_question
from greenshaping.attack.features import build_tables, build_time_slice_tables, link_features
from greenshaping.data.adapters import SessionDirectoryAdapter
from greenshaping.experiments.common import PLACEHOLDER_COST_MODEL, RunContext, cost_summary, shape_all
from greenshaping.metrics.confidence_intervals import bootstrap_by_group
from greenshaping.metrics.metrics import chance_level, majority_class, normalised_advantage, roc_auc
from greenshaping.shaping.padding import fixed_unit_for_percentile
from greenshaping.shaping.pipeline import DPParams, PaddingParams, RoundsParams, ShapingConfig, resolve_delta
from greenshaping.utils.io import write_csv, write_json
from greenshaping.utils.provenance import file_sha256

LABEL = "exploratory evaluation of single-host rehearsal; not a paper result"
MODELS = ("rf", "rf_linkaware", "cnn")
SECRETS = ("participation", "size_bucket", "query_family")
CNN: dict[str, Any] = {"epochs": 20, "batch_size": 64, "patience": 5, "val_fraction": 0.2, "lr": 0.001}


def linkaware_tables(sessions: list[Any]) -> tuple[dict, dict]:
    """Use public link identity and the full observable network, never labels as features."""
    links, global_rows = build_tables(sessions)
    link_x, session_x = [], []
    for j, session in enumerate(sessions):
        flat = np.asarray([link_features(session, site) for site in range(session.n_sites)]).ravel()
        for site in range(session.n_sites):
            link_x.append(np.concatenate([link_features(session, site), np.eye(session.n_sites)[site], flat]))
        session_x.append(np.concatenate([global_rows["X"][j], flat]))
    links["X"], global_rows["X"] = np.asarray(link_x), np.asarray(session_x)
    return links, global_rows


def candidates(calibration: list[Any]) -> list[ShapingConfig]:
    pad = PaddingParams(2.0, 512)
    rounds = RoundsParams(window=0.1, cover_prob=0.5, reply_frame=2048)
    dp = DPParams(eps_dir=4.0, interval=0.1, horizon=5.0, mode="arrivals")
    return [ShapingConfig(), ShapingConfig(padding=pad), ShapingConfig(rounds=rounds),
            ShapingConfig(rounds=RoundsParams(window=0.1, cover_prob=1.0, reply_frame=2048)),
            ShapingConfig(dp=DPParams(eps_dir=4.0, interval=0.1, horizon=5.0, mode="queue")),
            ShapingConfig(dp=dp), ShapingConfig(rounds=rounds, padding=pad),
            ShapingConfig(rounds=rounds, dp=dp), ShapingConfig(rounds=rounds, padding=pad, dp=dp),
            ShapingConfig(fixed_unit=fixed_unit_for_percentile(calibration, 99))]


def labels_and_features(tables: tuple[dict, dict], secret: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    links, global_rows = tables
    if secret == "query_family":
        return global_rows["X"], global_rows["family"], global_rows["group"]
    mask = links["part"] == 1 if secret == "size_bucket" else np.ones(len(links["part"]), bool)
    return links["X"][mask], links["bucket" if secret == "size_bucket" else "part"][mask], links["group"][mask]


def score(y: np.ndarray, pred: np.ndarray, p1: np.ndarray, groups: np.ndarray,
          train_y: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    acc, chance = float(np.mean(y == pred)), chance_level(y)
    balanced = float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))

    def adv(idx: np.ndarray) -> float:
        return normalised_advantage(float(np.mean(y[idx] == pred[idx])), chance_level(y[idx]))

    low, high = bootstrap_by_group(groups, adv, n_boot=n_boot, seed=seed)
    return {"accuracy": acc, "chance_test": chance, "train_majority_accuracy": float(np.mean(y == majority_class(train_y))),
            "balanced_accuracy": balanced, "adv_raw": normalised_advantage(acc, chance),
            "adv_clipped": max(0, normalised_advantage(acc, chance)), "adv_ci_low": low, "adv_ci_high": high,
            "auc": roc_auc(y, p1) if set(np.unique(y)) == {0, 1} else float("nan"),
            "n_test": len(y), "n_questions_test": len(np.unique(groups)),
            "train_counts": json.dumps({str(c): int(np.sum(train_y == c)) for c in np.unique(train_y)}),
            "test_counts": json.dumps({str(c): int(np.sum(y == c)) for c in np.unique(y)})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20261020, 20261025)))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    adapter = SessionDirectoryAdapter(root)
    sessions = adapter.load()
    if {s.meta["data_kind"] for s in sessions} != {"rehearsal"} or len({s.n_sites for s in sessions}) != 1:
        raise ValueError("this evaluator requires a single-N rehearsal capture")
    protocol = {"result_status": LABEL, "seeds": args.seeds, "rf_trees": 200, "rf_jobs": 4,
                "cnn": CNN, "slice_s": 0.05, "n_slices": 120, "n_boot": args.n_boot,
                "split": [0.4, 0.3, 0.3], "fit": "train+select; fixed exploratory candidates, no new controller selection",
                "seed_scope": "split/shaping/model only; same 96 acquired sessions for every seed",
                "source_sha256": file_sha256(__file__), "energy_model": asdict(PLACEHOLDER_COST_MODEL),
                "input_sha256": {str(p.relative_to(root)): file_sha256(p) for p in adapter.input_files()}}
    write_json(protocol, out / "protocol.json")
    status: dict[str, Any] = {"state": "running", "phase": "evaluate", "completed_configs": 0,
                              "total_configs": 10 * len(args.seeds)}
    all_scores, all_costs, paired = [], [], []
    pred_file = (out / "predictions.csv").open("w", newline="")
    pred_writer = csv.DictWriter(pred_file, fieldnames=["seed", "config", "model", "secret", "row", "question_id",
                                                      "y_true", "y_pred", "p1"])
    pred_writer.writeheader()
    try:
        for seed in args.seeds:
            train, select, test = split_by_question(sessions, seed=seed)
            assert_disjoint(train, select, test)
            fit = train + select
            write_json({name: sorted({s.question_id for s in group}) for name, group in
                        [("train", train), ("select", select), ("test", test)]}, out / f"split_{seed}.json")
            ctx = RunContext({}, "rehearsal", out, seed, sessions, train, select, test,
                             {"default": PLACEHOLDER_COST_MODEL}, [])
            baseline = {}
            for cfg in candidates(fit):
                cfg = resolve_delta(cfg, fit, seed)
                name = cfg.label()
                status.update(seed=seed, config=name)
                write_json(status, out / "status.json")
                sh_fit, _ = shape_all(fit, cfg, seed, "fit")
                sh_test, stats = shape_all(test, cfg, seed, "test")
                costs: dict[str, Any] = cost_summary(ctx, test, sh_test, stats, PLACEHOLDER_COST_MODEL)
                costs.update(seed=seed, config=name, config_json=json.dumps(cfg.to_dict(), sort_keys=True), result_status=LABEL)
                all_costs.append(costs)
                for model in MODELS:
                    if model == "cnn":
                        tr_tables = build_time_slice_tables(sh_fit, 0.05, 120)
                        te_tables = build_time_slice_tables(sh_test, 0.05, 120)
                    else:
                        builder = linkaware_tables if model == "rf_linkaware" else build_tables
                        tr_tables, te_tables = builder(sh_fit), builder(sh_test)
                    for secret in SECRETS:
                        Xtr, ytr, _ = labels_and_features(tr_tables, secret)
                        Xte, yte, groups = labels_and_features(te_tables, secret)
                        if len(np.unique(ytr)) < 2:
                            pred = np.full(len(yte), majority_class(ytr))
                            p1 = (pred == 1).astype(float)
                        else:
                            clf = (TimeSliceCNN(seed=seed, **CNN) if model == "cnn" else
                                   RandomForestAttacker(n_estimators=200, seed=seed, n_jobs=4))
                            clf.fit(Xtr, ytr)
                            pred = clf.predict(Xte)
                            classes = list(clf.classes_)
                            proba = clf.predict_proba(Xte)
                            p1 = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(yte))
                        scores = score(yte, pred, p1, groups, ytr, args.n_boot, seed)
                        # AUC is meaningful here only for binary participation, not bucket labels.
                        if secret != "participation":
                            scores["auc"] = float("nan")
                        all_scores.append({"seed": seed, "config": name, "model": model, "secret": secret,
                                           **scores, "result_status": LABEL})
                        for j in range(len(yte)):
                            pred_writer.writerow({"seed": seed, "config": name, "model": model, "secret": secret,
                                                  "row": j, "question_id": int(groups[j]), "y_true": int(yte[j]),
                                                  "y_pred": int(pred[j]), "p1": float(p1[j])})
                        if name == "none":
                            baseline[(model, secret)] = (yte.copy(), pred.copy(), groups.copy())
                        else:
                            by, bp, bg = baseline[(model, secret)]
                            assert np.array_equal(by, yte) and np.array_equal(bg, groups)

                            def delta(idx: np.ndarray, by: np.ndarray = by, bp: np.ndarray = bp,
                                      yte: np.ndarray = yte, pred: np.ndarray = pred) -> float:
                                return normalised_advantage(float(np.mean(by[idx] == bp[idx])), chance_level(by[idx])) - \
                                    normalised_advantage(float(np.mean(yte[idx] == pred[idx])), chance_level(yte[idx]))

                            lo, hi = bootstrap_by_group(groups, delta, n_boot=args.n_boot, seed=seed)
                            paired.append({"seed": seed, "config": name, "model": model, "secret": secret,
                                           "adv_gain_over_unshaped": normalised_advantage(float(np.mean(by == bp)),
                                                                                         chance_level(by)) - scores["adv_raw"],
                                           "gain_ci_low": lo, "gain_ci_high": hi, "result_status": LABEL})
                pred_file.flush()
                status["completed_configs"] += 1
                write_csv(all_scores, out / "scores.csv")
                write_csv(all_costs, out / "costs.csv")
                if paired:
                    write_csv(paired, out / "paired_gain.csv")
                print(f"{status['completed_configs']}/{status['total_configs']} seed={seed} {name}", flush=True)
        status.update(state="complete", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(status, out / "status.json")
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(status, out / "status.json")
        raise
    finally:
        pred_file.close()


if __name__ == "__main__":
    main()
