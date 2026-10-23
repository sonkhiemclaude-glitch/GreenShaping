"""Summarize pilot robustness, controlling for public site identity, and draw static figures."""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from evaluate_pilot import LABEL, SECRETS, score  # noqa: E402

from greenshaping.attack.classifiers import RandomForestAttacker  # noqa: E402
from greenshaping.attack.evaluation import split_by_question  # noqa: E402
from greenshaping.data.adapters import SessionDirectoryAdapter  # noqa: E402
from greenshaping.metrics.confidence_intervals import bootstrap_by_group  # noqa: E402
from greenshaping.metrics.metrics import chance_level, majority_class  # noqa: E402
from greenshaping.utils.io import write_csv, write_json  # noqa: E402
from greenshaping.utils.provenance import file_sha256  # noqa: E402

ALIASES = {
    "none": "Unshaped", "pad(r=2.0,beta=512)": "Bucket padding",
    "rounds(W=0.1,q=0.5,R=2048)": "Rounds q=0.5", "rounds(W=0.1,q=1.0,R=2048)": "Rounds q=1",
    "dp(queue,eps=4.0,tau=0.1,H=5.0,p=100.0)": "DP queue",
    "dp(arrivals,eps=4.0,tau=0.1,H=5.0,p=100.0)": "DP arrivals",
    "rounds(W=0.1,q=0.5,R=2048)+pad(r=2.0,beta=512)": "Rounds + padding",
    "rounds(W=0.1,q=0.5,R=2048)+dp(arrivals,eps=4.0,tau=0.1,H=5.0,p=100.0)": "Rounds + DP arrivals",
    "rounds(W=0.1,q=0.5,R=2048)+pad(r=2.0,beta=512)+dp(arrivals,eps=4.0,tau=0.1,H=5.0,p=100.0)":
        "Rounds + padding + DP",
    "fixed": "Fixed padding",
}


def identity_rows(sessions, secret):
    y, ids = [], []
    for session in sessions:
        if secret == "query_family":
            y.append(session.query_family)
            ids.append(-1)
        else:
            for site in range(session.n_sites):
                if secret == "size_bucket" and not session.participation[site]:
                    continue
                y.append(session.size_bucket[site] if secret == "size_bucket" else session.participation[site])
                ids.append(site)
    return np.asarray(y), np.asarray(ids)


def method(values):
    return values.where(~values.str.startswith("fixed("), "fixed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    out = Path(args.evaluation)
    protocol = json.loads((out / "protocol.json").read_text())
    for name, expected_hash in protocol["input_sha256"].items():
        if file_sha256(Path(args.data_root) / name) != expected_hash:
            raise ValueError(f"input changed since evaluation: {name}")
    scores, costs = pd.read_csv(out / "scores.csv"), pd.read_csv(out / "costs.csv")
    preds = pd.read_csv(out / "predictions.csv")
    scores["method"], costs["method"] = method(scores.config), method(costs.config)
    sessions = SessionDirectoryAdapter(args.data_root).load()
    identity_scores, splits, identity_comparisons = [], [], []
    conditional_chance = {}
    for seed in protocol["seeds"]:
        train, select, test = split_by_question(sessions, seed=seed)
        for part, ss in [("train", train), ("select", select), ("fit", train + select), ("test", test)]:
            splits.append({"seed": seed, "split": part, "sessions": len(ss), "questions": len({s.question_id for s in ss}),
                           "participation_prior": float(np.mean([x for s in ss for x in s.participation]))})
        for secret in SECRETS:
            ytr, itr = identity_rows(train + select, secret)
            yte, ite = identity_rows(test, secret)
            rows = preds[(preds.seed == seed) & (preds.config == "none") & (preds.model == "rf") &
                         (preds.secret == secret)].sort_values("row")
            assert np.array_equal(rows.y_true.to_numpy(), yte)
            n = sessions[0].n_sites
            Xtr = np.zeros((len(ytr), 1)) if secret == "query_family" else np.eye(n)[itr]
            Xte = np.zeros((len(yte), 1)) if secret == "query_family" else np.eye(n)[ite]
            if len(np.unique(ytr)) < 2:
                pred = np.full(len(yte), majority_class(ytr))
                p1 = np.zeros(len(yte))
            else:
                clf = RandomForestAttacker(200, seed, n_jobs=4).fit(Xtr, ytr)
                pred, probs = clf.predict(Xte), clf.predict_proba(Xte)
                p1 = probs[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(yte))
            baseline = score(yte, pred, p1, rows.question_id.to_numpy(), ytr, protocol["n_boot"], seed)
            if secret != "participation":
                baseline["auc"] = float("nan")
            identity_scores.append({"seed": seed, "secret": secret, "model": "rf_identity_only",
                                    **baseline, "result_status": LABEL})
            for attack in ("rf", "rf_linkaware", "cnn"):
                traffic = preds[(preds.seed == seed) & (preds.config == "none") & (preds.model == attack) &
                                (preds.secret == secret)].sort_values("row")
                tp = traffic.y_pred.to_numpy()

                def gain(idx, tp=tp, pred=pred, yte=yte):
                    return float(np.mean(tp[idx] == yte[idx]) - np.mean(pred[idx] == yte[idx]))

                lo, hi = bootstrap_by_group(rows.question_id.to_numpy(), gain, n_boot=protocol["n_boot"], seed=seed)
                identity_comparisons.append({"seed": seed, "secret": secret, "model": attack,
                                             "traffic_accuracy_minus_identity": float(np.mean(tp == yte)) - baseline["accuracy"],
                                             "gain_ci_low": lo, "gain_ci_high": hi, "result_status": LABEL})
            # This is an oracle descriptive test-set chance, analogous to the existing
            # global majority chance, conditioned on the already-public site ID.
            ch = sum(len(yte[ite == i]) * chance_level(yte[ite == i]) for i in np.unique(ite)) / len(yte)
            conditional_chance[(seed, secret)] = ch
    write_csv(identity_scores, out / "identity_baseline.csv")
    write_csv(identity_comparisons, out / "traffic_over_identity.csv")
    write_csv(splits, out / "split_diagnostics.csv")
    scores["chance_given_site_test"] = [conditional_chance[(int(r.seed), r.secret)] for r in scores.itertuples()]
    scores["adv_given_site"] = np.where(scores.chance_given_site_test < 1,
                                        (scores.accuracy - scores.chance_given_site_test) /
                                        (1 - scores.chance_given_site_test), np.nan)
    scores["adv_given_site_clipped"] = scores.adv_given_site.clip(lower=0)
    scores.to_csv(out / "scores_with_identity_control.csv", index=False)
    worst = scores.groupby(["method", "model", "seed"], as_index=False)[["adv_clipped", "adv_given_site_clipped"]].max()
    worst.to_csv(out / "worst_by_model_seed.csv", index=False)
    portfolio = scores.groupby(["method", "seed"], as_index=False)[["adv_clipped", "adv_given_site_clipped"]].max()
    identity_floor = pd.DataFrame(identity_scores).groupby("seed").adv_clipped.max()
    portfolio["adv_three_traffic_models"] = portfolio.adv_clipped
    portfolio["identity_baseline_adv"] = portfolio.seed.map(identity_floor)
    portfolio["adv_clipped"] = portfolio[["adv_clipped", "identity_baseline_adv"]].max(axis=1)
    portfolio.to_csv(out / "portfolio_by_seed.csv", index=False)
    summary = []
    for name, group in portfolio.groupby("method"):
        cost = costs[costs.method == name]
        summary.append({"method": name, "label": ALIASES[name], "computational_seeds": len(group),
                        "adv_max_mean": group.adv_clipped.mean(), "adv_max_min": group.adv_clipped.min(),
                        "adv_max_max": group.adv_clipped.max(), "adv_given_site_mean": group.adv_given_site_clipped.mean(),
                        "adv_given_site_min": group.adv_given_site_clipped.min(),
                        "adv_given_site_max": group.adv_given_site_clipped.max(),
                        "byte_factor_mean": 1 + cost.bandwidth_overhead.mean(),
                        "byte_factor_min": 1 + cost.bandwidth_overhead.min(),
                        "byte_factor_max": 1 + cost.bandwidth_overhead.max(),
                        "added_latency_median_s_mean": cost.latency_median_s.mean(),
                        "added_latency_p95_s_mean": cost.latency_p95_s.mean(),
                        "failure_rate_mean": cost.privacy_failure_rate.mean(),
                        "failure_rate_max": cost.privacy_failure_rate.max(),
                        "overflow_rate_mean": cost.overflow_rate.mean(),
                        "energy_modelled_site_max_J_mean": cost.energy_modelled_site_max_J.mean(),
                        "energy_model_fitted": False, "result_status": LABEL})
    write_csv(summary, out / "evaluation_summary.csv")
    write_json({"type": "descriptive sensitivity analysis, not new final-test confirmation",
                "source_sha256": file_sha256(__file__), "numpy": version("numpy"), "sklearn": version("scikit-learn"),
                "portfolio": "maximum over RF, link-aware RF, CNN and a separately fitted identity-only RF",
                "identity_control": "oracle best constant class per known site on the test labels; family uses global chance",
                "undefined_conditioned_advantage": "NaN when conditioned chance is 1; max ignores NaN",
                "intervals": "figure whiskers are min/max across computational seeds, not independent-run confidence intervals",
                "replications": "same acquired sessions reused; only split, shaping and model seeds vary"},
               out / "analysis_notes.json")

    order = list(ALIASES)
    palette = {"rf": "#2878b5", "rf_linkaware": "#d55e00", "cnn": "#009e73"}
    titles = {"rf": "RF pooled", "rf_linkaware": "RF link + network", "cnn": "CNN"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, field, heading in zip(axes, ["adv_clipped", "adv_given_site_clipped"],
                                  ["Original global chance", "Chance conditioned on public site identity"], strict=True):
        for j, model_name in enumerate(palette):
            means, lower, upper = [], [], []
            for name in order:
                vals = worst[(worst.method == name) & (worst.model == model_name)][field]
                means.append(vals.mean())
                lower.append(vals.mean() - vals.min())
                upper.append(vals.max() - vals.mean())
            ax.errorbar(means, np.arange(len(order)) + (j - 1) * 0.19, xerr=[lower, upper], fmt="o",
                        capsize=2, markersize=4, color=palette[model_name], label=titles[model_name])
        if field == "adv_clipped":
            ax.axvline(identity_floor.mean(), color="#777777", linestyle="--", linewidth=1,
                       label="Identity-only mean (no traffic)")
        ax.set_title(heading, fontsize=10)
        ax.set_xlim(-0.03, 1.06)
        ax.set_xlabel("Max clipped advantage over three secrets")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(range(len(order)), [ALIASES[x] for x in order])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Pilot robustness: mean and seed range\nSingle-host rehearsal; same capture; no measured energy", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(out / "robustness.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    by_name = {row["method"]: row for row in summary}
    offsets = [(0, 10), (0, 18), (-7, 11), (6, -12), (-12, 12), (-6, 11), (7, -14), (0, 27), (6, -14), (7, 10)]
    legend_handles, legend_labels = [], []
    for j, name in enumerate(order):
        row = by_name[name]
        marker = "X" if row["failure_rate_mean"] > 0 else "o"
        point = ax.scatter(row["byte_factor_mean"], row["adv_given_site_mean"], marker=marker, s=65, color="#2878b5")
        ax.annotate(str(j + 1), (row["byte_factor_mean"], row["adv_given_site_mean"]),
                    xytext=offsets[j], textcoords="offset points", fontsize=9)
        legend_handles.append(point)
        legend_labels.append(f"{j + 1}. {row['label']}")
    ax.set_xscale("log")
    ax.set_xlabel("Total byte factor, offline shaping / observed traffic")
    ax.set_ylabel("Max tested advantage conditioned on public site identity")
    ax.set_ylim(-0.04, 1.1)
    ax.grid(alpha=0.2)
    ax.set_title("Pilot trade-off: means over five computational seeds\nX = observed DP failures; single-host rehearsal", fontsize=11)
    ax.legend(legend_handles, legend_labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=180)
    plt.close(fig)
    print(pd.DataFrame(summary)[["label", "adv_max_mean", "adv_given_site_mean", "byte_factor_mean",
                                  "failure_rate_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
