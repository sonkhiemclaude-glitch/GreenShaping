"""Read-only reanalysis of frozen evidence; no model fitting, selection or acquisitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODES = ["none", "pad", "rounds", "dp", "combo"]
SECRETS = ["participation", "size_bucket", "query_family"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/contribution_review_20261020")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    live = ROOT / "outputs/remote_v4_20261020/results"
    scores = pd.read_csv(live / "evaluation/scores.csv")
    preds = pd.read_csv(live / "evaluation/predictions.csv")
    old = json.loads((live / "summary/summary.json").read_text())
    ids = np.sort(preds.question_id.unique())
    seeds = sorted(preds.seed.unique().tolist())
    assert len(ids) == 48 and len(seeds) == 5 and len(scores) == 300
    rng = np.random.default_rng(20261020)
    weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=args.n_boot)
    weights = np.vstack([np.ones(len(ids), dtype=int), weights])
    values = {}
    for key, frame in preds.groupby(["seed", "mode", "model", "secret"]):
        correct = (frame.y_true == frame.y_pred).groupby(frame.question_id).sum().reindex(ids, fill_value=0).to_numpy()
        counts = frame.groupby("question_id").size().reindex(ids, fill_value=0).to_numpy()
        n = weights @ counts
        chance_count = np.zeros(len(weights))
        for _, site in frame.groupby("site"):
            classes = pd.crosstab(site.question_id, site.y_true).reindex(ids, fill_value=0).to_numpy()
            chance_count += (weights @ classes).max(axis=1)
        accuracy = np.divide(weights @ correct, n, out=np.zeros(len(n)), where=n > 0)
        chance = np.divide(chance_count, n, out=np.ones(len(n)), where=n > 0)
        advantage = np.divide(accuracy - chance, 1 - chance, out=np.zeros(len(n)), where=chance < 1)
        advantage = np.maximum(0, advantage)
        seed, mode, model, secret = key
        recorded = scores[(scores.seed == seed) & (scores["mode"] == mode) & (scores.model == model) & (scores.secret == secret)].iloc[0]
        assert abs(advantage[0] - recorded.adv_clipped) < 1e-10, key
        values[key] = advantage
    strongest = {}
    for seed in seeds:
        for mode in MODES:
            strongest[(seed, mode)] = np.max([v for (s, m, _, _), v in values.items() if s == seed and m == mode], axis=0)
    # Average computational seeds within each question resample, not as independent acquisitions.
    mean = {mode: np.mean([strongest[(seed, mode)] for seed in seeds], axis=0) for mode in MODES}
    delta = mean["dp"] - mean["combo"]
    pd.DataFrame({"bootstrap_dp_minus_combo": delta[1:]}).to_csv(out / "paired_bootstrap.csv", index=False)
    strongest_secret = scores.groupby(["seed", "mode", "secret"]).adv_clipped.max().reset_index()
    secret_summary = strongest_secret.groupby(["mode", "secret"]).adv_clipped.agg(["mean", "min", "max"]).reset_index()
    secret_summary.to_csv(out / "advantage_by_secret.csv", index=False)
    for row in old["test_strongest_summary"]:
        assert abs(mean[row["mode"]][0] - row["mean"]) < 1e-12
    bootstrap = dict(
        estimand="Mean across five computational seeds of maximum clipped advantage over four attackers and three secrets",
        contrast="DP alone minus fixed composition; positive favors composition",
        point=float(delta[0]),
        ci95=np.quantile(delta[1:], [0.025, 0.975]).tolist(),
        n_boot=args.n_boot,
        seed=20261020,
        question_clusters=len(ids),
        computational_seeds=seeds,
        independent_acquisitions_per_question_mode=1,
        status="Exploratory reanalysis of already-inspected v4 holdout, not confirmatory H3 or equal-energy evidence",
    )
    (out / "paired_bootstrap_summary.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    eps_rows = []
    cover_rows = []
    K = 80
    Delta = 65536
    eps_dir = 4
    bias = 100000
    for N in (5, 10, 20):
        eps = 2 * N * eps_dir
        eps_rows.append(
            dict(
                N=N,
                link_directions=2 * N,
                eps_dir=eps_dir,
                eps_total=eps,
                accuracy_upper_bound=1 / (1 + math.exp(-eps)),
                one_minus_accuracy_upper_bound=math.exp(-eps) / (1 + math.exp(-eps)),
                eps_dir_for_global_eps1=1 / (2 * N),
                eps_dir_for_accuracy_upper_bound_06=math.log(0.6 / 0.4) / (2 * N),
            )
        )
        for policy, e in (("eps_dir_fixed_4", 4.0), ("global_eps_fixed_1", 1 / (2 * N))):
            scale = Delta / e
            theory = 2 * N * K * scale / 2
            draws = np.maximum(0, rng.laplace(0, scale, size=200000))
            estimate = 2 * N * K * float(draws.mean())
            se = 2 * N * K * float(draws.std(ddof=1)) / math.sqrt(len(draws))
            assert abs(estimate - theory) < 5 * se
            cover_rows.append(
                dict(
                    N=N,
                    policy=policy,
                    eps_dir=e,
                    eps_total=2 * N * e,
                    empty_arrivals_expected_network_padding_bytes=theory,
                    mc_estimate_bytes=estimate,
                    mc_se_bytes=se,
                    queue_expected_network_padding_bytes=2 * N * K * K * Delta / (2 * e),
                )
            )
    pd.DataFrame(eps_rows).to_csv(out / "epsilon_accounting.csv", index=False)
    pd.DataFrame(cover_rows).to_csv(out / "mechanism_scaling.csv", index=False)
    scale = Delta / eps_dir
    empty_bias_per_tick = bias + (scale / 2) * math.exp(-bias / scale)
    analytic = dict(
        scope="Ideal releases, empty queues, independent noise, no byte rounding; illustrative parameters from v4, not measured energy",
        N=5,
        K=K,
        Delta=Delta,
        bias=bias,
        eps_dir=eps_dir,
        empty_arrivals_expected_network_padding_bytes=2 * 5 * K * empty_bias_per_tick,
        empty_arrivals_zero_bias_expected_network_padding_bytes=2 * 5 * K * scale / 2,
        normalized_global_budget_observation="With d=2N and uniform eps_dir=eps_total/(2N), zero-bias arrival empty-cover expectation "
        "is 2*N^2*K*Delta/eps_total; queue padding is 2*N^2*K^2*Delta/eps_total. "
        "Mechanism-specific, not universal lower bounds or novelty claims.",
    )
    (out / "analytical_checks.json").write_text(json.dumps(analytic, indent=2) + "\n")
    grid = dict(
        current_template_candidates=5 * 10 * 49,
        with_two_reply_frame_options=5 * 19 * 49,
        n_single_or_none_with_two_reply_frames=1 + 4 + 18 + 48,
        note="Counts before feasibility pruning; controller prototype profiles an offline grid. "
        "V4 live pilot used only five fixed modes and no measured-energy budget.",
    )
    (out / "grid_size.json").write_text(json.dumps(grid, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    colors = ["#597185", "#d89b48", "#579e8d"]
    for j, secret in enumerate(SECRETS):
        s = secret_summary[secret_summary.secret == secret].set_index("mode").reindex(MODES)
        axes[0].bar(np.arange(5) + (j - 1) * 0.23, s["mean"], width=0.23, color=colors[j], label=secret)
    axes[0].set(
        xticks=range(5),
        xticklabels=MODES,
        ylabel="Mean strongest attacker advantage",
        ylim=(0, 1.05),
        title="Existing synthetic pilot, 48 test questions",
    )
    axes[0].legend(fontsize=7)
    costs = {x["mode"]: x["observed_payload_factor"] for x in old["observed_costs"]}
    axes[1].bar(MODES, [costs[m] for m in MODES], color="#597185")
    axes[1].set(yscale="log", ylabel="Observed TCP payload factor", title="Bytes measured; joules not measured")
    for policy, label in [("eps_dir_fixed_4", "fixed epsilon per direction = 4"), ("global_eps_fixed_1", "fixed global epsilon = 1")]:
        rows = [x for x in cover_rows if x["policy"] == policy]
        axes[2].plot(
            [x["N"] for x in rows], [x["empty_arrivals_expected_network_padding_bytes"] / 2**20 for x in rows], marker="o", label=label
        )
    axes[2].set(
        xlabel="Logical site count N",
        ylabel="Expected empty-cover MiB/session",
        yscale="log",
        xticks=[5, 10, 20],
        title="Arrival mechanism: analytical, not measured",
    )
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Contribution evidence review — empirical leakage, traffic cost and privacy accounting", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "contribution_evidence.png", dpi=180)
    fig.savefig(out / "contribution_evidence.pdf")
    plt.close(fig)
    input_files = [
        live / "evaluation/scores.csv",
        live / "evaluation/predictions.csv",
        live / "summary/summary.json",
        ROOT / "configs/pilot_v4_protocol.json",
        ROOT.parent / "GreenShaping_manuscript.pdf",
        Path(__file__),
    ]
    (out / "input_hashes.json").write_text(
        json.dumps({str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}, indent=2) + "\n"
    )
    (out / "verification.json").write_text(
        json.dumps(
            dict(
                prediction_rows=len(preds),
                score_rows=len(scores),
                original_point_metrics_reproduced=True,
                all_prediction_groups_reproduced=True,
                mc_checks_within_5_standard_errors=True,
                new_model_training=False,
                new_network_acquisition=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(bootstrap=bootstrap, analytic=analytic, grid=grid), indent=2))


if __name__ == "__main__":
    main()
