from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, ttest_1samp

from evaluate_smh_dynamic_guard import (
    OpenGapGuard,
    STRATEGY,
    load_inputs,
    simulate,
    variants,
)
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/smh_dynamic_guard")
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_present": ("2015-01-01", None),
    "recent_2024_present": ("2024-01-01", None),
}
SELECTED = {
    "direct": "neighborhood_gap300bp_min50_cap15_hold5",
    "direct_reentry_hold6": "neighborhood_gap300bp_min50_cap15_hold6",
    "account_loss_200bp_cap25_hold5": "accountloss200bp_cap25_hold5",
    "account_loss_220bp_cap25_hold5": "accountloss220bp_cap25_hold5",
    "conditional_account_loss_hold5": (
        "accountloss200bp_rel2_or220bp_cap25_hold5"
    ),
    "conditional_account_loss_hold6": (
        "accountloss200bp_rel2_or220bp_cap25_hold6"
    ),
    "conditional_stage_gap1_hold5": (
        "accountloss200bp_rel2_or220bp_cap25"
        "_continuegap1_deep15_hold5"
    ),
    "conditional_stage_gap1_hold6": (
        "accountloss200bp_rel2_or220bp_cap25"
        "_continuegap1_deep15_hold6"
    ),
    "conditional_stage_gap3_hold5": (
        "accountloss200bp_rel2_or220bp_cap25"
        "_continuegap3_deep15_hold5"
    ),
    "account_loss_200bp_stage_gap1": (
        "accountloss200bp_cap25_continuegap1_deep15_hold5"
    ),
    "account_loss_200bp_stage_gap3": (
        "accountloss200bp_cap25_continuegap3_deep15_hold5"
    ),
    "direct_hold6_delayed": (
        "neighborhood_gap300bp_min50_cap15_hold6_delay1"
    ),
    "direct_hold5_delayed": (
        "neighborhood_gap300bp_min50_cap15_hold5_delay1"
    ),
    "causal_capture": (
        "causal_gap250bp_min60_cap35_confirm2_hold4"
    ),
    "causal_balanced": (
        "causal_gap250bp_min60_cap20_confirm2_hold4"
    ),
    "causal_production": (
        "causal_gap250bp_min60_cap25_confirm2_hold4"
    ),
    "staged": "staged_initial25_continuegap1_deep15_confirm2_hold5",
    "delayed": "staged_initial25_deep15_hold5_delay1",
}


def guard_by_name(name: str) -> OpenGapGuard:
    return next(guard for guard in variants() if guard.name == name)


def annualized_relative_log_return(
    candidate: pd.Series,
    baseline: pd.Series,
) -> float:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    return float(
        (
            np.log1p(aligned["candidate"])
            - np.log1p(aligned["baseline"])
        ).mean()
        * 252.0
    )


def event_cycles(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cycle_bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    trigger_date: pd.Timestamp | None = None
    for date, row in candidate.iterrows():
        if trigger_date is None and bool(row["triggered"]):
            trigger_date = date
        if (
            trigger_date is not None
            and bool(row["recovered"])
            and not bool(row["triggered"])
        ):
            cycle_bounds.append((trigger_date, date))
            trigger_date = None
    if trigger_date is not None:
        cycle_bounds.append((trigger_date, candidate.index[-1]))

    for trigger_date, recovery_date in cycle_bounds:
        selected = candidate.loc[trigger_date:recovery_date, "net_return"]
        reference = baseline.loc[trigger_date:recovery_date, "net_return"]
        candidate_return = float((1.0 + selected).prod() - 1.0)
        baseline_return = float((1.0 + reference).prod() - 1.0)
        rows.append(
            {
                "variant": label,
                "trigger_date": trigger_date,
                "recovery_date": recovery_date,
                "sessions": len(selected),
                "candidate_return": candidate_return,
                "baseline_return": baseline_return,
                "return_delta": candidate_return - baseline_return,
                "candidate_worst_day": float(selected.min()),
                "baseline_worst_day": float(reference.min()),
            }
        )
    return pd.DataFrame(rows)


def event_clusters(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    label: str,
    maximum_gap_sessions: int = 5,
) -> pd.DataFrame:
    cycles = event_cycles(candidate, baseline, label)
    if cycles.empty:
        return cycles

    positions = pd.Series(
        np.arange(len(candidate.index)),
        index=candidate.index,
    )
    merged_bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cluster_start = pd.Timestamp(cycles.iloc[0]["trigger_date"])
    cluster_end = pd.Timestamp(cycles.iloc[0]["recovery_date"])
    for _, cycle in cycles.iloc[1:].iterrows():
        trigger_date = pd.Timestamp(cycle["trigger_date"])
        recovery_date = pd.Timestamp(cycle["recovery_date"])
        separation = int(
            positions.loc[trigger_date] - positions.loc[cluster_end]
        )
        if separation <= maximum_gap_sessions:
            cluster_end = recovery_date
        else:
            merged_bounds.append((cluster_start, cluster_end))
            cluster_start = trigger_date
            cluster_end = recovery_date
    merged_bounds.append((cluster_start, cluster_end))

    rows: list[dict[str, object]] = []
    for trigger_date, recovery_date in merged_bounds:
        selected = candidate.loc[trigger_date:recovery_date, "net_return"]
        reference = baseline.loc[trigger_date:recovery_date, "net_return"]
        candidate_return = float((1.0 + selected).prod() - 1.0)
        baseline_return = float((1.0 + reference).prod() - 1.0)
        rows.append(
            {
                "variant": label,
                "trigger_date": trigger_date,
                "recovery_date": recovery_date,
                "sessions": len(selected),
                "candidate_return": candidate_return,
                "baseline_return": baseline_return,
                "return_delta": candidate_return - baseline_return,
                "candidate_worst_day": float(selected.min()),
                "baseline_worst_day": float(reference.min()),
            }
        )
    return pd.DataFrame(rows)


def leave_one_cluster_out(
    candidate: pd.Series,
    baseline: pd.Series,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    relative = (
        np.log1p(aligned["candidate"])
        - np.log1p(aligned["baseline"])
    )
    rows: list[dict[str, object]] = []
    for _, cluster in clusters.iterrows():
        kept = relative.loc[
            ~relative.index.to_series().between(
                pd.Timestamp(cluster["trigger_date"]),
                pd.Timestamp(cluster["recovery_date"]),
            )
        ]
        rows.append(
            {
                "variant": cluster["variant"],
                "omitted_trigger_date": cluster["trigger_date"],
                "omitted_recovery_date": cluster["recovery_date"],
                "remaining_observations": len(kept),
                "annualized_relative_log_return": float(
                    kept.mean() * 252.0
                ),
            }
        )
    return pd.DataFrame(rows)


def circular_block_relative_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    block_days: int,
    samples: int = 20_000,
    seed: int = 20_260_727,
) -> dict[str, float | int]:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    relative = (
        np.log1p(aligned["candidate"])
        - np.log1p(aligned["baseline"])
    ).to_numpy(dtype=float)
    observations = len(relative)
    rng = np.random.default_rng(seed + block_days)
    full_blocks, remainder = divmod(observations, block_days)
    starts = rng.integers(
        0,
        observations,
        size=(samples, full_blocks + int(remainder > 0)),
    )
    full_block_sums = np.asarray(
        [
            relative[(start + np.arange(block_days)) % observations].sum()
            for start in range(observations)
        ]
    )
    totals = full_block_sums[starts[:, :full_blocks]].sum(axis=1)
    if remainder:
        remainder_sums = np.asarray(
            [
                relative[(start + np.arange(remainder)) % observations].sum()
                for start in range(observations)
            ]
        )
        totals += remainder_sums[starts[:, -1]]
    annualized = totals / observations * 252.0
    return {
        "block_days": block_days,
        "samples": samples,
        "observed_annualized_relative_log_return": float(
            relative.mean() * 252.0
        ),
        "probability_positive": float((annualized > 0.0).mean()),
        "relative_log_return_5pct": float(np.quantile(annualized, 0.05)),
        "relative_log_return_median": float(np.median(annualized)),
        "relative_log_return_95pct": float(np.quantile(annualized, 0.95)),
    }


def event_bootstrap(
    cycles: pd.DataFrame,
    samples: int = 50_000,
    seed: int = 20_260_727,
) -> dict[str, float | int]:
    values = cycles["return_delta"].to_numpy(dtype=float)
    positive_events = int((values > 0.0).sum())
    rng = np.random.default_rng(seed)
    sampled_means = rng.choice(
        values,
        size=(samples, len(values)),
        replace=True,
    ).mean(axis=1)
    return {
        "events": len(values),
        "samples": samples,
        "observed_mean_delta": float(values.mean()),
        "positive_event_share": float((values > 0.0).mean()),
        "positive_event_rate_95pct_lower": float(
            beta.ppf(
                0.05,
                positive_events,
                len(values) - positive_events + 1,
            )
            if positive_events > 0
            else 0.0
        ),
        "one_sided_sign_test_pvalue": float(
            binomtest(
                positive_events,
                len(values),
                0.5,
                alternative="greater",
            ).pvalue
        ),
        "one_sided_mean_t_test_pvalue": float(
            ttest_1samp(
                values,
                0.0,
                alternative="greater",
            ).pvalue
            if len(values) >= 2
            else float("nan")
        ),
        "probability_positive_mean": float((sampled_means > 0.0).mean()),
        "mean_delta_5pct": float(np.quantile(sampled_means, 0.05)),
        "mean_delta_median": float(np.median(sampled_means)),
        "mean_delta_95pct": float(np.quantile(sampled_means, 0.95)),
    }


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    weights, base_daily, opens, closes = load_inputs()
    baseline_guard = guard_by_name("baseline")
    stress_rows: list[dict[str, object]] = []
    standard_results: dict[str, pd.DataFrame] = {}

    for cost_bps in (7.5, 15.0, 25.0):
        baseline, _ = simulate(
            weights,
            base_daily,
            opens,
            closes,
            baseline_guard,
            cost_bps=cost_bps,
        )
        for slippage_bps in (
            0.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            75.0,
            100.0,
            250.0,
            500.0,
            1000.0,
        ):
            for label, name in SELECTED.items():
                guard = replace(
                    guard_by_name(name),
                    emergency_slippage_bps=slippage_bps,
                )
                candidate, _ = simulate(
                    weights,
                    base_daily,
                    opens,
                    closes,
                    guard,
                    cost_bps=cost_bps,
                )
                if cost_bps == 7.5 and slippage_bps == 20.0:
                    standard_results[label] = candidate
                for period, (start, end) in PERIODS.items():
                    selected = candidate.loc[start:end, "net_return"]
                    reference = baseline.loc[start:end, "net_return"]
                    candidate_metrics = performance_metrics(selected)
                    baseline_metrics = performance_metrics(reference)
                    stress_rows.append(
                        {
                            "variant": label,
                            "cost_bps": cost_bps,
                            "emergency_slippage_bps": slippage_bps,
                            "period": period,
                            **candidate_metrics,
                            "cagr_delta_vs_baseline": (
                                candidate_metrics["cagr"]
                                - baseline_metrics["cagr"]
                            ),
                            "max_drawdown_delta_vs_baseline": (
                                candidate_metrics["max_drawdown"]
                                - baseline_metrics["max_drawdown"]
                            ),
                            "annualized_relative_log_return": (
                                annualized_relative_log_return(
                                    selected,
                                    reference,
                                )
                            ),
                        }
                    )

    standard_baseline, _ = simulate(
        weights,
        base_daily,
        opens,
        closes,
        baseline_guard,
    )
    cycle_frames = [
        event_cycles(candidate, standard_baseline, label)
        for label, candidate in standard_results.items()
    ]
    cycles = pd.concat(cycle_frames, ignore_index=True)
    cluster_frames = [
        event_clusters(candidate, standard_baseline, label)
        for label, candidate in standard_results.items()
    ]
    clusters = pd.concat(cluster_frames, ignore_index=True)
    leave_one_out = pd.concat(
        [
            leave_one_cluster_out(
                candidate["net_return"],
                standard_baseline["net_return"],
                clusters.loc[clusters["variant"] == label],
            )
            for label, candidate in standard_results.items()
        ],
        ignore_index=True,
    )
    bootstrap_rows: list[dict[str, object]] = []
    event_bootstrap_rows: list[dict[str, object]] = []
    for label, candidate in standard_results.items():
        for block_days in (21, 63, 126):
            bootstrap_rows.append(
                {
                    "variant": label,
                    **circular_block_relative_bootstrap(
                        candidate["net_return"],
                        standard_baseline["net_return"],
                        block_days,
                    ),
                }
            )
        event_bootstrap_rows.append(
            {
                "variant": label,
                **event_bootstrap(
                    clusters.loc[clusters["variant"] == label]
                ),
            }
        )

    seed_rows: list[dict[str, object]] = []
    for seed in (7, 42, 123):
        member_root = (
            Path("output")
            / STRATEGY
            / "members"
            / f"seed_{seed}"
        )
        member_weights = pd.read_csv(
            member_root / "weights.csv",
            index_col=0,
            parse_dates=True,
        )
        member_daily = pd.read_csv(
            member_root / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )
        member_baseline, _ = simulate(
            member_weights,
            member_daily,
            opens,
            closes,
            baseline_guard,
        )
        for label, name in SELECTED.items():
            member_candidate, _ = simulate(
                member_weights,
                member_daily,
                opens,
                closes,
                guard_by_name(name),
            )
            for period, (start, end) in PERIODS.items():
                selected = member_candidate.loc[start:end, "net_return"]
                reference = member_baseline.loc[start:end, "net_return"]
                candidate_metrics = performance_metrics(selected)
                baseline_metrics = performance_metrics(reference)
                seed_rows.append(
                    {
                        "seed": seed,
                        "variant": label,
                        "period": period,
                        "cagr_delta_vs_baseline": (
                            candidate_metrics["cagr"]
                            - baseline_metrics["cagr"]
                        ),
                        "max_drawdown_delta_vs_baseline": (
                            candidate_metrics["max_drawdown"]
                            - baseline_metrics["max_drawdown"]
                        ),
                        "annualized_relative_log_return": (
                            annualized_relative_log_return(
                                selected,
                                reference,
                            )
                        ),
                        "trigger_count": int(
                            member_candidate.loc[start:end, "triggered"].sum()
                        ),
                    }
                )

    stress = pd.DataFrame(stress_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    event_uncertainty = pd.DataFrame(event_bootstrap_rows)
    seed_robustness = pd.DataFrame(seed_rows)

    proxy_root = Path("output") / (
        "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
        "_20y_proxy"
    )
    proxy_weights = pd.read_csv(
        proxy_root / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    proxy_daily = pd.read_csv(
        proxy_root / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    proxy_baseline, _ = simulate(
        proxy_weights,
        proxy_daily,
        opens,
        closes,
        baseline_guard,
        start_date="2012-01-01",
    )
    proxy_rows: list[dict[str, object]] = []
    proxy_periods = {
        "early_proxy_2012_2014": ("2012-01-01", "2014-12-31"),
        "complete_proxy_2012_present": ("2012-01-01", None),
    }
    for label, name in SELECTED.items():
        proxy_candidate, _ = simulate(
            proxy_weights,
            proxy_daily,
            opens,
            closes,
            guard_by_name(name),
            start_date="2012-01-01",
        )
        for period, (start, end) in proxy_periods.items():
            selected = proxy_candidate.loc[start:end, "net_return"]
            reference = proxy_baseline.loc[start:end, "net_return"]
            candidate_metrics = performance_metrics(selected)
            baseline_metrics = performance_metrics(reference)
            proxy_rows.append(
                {
                    "variant": label,
                    "period": period,
                    **candidate_metrics,
                    "cagr_delta_vs_baseline": (
                        candidate_metrics["cagr"]
                        - baseline_metrics["cagr"]
                    ),
                    "max_drawdown_delta_vs_baseline": (
                        candidate_metrics["max_drawdown"]
                        - baseline_metrics["max_drawdown"]
                    ),
                    "annualized_relative_log_return": (
                        annualized_relative_log_return(
                            selected,
                            reference,
                        )
                    ),
                    "trigger_count": int(
                        proxy_candidate.loc[start:end, "triggered"].sum()
                    ),
                }
            )
    proxy_robustness = pd.DataFrame(proxy_rows)

    stress.to_csv(DESTINATION / "execution_stress.csv", index=False)
    cycles.to_csv(DESTINATION / "event_cycles.csv", index=False)
    clusters.to_csv(DESTINATION / "event_clusters.csv", index=False)
    leave_one_out.to_csv(
        DESTINATION / "leave_one_cluster_out.csv",
        index=False,
    )
    bootstrap.to_csv(DESTINATION / "block_bootstrap.csv", index=False)
    event_uncertainty.to_csv(
        DESTINATION / "event_bootstrap.csv",
        index=False,
    )
    seed_robustness.to_csv(
        DESTINATION / "seed_robustness.csv",
        index=False,
    )
    proxy_robustness.to_csv(
        DESTINATION / "extended_proxy.csv",
        index=False,
    )

    full = stress.loc[
        (stress["period"] == "complete_2015_present")
        & (stress["cost_bps"].isin([7.5, 15.0]))
        & (stress["emergency_slippage_bps"].isin([20.0, 250.0, 1000.0]))
    ]
    print("Execution stress, complete period:")
    print(
        full[
            [
                "variant",
                "cost_bps",
                "emergency_slippage_bps",
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
                "annualized_relative_log_return",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nBlock bootstrap:")
    print(bootstrap.round(6).to_string(index=False))
    print("\nEvent bootstrap:")
    print(event_uncertainty.round(6).to_string(index=False))
    print("\nSeed robustness, complete period:")
    print(
        seed_robustness.loc[
            seed_robustness["period"] == "complete_2015_present"
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nExtended proxy:")
    print(
        proxy_robustness[
            [
                "variant",
                "period",
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
                "trigger_count",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
