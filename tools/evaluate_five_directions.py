from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "five_direction_validation"
BASELINE = "paper_core_growth"
STRATEGIES = {
    BASELINE: "baseline",
    "paper_core_vrp_feature": "vrp_hmm_feature",
    "paper_core_bipower_guard": "daily_bipower_proxy",
    "paper_core_vix_trend_recovery": "trend_confirmed_vix_recovery",
    "paper_core_smh_high_vol_budget": "smh_high_volatility_budget",
    "paper_core_vx_futures_guard": "exact_vx_futures_guard",
}
PERIODS = {
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "full_2015_present": ("2015-01-01", None),
}


def load_daily(strategy: str) -> pd.Series:
    frame = load_daily_frame(strategy)
    return frame["net_return"].rename(strategy)


def load_daily_frame(strategy: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_ROOT / strategy / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def metrics_table(returns: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for strategy, series in returns.items():
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "strategy": STRATEGIES[strategy],
                    "period": period,
                    **performance_metrics(series.loc[start:end]),
                }
            )
    return pd.DataFrame(rows).set_index(["strategy", "period"])


def circular_block_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    block_days: int = 21,
    samples: int = 10_000,
    seed: int = 20_260_722,
) -> tuple[float, float, float, float]:
    aligned = pd.concat([candidate, baseline], axis=1, join="inner").dropna()
    relative_log_return = (
        np.log1p(aligned.iloc[:, 0].to_numpy())
        - np.log1p(aligned.iloc[:, 1].to_numpy())
    )
    count = len(relative_log_return)
    blocks = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets[None, :]).ravel()[:count] % count
        estimates[sample] = np.expm1(relative_log_return[indices].mean() * 252)
    observed = float(np.expm1(relative_log_return.mean() * 252))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    probability_positive = float((estimates > 0.0).mean())
    return observed, float(lower), float(upper), probability_positive


def annual_win_count(candidate: pd.Series, baseline: pd.Series) -> tuple[int, int]:
    aligned = pd.concat([candidate, baseline], axis=1, join="inner").loc[
        "2015-01-01":"2025-12-31"
    ]
    annual = (1.0 + aligned).groupby(aligned.index.year).prod() - 1.0
    return int((annual.iloc[:, 0] > annual.iloc[:, 1]).sum()), len(annual)


def regime_frame(strategy: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_ROOT / strategy / "regimes.csv", index_col=0, parse_dates=True
    )


def trigger_table() -> pd.DataFrame:
    baseline = regime_frame(BASELINE)
    rows: list[dict[str, float | int | str]] = []
    for strategy, label in STRATEGIES.items():
        if strategy == BASELINE:
            continue
        frame = regime_frame(strategy)
        shared = frame.index.intersection(baseline.index)
        decision_disagreements = int(
            (
                frame.loc[shared, "risk_on_leverage"].astype(int)
                != baseline.loc[shared, "risk_on_leverage"].astype(int)
            ).sum()
        )
        active = 0
        detail = float("nan")
        active_definition = ""
        detail_definition = ""
        if strategy == "paper_core_vrp_feature":
            active = int(
                (
                    frame.loc[shared, "hmm_order"].astype(int)
                    != baseline.loc[shared, "hmm_order"].astype(int)
                ).sum()
            )
            detail = float(frame["hmm_order"].mean())
            active_definition = "hmm_order_differs_from_baseline"
            detail_definition = "average_hmm_order"
        elif strategy == "paper_core_bipower_guard":
            active = int(
                (
                    frame.loc[shared, "trend_stress"].astype(int)
                    != baseline.loc[shared, "trend_stress"].astype(int)
                ).sum()
            )
            detail = float(frame["trend_stress"].mean())
            active_definition = "stress_flag_differs_from_baseline"
            detail_definition = "stress_flag_share"
        elif strategy == "paper_core_vix_trend_recovery":
            active = int(frame["vix_recovery_bridge_active"].astype(int).sum())
            detail = float(frame["vix_recovery_trend_confirmed"].mean())
            active_definition = "recovery_bridge_active"
            detail_definition = "positive_trend_confirmation_share"
        elif strategy == "paper_core_smh_high_vol_budget":
            active_mask = frame["paper_core_high_volatility"].astype(bool)
            active = int(active_mask.sum())
            detail = float(frame.loc[active_mask, "paper_core_SEMIS_weight"].mean())
            active_definition = "smh_high_volatility_state"
            detail_definition = "mean_smh_core_weight_while_active"
        elif strategy == "paper_core_vx_futures_guard":
            active = int(frame["vix_backwardation"].astype(int).sum())
            detail = float(
                (
                    frame["vix_backwardation"].astype(bool)
                    & baseline.loc[frame.index, "paper_risk_on_candidate"].astype(bool)
                ).sum()
            )
            active_definition = "vx1_above_vx2"
            detail_definition = "backwardation_conflicts_with_baseline_risk_on"
        rows.append(
            {
                "strategy": label,
                "rebalance_observations": len(frame),
                "active_or_changed_rebalances": active,
                "risk_decision_disagreements_vs_baseline": decision_disagreements,
                "active_definition": active_definition,
                "direction_specific_detail": detail,
                "detail_definition": detail_definition,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    returns = {strategy: load_daily(strategy) for strategy in STRATEGIES}
    metrics = metrics_table(returns)
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    baseline_complete = metrics.loc[("baseline", "complete_2015_2025")]
    baseline_development = metrics.loc[("baseline", "development_2015_2021")]
    baseline_holdout = metrics.loc[("baseline", "holdout_2022_2025")]
    acceptance_rows: list[dict[str, float | int | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    for strategy, label in STRATEGIES.items():
        if strategy == BASELINE:
            continue
        complete = metrics.loc[(label, "complete_2015_2025")]
        development = metrics.loc[(label, "development_2015_2021")]
        holdout = metrics.loc[(label, "holdout_2022_2025")]
        cagr_pass = bool(complete["cagr"] >= baseline_complete["cagr"] + 0.01)
        drawdown_pass = bool(complete["max_drawdown"] >= -0.18)
        development_pass = bool(development["cagr"] >= baseline_development["cagr"])
        holdout_pass = bool(holdout["cagr"] >= baseline_holdout["cagr"])
        acceptance_rows.append(
            {
                "strategy": label,
                "complete_cagr": complete["cagr"],
                "cagr_delta_vs_baseline": complete["cagr"] - baseline_complete["cagr"],
                "complete_max_drawdown": complete["max_drawdown"],
                "development_cagr_delta": development["cagr"] - baseline_development["cagr"],
                "holdout_cagr_delta": holdout["cagr"] - baseline_holdout["cagr"],
                "cagr_plus_1pp_pass": int(cagr_pass),
                "drawdown_18pct_pass": int(drawdown_pass),
                "development_no_degradation_pass": int(development_pass),
                "holdout_no_degradation_pass": int(holdout_pass),
                "overall_pass": int(
                    cagr_pass and drawdown_pass and development_pass and holdout_pass
                ),
            }
        )
        observed, lower, upper, probability = circular_block_bootstrap(
            returns[strategy].loc["2015-01-01":"2025-12-31"],
            returns[BASELINE].loc["2015-01-01":"2025-12-31"],
        )
        wins, years = annual_win_count(returns[strategy], returns[BASELINE])
        bootstrap_rows.append(
            {
                "strategy": label,
                "paired_annual_relative_return": observed,
                "bootstrap_95pct_lower": lower,
                "bootstrap_95pct_upper": upper,
                "bootstrap_probability_positive": probability,
                "annual_wins_vs_baseline": wins,
                "annual_periods": years,
                "block_days": 21,
                "bootstrap_samples": 10_000,
            }
        )

    acceptance = pd.DataFrame(acceptance_rows).set_index("strategy")
    bootstrap = pd.DataFrame(bootstrap_rows).set_index("strategy")
    triggers = trigger_table()
    acceptance.to_csv(DESTINATION / "acceptance.csv")
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")
    triggers.to_csv(DESTINATION / "trigger_diagnostics.csv")

    implementation_rows: list[dict[str, float | int | str]] = []
    for strategy, label in STRATEGIES.items():
        daily = load_daily_frame(strategy).loc["2015-01-01":"2025-12-31"]
        implementation_rows.append(
            {
                "strategy": label,
                "gross_cagr": performance_metrics(daily["gross_return"])["cagr"],
                "net_cagr": performance_metrics(daily["net_return"])["cagr"],
                "annualized_average_cost": float(daily["cost"].mean() * 252),
                "average_daily_turnover": float(daily["turnover"].mean()),
                "days_with_any_cost": int((daily["cost"] > 0.0).sum()),
            }
        )
    pd.DataFrame(implementation_rows).set_index("strategy").to_csv(
        DESTINATION / "implementation.csv"
    )

    exact = regime_frame("paper_core_vx_futures_guard")
    approximate = regime_frame("paper_core_vix_guard")
    baseline_regimes = regime_frame(BASELINE)
    shared = exact.index.intersection(approximate.index).intersection(
        baseline_regimes.index
    )
    exact_state = exact.loc[shared, "vix_backwardation"].astype(bool)
    approximate_state = approximate.loc[shared, "vix_backwardation"].astype(bool)
    baseline_candidate = baseline_regimes.loc[
        shared, "paper_risk_on_candidate"
    ].astype(bool)
    pd.DataFrame(
        [
            {
                "rebalance_observations": len(shared),
                "exact_vx_backwardation": int(exact_state.sum()),
                "approximate_vix_vix3m_backwardation": int(
                    approximate_state.sum()
                ),
                "measurement_state_disagreements": int(
                    (exact_state != approximate_state).sum()
                ),
                "exact_conflicts_with_baseline_risk_on": int(
                    (exact_state & baseline_candidate).sum()
                ),
                "approximate_conflicts_with_baseline_risk_on": int(
                    (approximate_state & baseline_candidate).sum()
                ),
            }
        ],
        index=["term_structure"],
    ).to_csv(DESTINATION / "term_structure_measurement.csv")

    columns = [
        "complete_cagr",
        "cagr_delta_vs_baseline",
        "complete_max_drawdown",
        "development_cagr_delta",
        "holdout_cagr_delta",
        "overall_pass",
    ]
    print(acceptance[columns].round(4).to_string())
    print("\nPaired 21-session block bootstrap:")
    print(bootstrap.round(4).to_string())
    print("\nTrigger diagnostics:")
    print(triggers.round(4).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
