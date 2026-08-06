from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from evaluate_open_execution import load_open_close
from evaluate_smh_dynamic_guard import (
    OpenGapGuard,
    positive_capture,
    simulate as simulate_gap_guard,
)
from evaluate_smh_dynamic_guard_robustness import (
    annualized_relative_log_return,
    circular_block_relative_bootstrap,
)
from evaluate_smh_shock_guard import (
    Guard,
    SAMPLES,
    five_day_compound,
    load_sample,
    simulate as simulate_structural_guard,
)
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/smh_hybrid_tail_guard")
STRATEGY = str(SAMPLES["normal"]["strategy"])
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_present": ("2015-01-01", None),
    "recent_2024_present": ("2024-01-01", None),
}
EVENT_DATES = (
    "2024-07-17",
    "2024-09-03",
    "2025-01-27",
    "2025-04-03",
    "2025-04-04",
    "2026-06-05",
    "2026-06-23",
)


def gap_guard(hold_days: int, slippage_bps: float) -> OpenGapGuard:
    return OpenGapGuard(
        name=f"gap3_min50_cap15_hold{hold_days}",
        absolute_gap_trigger=-0.03,
        post_trigger_cap=0.15,
        minimum_semis_weight=0.50,
        recovery_signal="relative",
        recovery_confirmations=2,
        maximum_hold_days=hold_days,
        overflow_asset="CASH",
        emergency_slippage_bps=slippage_bps,
    )


def structural_frames(
    base_weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cap: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    guard = (
        Guard("baseline")
        if cap is None
        else Guard(
            name=f"cap{int(round(cap * 100))}_toQQQ",
            scheduled_cap=cap,
            overflow_asset="QQQ",
        )
    )
    return simulate_structural_guard(
        base_weights,
        base_daily,
        asset_returns,
        guard,
    )


def metric_row(
    scenario: str,
    cost_bps: float,
    slippage_bps: float,
    period: str,
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, float | str]:
    start, end = PERIODS[period]
    selected = candidate.loc[start:end]
    reference = baseline.loc[start:end]
    candidate_metrics = performance_metrics(selected["net_return"])
    baseline_metrics = performance_metrics(reference["net_return"])
    return {
        "scenario": scenario,
        "cost_bps": cost_bps,
        "emergency_slippage_bps": slippage_bps,
        "period": period,
        **candidate_metrics,
        "baseline_cagr": baseline_metrics["cagr"],
        "baseline_max_drawdown": baseline_metrics["max_drawdown"],
        "cagr_delta_vs_baseline": (
            candidate_metrics["cagr"] - baseline_metrics["cagr"]
        ),
        "max_drawdown_delta_vs_baseline": (
            candidate_metrics["max_drawdown"]
            - baseline_metrics["max_drawdown"]
        ),
        "worst_day": float(selected["net_return"].min()),
        "worst_day_delta_vs_baseline": float(
            selected["net_return"].min()
            - reference["net_return"].min()
        ),
        "worst_five_days_delta_vs_baseline": float(
            five_day_compound(selected["net_return"]).min()
            - five_day_compound(reference["net_return"]).min()
        ),
        "positive_capture": positive_capture(
            selected["net_return"],
            reference["net_return"],
        ),
        "annualized_relative_log_return": (
            annualized_relative_log_return(
                selected["net_return"],
                reference["net_return"],
            )
        ),
        "annualized_turnover": float(
            selected["turnover"].mean() * 252.0
        ),
        "annualized_cost": float(
            (
                selected["trading_cost"]
                + selected["slippage_cost"]
            ).mean()
            * 252.0
        ),
        "trigger_count": int(selected["triggered"].sum()),
    }


def simulate_scenarios(
    base_weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    asset_returns: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    cost_bps: float,
    slippage_bps: float,
) -> dict[str, pd.DataFrame]:
    structures = {
        cap: structural_frames(
            base_weights,
            base_daily,
            asset_returns,
            cap,
        )
        for cap in (None, 0.65, 0.70, 0.75, 0.80)
    }
    results: dict[str, pd.DataFrame] = {}
    baseline_guard = OpenGapGuard("baseline")
    for cap, (structural_daily, structural_weights) in structures.items():
        cap_label = (
            "baseline"
            if cap is None
            else f"cap{int(round(cap * 100))}"
        )
        cap_only, _ = simulate_gap_guard(
            structural_weights,
            structural_daily,
            opens,
            closes,
            baseline_guard,
            cost_bps=cost_bps,
        )
        results[cap_label] = cap_only
        for hold_days in (5, 6):
            candidate, _ = simulate_gap_guard(
                structural_weights,
                structural_daily,
                opens,
                closes,
                gap_guard(hold_days, slippage_bps),
                cost_bps=cost_bps,
            )
            scenario = (
                f"gap_hold{hold_days}"
                if cap is None
                else f"{cap_label}_gap_hold{hold_days}"
            )
            results[scenario] = candidate
    return results


def load_member(
    seed: int,
    asset_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path("output") / STRATEGY / "members" / f"seed_{seed}"
    weights = pd.read_csv(
        root / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        root / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    return weights, daily, asset_returns.reindex(weights.index)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    base_weights, base_daily, asset_returns = load_sample("normal")
    opens, closes = load_open_close(refresh=False)
    metric_rows: list[dict[str, float | str]] = []
    standard_results: dict[str, pd.DataFrame] = {}

    for cost_bps in (7.5, 15.0):
        for slippage_bps in (20.0, 50.0, 100.0):
            results = simulate_scenarios(
                base_weights,
                base_daily,
                asset_returns,
                opens,
                closes,
                cost_bps,
                slippage_bps,
            )
            baseline = results["baseline"]
            if cost_bps == 7.5 and slippage_bps == 20.0:
                standard_results = results
            for scenario, candidate in results.items():
                for period in PERIODS:
                    metric_rows.append(
                        metric_row(
                            scenario,
                            cost_bps,
                            slippage_bps,
                            period,
                            candidate,
                            baseline,
                        )
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(DESTINATION / "metrics.csv", index=False)

    event_rows: list[dict[str, float | str]] = []
    baseline = standard_results["baseline"]
    for scenario, candidate in standard_results.items():
        for date in EVENT_DATES:
            timestamp = pd.Timestamp(date)
            if timestamp not in candidate.index:
                continue
            event_rows.append(
                {
                    "scenario": scenario,
                    "date": date,
                    "candidate_return": float(
                        candidate.loc[timestamp, "net_return"]
                    ),
                    "baseline_return": float(
                        baseline.loc[timestamp, "net_return"]
                    ),
                    "return_delta": float(
                        candidate.loc[timestamp, "net_return"]
                        - baseline.loc[timestamp, "net_return"]
                    ),
                    "candidate_turnover": float(
                        candidate.loc[timestamp, "turnover"]
                    ),
                    "triggered": int(
                        candidate.loc[timestamp, "triggered"]
                    ),
                }
            )
    events = pd.DataFrame(event_rows)
    events.to_csv(DESTINATION / "events.csv", index=False)

    bootstrap_rows: list[dict[str, float | int | str]] = []
    for scenario, candidate in standard_results.items():
        if scenario == "baseline":
            continue
        for block_days in (21, 63, 126):
            bootstrap_rows.append(
                {
                    "scenario": scenario,
                    **circular_block_relative_bootstrap(
                        candidate["net_return"],
                        baseline["net_return"],
                        block_days,
                    ),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(DESTINATION / "block_bootstrap.csv", index=False)

    seed_rows: list[dict[str, float | int | str]] = []
    for seed in (7, 42, 123):
        member_weights, member_daily, member_returns = load_member(
            seed,
            asset_returns,
        )
        results = simulate_scenarios(
            member_weights,
            member_daily,
            member_returns,
            opens,
            closes,
            7.5,
            20.0,
        )
        member_baseline = results["baseline"]
        for scenario, candidate in results.items():
            if scenario == "baseline":
                continue
            row = metric_row(
                scenario,
                7.5,
                20.0,
                "complete_2015_present",
                candidate,
                member_baseline,
            )
            seed_rows.append({"seed": seed, **row})
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(DESTINATION / "seed_robustness.csv", index=False)

    standard = metrics.loc[
        (metrics["cost_bps"] == 7.5)
        & (metrics["emergency_slippage_bps"] == 20.0)
    ]
    pivot = standard.pivot(
        index="scenario",
        columns="period",
        values="cagr_delta_vs_baseline",
    )
    eligible_names = pivot.index[
        (pivot[list(PERIODS)] > 0.0).all(axis=1)
    ]
    eligible = standard.loc[
        (standard["period"] == "complete_2015_present")
        & standard["scenario"].isin(eligible_names)
    ].sort_values("cagr_delta_vs_baseline", ascending=False)

    print(f"All-period positive scenarios: {len(eligible_names)}")
    print("\nEligible standard scenarios:")
    print(
        eligible[
            [
                "scenario",
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
                "worst_day_delta_vs_baseline",
                "positive_capture",
                "annualized_turnover",
                "annualized_cost",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nSeed robustness for eligible scenarios:")
    print(
        seeds.loc[seeds["scenario"].isin(eligible_names)][
            [
                "seed",
                "scenario",
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
