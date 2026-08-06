from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from regime_strategy.report import performance_metrics
from tools.evaluate_r10_gde_capital_efficiency import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    join_live_gde,
    load_adjusted_open_close,
    simulate_gde_substitution,
)
from tools.evaluate_r10_gde_tracking_uncertainty import (
    ADJUSTED_CLOSE,
    bootstrap_tracking_uncertainty,
    gde_implementation_residual,
)
from tools.evaluate_r11_diversified_capital_grid import (
    load_strategy_inputs,
    scale_non_cash_weights,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)


OUTPUT = Path("output/r11_selected_audit")
RISK_MULTIPLIER = 1.025
GDE_FRACTION = 0.40
ROLLOUT_SHARES = (0.25, 0.50, 0.75, 1.00)


def candidate_name() -> str:
    return (
        f"risk{RISK_MULTIPLIER:.3f}"
        f"_gde{int(round(GDE_FRACTION * 100))}"
    )


def drawdown_episode(returns: pd.Series) -> dict[str, float | int | str]:
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = wealth.loc[:trough].idxmax()
    recovery_candidates = wealth.loc[trough:].loc[
        wealth.loc[trough:] >= wealth.loc[peak]
    ]
    recovery = (
        recovery_candidates.index[0]
        if len(recovery_candidates)
        else None
    )
    return {
        "peak_date": peak.date().isoformat(),
        "trough_date": trough.date().isoformat(),
        "recovery_date": (
            recovery.date().isoformat() if recovery is not None else ""
        ),
        "max_drawdown": float(drawdown.loc[trough]),
        "peak_to_trough_trading_days": int(
            returns.loc[peak:trough].shape[0] - 1
        ),
        "trough_to_recovery_trading_days": (
            int(returns.loc[trough:recovery].shape[0] - 1)
            if recovery is not None
            else -1
        ),
    }


def metric_delta(
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict[str, float]:
    base = performance_metrics(baseline)
    trial = performance_metrics(candidate)
    return {
        "baseline_cagr": base["cagr"],
        "candidate_cagr": trial["cagr"],
        "cagr_delta": trial["cagr"] - base["cagr"],
        "baseline_sharpe": base["sharpe"],
        "candidate_sharpe": trial["sharpe"],
        "sharpe_delta": trial["sharpe"] - base["sharpe"],
        "baseline_max_drawdown": base["max_drawdown"],
        "candidate_max_drawdown": trial["max_drawdown"],
        "max_drawdown_delta": (
            trial["max_drawdown"] - base["max_drawdown"]
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(PROXY_OPEN_CLOSE)
    live_opens, live_closes = join_live_gde(
        normal_opens,
        normal_closes,
    )
    normal_weights, normal_daily = load_strategy_inputs(NORMAL_DIRECTORY)
    proxy_weights, proxy_daily = load_strategy_inputs(PROXY_DIRECTORY)
    samples = {
        "normal_synthetic": {
            "directory": NORMAL_DIRECTORY,
            "weights": normal_weights,
            "daily": normal_daily,
            "opens": normal_opens,
            "closes": normal_closes,
            "mode": "synthetic",
            "start": "2015-01-01",
            "full_period": "complete_2015_2026",
            "periods": {
                "development_2015_2021": (
                    "2015-01-01",
                    "2021-12-31",
                ),
                "holdout_2022_2025": (
                    "2022-01-01",
                    "2025-12-31",
                ),
                "recent_2026": ("2026-01-01", "2026-12-31"),
                "complete_2015_2026": (
                    "2015-01-01",
                    "2026-12-31",
                ),
            },
        },
        "proxy_synthetic": {
            "directory": PROXY_DIRECTORY,
            "weights": proxy_weights,
            "daily": proxy_daily,
            "opens": proxy_opens,
            "closes": proxy_closes,
            "mode": "synthetic",
            "start": "2006-08-01",
            "full_period": "complete_2006_2026",
            "periods": {
                "early_2006_2014": ("2006-08-01", "2014-12-31"),
                "late_2015_2026": ("2015-01-01", "2026-12-31"),
                "complete_2006_2026": (
                    "2006-08-01",
                    "2026-12-31",
                ),
            },
        },
        "normal_live": {
            "directory": NORMAL_DIRECTORY,
            "weights": normal_weights,
            "daily": normal_daily,
            "opens": live_opens,
            "closes": live_closes,
            "mode": "live",
            "start": "2022-03-17",
            "full_period": "live_2022_2026",
            "periods": {
                "live_2022": ("2022-03-17", "2022-12-31"),
                "live_2023_2024": ("2023-01-01", "2024-12-31"),
                "live_2025_2026": ("2025-01-01", "2026-12-31"),
                "live_2022_2026": ("2022-03-17", "2026-12-31"),
            },
        },
    }
    metric_rows: list[dict[str, float | str]] = []
    annual_rows: list[dict[str, float | int | str]] = []
    drawdown_rows: list[dict[str, float | int | str]] = []
    leave_one_year_out_rows: list[dict[str, float | int | str]] = []
    paths: dict[
        tuple[str, str, float],
        tuple[pd.Series, pd.DataFrame],
    ] = {}

    for sample, settings in samples.items():
        weights = settings["weights"]
        daily = settings["daily"]
        opens = settings["opens"]
        closes = settings["closes"]
        periods = settings["periods"]
        assert isinstance(weights, pd.DataFrame)
        assert isinstance(daily, pd.DataFrame)
        assert isinstance(opens, pd.DataFrame)
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)
        levered_weights = scale_non_cash_weights(
            weights,
            RISK_MULTIPLIER,
        )
        for scenario in COST_SCENARIOS:
            baseline_frame = simulate_gde_substitution(
                str(settings["directory"]),
                opens,
                closes,
                substitution_fraction=0.0,
                gate_mode="always",
                gde_return_mode=str(settings["mode"]),
                start_date=str(settings["start"]),
                end_date=None,
                base_one_way_cost_bps=scenario.base_one_way_cost_bps,
                gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,
                financing_spread_bps=scenario.financing_spread_bps,
            )
            guard_daily, guard_weights = simulate_guard(
                levered_weights,
                daily,
                opens,
                closes,
                guard_for_multiplier(
                    RISK_MULTIPLIER,
                    scenario.emergency_slippage_bps,
                ),
                cost_bps=scenario.base_one_way_cost_bps,
                start_date=str(settings["start"]),
            )
            candidate_frame = simulate_gde_substitution(
                str(settings["directory"]),
                opens,
                closes,
                substitution_fraction=GDE_FRACTION,
                gate_mode="growth_linked",
                gde_return_mode=str(settings["mode"]),
                start_date=str(settings["start"]),
                end_date=None,
                base_one_way_cost_bps=scenario.base_one_way_cost_bps,
                gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,
                financing_spread_bps=scenario.financing_spread_bps,
                weights_override=guard_weights,
                daily_override=guard_daily,
                extra_slippage=guard_daily["slippage_cost"],
                gde_no_trade_band=GDE_NO_TRADE_BAND,
            )
            common = baseline_frame.index.intersection(
                candidate_frame.index
            )
            baseline = baseline_frame.loc[common, "net_return"]
            full_candidate = candidate_frame.loc[common, "net_return"]
            for share in ROLLOUT_SHARES:
                staged = (1.0 - share) * baseline + share * full_candidate
                staged_frame = candidate_frame.loc[common].copy()
                staged_frame["net_return"] = staged
                staged_frame["gde_target"] *= share
                paths[(sample, scenario.name, share)] = (
                    baseline,
                    staged_frame,
                )
                label = int(round(share * 100))
                staged_frame.to_csv(
                    OUTPUT
                    / (
                        f"{sample}_{scenario.name}"
                        f"_share{label}_daily.csv"
                    ),
                    index_label="date",
                )
                for period, (period_start, period_end) in periods.items():
                    selected = common[
                        (common >= period_start)
                        & (common <= period_end)
                    ]
                    metric_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "rollout_share": share,
                            "period": period,
                            **metric_delta(
                                baseline.loc[selected],
                                staged.loc[selected],
                            ),
                        }
                    )
                for year, year_returns in staged.groupby(staged.index.year):
                    base_year = baseline.reindex(year_returns.index)
                    annual_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "rollout_share": share,
                            "year": int(year),
                            "r9_return": float(
                                (1.0 + base_year).prod() - 1.0
                            ),
                            "candidate_return": float(
                                (1.0 + year_returns).prod() - 1.0
                            ),
                            "relative_log_return": float(
                                (
                                    np.log1p(year_returns)
                                    - np.log1p(base_year)
                                ).sum()
                            ),
                        }
                    )
                drawdown_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "rollout_share": share,
                        **drawdown_episode(staged),
                    }
                )
                if sample != "normal_live":
                    for omitted_year in sorted(staged.index.year.unique()):
                        keep = staged.index.year != omitted_year
                        leave_one_year_out_rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario.name,
                                "rollout_share": share,
                                "omitted_year": int(omitted_year),
                                **metric_delta(
                                    baseline.loc[keep],
                                    staged.loc[keep],
                                ),
                            }
                        )

    metrics = pd.DataFrame(metric_rows)
    annual = pd.DataFrame(annual_rows)
    drawdowns = pd.DataFrame(drawdown_rows)
    leave_one_year_out = pd.DataFrame(leave_one_year_out_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    annual.to_csv(OUTPUT / "annual_returns.csv", index=False)
    drawdowns.to_csv(OUTPUT / "maximum_drawdown_episodes.csv", index=False)
    leave_one_year_out.to_csv(
        OUTPUT / "leave_one_year_out.csv",
        index=False,
    )

    prices = pd.read_csv(
        ADJUSTED_CLOSE,
        index_col=0,
        parse_dates=True,
    )
    residual = gde_implementation_residual(prices)
    tracking_rows = []
    for scenario in COST_SCENARIOS:
        for share in (0.25, 0.50, 1.00):
            baseline, staged_frame = paths[
                ("proxy_synthetic", scenario.name, share)
            ]
            tracking_rows.append(
                {
                    "scenario": scenario.name,
                    "rollout_share": share,
                    **bootstrap_tracking_uncertainty(
                        baseline,
                        staged_frame,
                        residual,
                    ),
                }
            )
    tracking = pd.DataFrame(tracking_rows)
    tracking.to_csv(
        OUTPUT / "tracking_uncertainty.csv",
        index=False,
    )

    full_periods = {
        str(settings["full_period"])
        for settings in samples.values()
    }
    full = metrics.loc[metrics["period"].isin(full_periods)].copy()
    stage25 = full.loc[full["rollout_share"].eq(0.25)]
    stage25_tracking = tracking.loc[tracking["rollout_share"].eq(0.25)]
    stage25_loo = leave_one_year_out.loc[
        leave_one_year_out["rollout_share"].eq(0.25)
    ]
    acceptance = pd.Series(
        {
            "candidate": candidate_name(),
            "initial_rollout_share": 0.25,
            "full_period_cagr_positive_all_samples_costs": bool(
                stage25["cagr_delta"].gt(0.0).all()
            ),
            "stress_drawdown_within_18pct_all_samples": bool(
                stage25.loc[
                    stage25["scenario"].eq("cost_stress"),
                    "candidate_max_drawdown",
                ]
                .ge(-0.18)
                .all()
            ),
            "leave_one_year_out_cagr_positive": bool(
                stage25_loo["cagr_delta"].gt(0.0).all()
            ),
            "tracking_positive_cagr_probability_at_least_95pct": bool(
                stage25_tracking[
                    "probability_positive_cagr_delta"
                ]
                .ge(0.95)
                .all()
            ),
            "tracking_drawdown_probability_at_least_95pct": bool(
                stage25_tracking[
                    "probability_drawdown_within_18pct"
                ]
                .ge(0.95)
                .all()
            ),
            "tracking_all_objectives_probability_at_least_95pct": bool(
                stage25_tracking["probability_all_objectives"]
                .ge(0.95)
                .all()
            ),
        },
        name="value",
    )
    acceptance["all_non_family_gates_pass"] = bool(
        acceptance.drop(
            labels=["candidate", "initial_rollout_share"]
        ).all()
    )
    acceptance.to_csv(OUTPUT / "acceptance.csv")

    print("Full-period rollout metrics:")
    print(
        full[
            [
                "sample",
                "scenario",
                "rollout_share",
                "cagr_delta",
                "sharpe_delta",
                "candidate_max_drawdown",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nTracking uncertainty:")
    print(tracking.round(6).to_string(index=False))
    print("\nAcceptance before strict-family audit:")
    print(acceptance.to_string())
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
