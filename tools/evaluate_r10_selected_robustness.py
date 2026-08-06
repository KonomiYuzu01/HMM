from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_r10_combined_tail_capital import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    selected_guard,
)
from tools.evaluate_r10_gde_capital_efficiency import (
    load_adjusted_open_close,
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import (
    scale_non_cash_weights,
)
from tools.evaluate_smh_dynamic_guard import (
    OpenGapGuard,
    simulate as simulate_guard,
)
from tools.evaluate_smh_dynamic_guard_robustness import event_clusters


OUTPUT = Path("output/r10_final_candidate_robustness")
SELECTED_PATHS = Path("output/r10_high_fraction_band_frontier")
SUBSTITUTION_FRACTION = 0.50
GDE_NO_TRADE_BAND = 0.02
RISK_MULTIPLIER = 1.0
CANDIDATE_PATH_TEMPLATE = (
    "{sample}_fraction50_band200bp_daily.csv"
)


def selected_configuration(slippage_bps: float) -> OpenGapGuard:
    return selected_guard(
        slippage_bps,
        post_trigger_cap=0.15,
        relative_gap_trigger=-0.005,
        minimum_semis_weight=0.60 * RISK_MULTIPLIER,
    )


def annualized_relative_log_return(
    candidate: pd.Series,
    baseline: pd.Series,
) -> float:
    aligned = pd.concat(
        [
            baseline.rename("baseline"),
            candidate.rename("candidate"),
        ],
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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(PROXY_OPEN_CLOSE)
    samples = {
        "normal": {
            "directory": NORMAL_DIRECTORY,
            "opens": normal_opens,
            "closes": normal_closes,
            "start": "2015-01-01",
            "periods": {
                "development_2015_2021": (
                    "2015-01-01",
                    "2021-12-31",
                ),
                "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
                "complete_2015_2026": ("2015-01-01", "2026-12-31"),
            },
            "candidate_path": (
                SELECTED_PATHS
                / CANDIDATE_PATH_TEMPLATE.format(
                    sample="normal_synthetic"
                )
            ),
            "baseline_path": (
                Path("output/r10_combined_tail_capital")
                / "normal_synthetic_baseline_daily.csv"
            ),
        },
        "proxy": {
            "directory": PROXY_DIRECTORY,
            "opens": proxy_opens,
            "closes": proxy_closes,
            "start": "2006-08-01",
            "periods": {
                "early_2006_2014": ("2006-08-01", "2014-12-31"),
                "late_2015_2026": ("2015-01-01", "2026-12-31"),
                "complete_2006_2026": ("2006-08-01", "2026-12-31"),
            },
            "candidate_path": (
                SELECTED_PATHS
                / CANDIDATE_PATH_TEMPLATE.format(
                    sample="proxy_synthetic"
                )
            ),
            "baseline_path": (
                Path("output/r10_combined_tail_capital")
                / "proxy_synthetic_baseline_daily.csv"
            ),
        },
    }
    seed_rows: list[dict[str, float | int | str]] = []
    annual_rows: list[dict[str, float | int | str]] = []
    leave_one_year_rows: list[dict[str, float | int | str]] = []
    event_rows: list[dict[str, float | int | str]] = []
    leave_one_event_rows: list[dict[str, float | int | str]] = []
    for sample, settings in samples.items():
        directory = str(settings["directory"])
        opens = settings["opens"]
        closes = settings["closes"]
        start = str(settings["start"])
        periods = settings["periods"]
        assert isinstance(opens, pd.DataFrame)
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)
        for scenario, (
            base_cost_bps,
            slippage_bps,
            gde_cost_bps,
            financing_spread_bps,
        ) in {
            "current_liquidity": (7.5, 20.0, 40.0, 100.0),
            "cost_stress": (15.0, 50.0, 75.0, 150.0),
        }.items():
            for seed in (7, 42, 123):
                member_root = (
                    Path("output")
                    / directory
                    / "members"
                    / f"seed_{seed}"
                )
                weights = pd.read_csv(
                    member_root / "weights.csv",
                    index_col=0,
                    parse_dates=True,
                )
                daily = pd.read_csv(
                    member_root / "daily_returns.csv",
                    index_col=0,
                    parse_dates=True,
                )
                baseline_guard, _ = simulate_guard(
                    weights,
                    daily,
                    opens,
                    closes,
                    OpenGapGuard("baseline"),
                    cost_bps=base_cost_bps,
                    start_date=start,
                )
                baseline = simulate_gde_substitution(
                    directory,
                    opens,
                    closes,
                    substitution_fraction=0.0,
                    gate_mode="always",
                    gde_return_mode="synthetic",
                    start_date=start,
                    end_date=None,
                    base_one_way_cost_bps=base_cost_bps,
                    gde_one_way_cost_bps=gde_cost_bps,
                    financing_spread_bps=financing_spread_bps,
                    weights_override=weights,
                    daily_override=daily,
                )
                candidate_weights = scale_non_cash_weights(
                    weights,
                    RISK_MULTIPLIER,
                )
                guard_daily, guard_weights = simulate_guard(
                    candidate_weights,
                    daily,
                    opens,
                    closes,
                    selected_configuration(slippage_bps),
                    cost_bps=base_cost_bps,
                    start_date=start,
                )
                candidate = simulate_gde_substitution(
                    directory,
                    opens,
                    closes,
                    substitution_fraction=SUBSTITUTION_FRACTION,
                    gate_mode="growth_linked",
                    gde_return_mode="synthetic",
                    start_date=start,
                    end_date=None,
                    base_one_way_cost_bps=base_cost_bps,
                    gde_one_way_cost_bps=gde_cost_bps,
                    financing_spread_bps=financing_spread_bps,
                    weights_override=guard_weights,
                    daily_override=guard_daily,
                    extra_slippage=guard_daily["slippage_cost"],
                    gde_no_trade_band=GDE_NO_TRADE_BAND,
                )
                common = baseline.index.intersection(candidate.index)
                reconstruction_error = float(
                    (
                        baseline.loc[common, "net_return"]
                        - baseline_guard.loc[common, "net_return"]
                    )
                    .abs()
                    .max()
                )
                for period, (period_start, period_end) in periods.items():
                    selected = common[
                        (common >= period_start)
                        & (common <= period_end)
                    ]
                    base_metrics = performance_metrics(
                        baseline.loc[selected, "net_return"]
                    )
                    metrics = performance_metrics(
                        candidate.loc[selected, "net_return"]
                    )
                    seed_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario,
                            "seed": seed,
                            "period": period,
                            "cagr_delta": (
                                metrics["cagr"] - base_metrics["cagr"]
                            ),
                            "sharpe_delta": (
                                metrics["sharpe"] - base_metrics["sharpe"]
                            ),
                            "candidate_max_drawdown": metrics[
                                "max_drawdown"
                            ],
                            "guard_triggers": int(
                                guard_daily.loc[
                                    selected,
                                    "triggered",
                                ].sum()
                            ),
                            "baseline_reconstruction_error": (
                                reconstruction_error
                            ),
                        }
                    )

        baseline = pd.read_csv(
            settings["baseline_path"],
            index_col=0,
            parse_dates=True,
        )["net_return"]
        candidate = pd.read_csv(
            settings["candidate_path"],
            index_col=0,
            parse_dates=True,
        )["net_return"]
        aligned = pd.concat(
            [
                baseline.rename("baseline"),
                candidate.rename("candidate"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        for year, year_frame in aligned.groupby(aligned.index.year):
            annual_rows.append(
                {
                    "sample": sample,
                    "year": int(year),
                    "baseline_return": float(
                        (1.0 + year_frame["baseline"]).prod() - 1.0
                    ),
                    "candidate_return": float(
                        (1.0 + year_frame["candidate"]).prod() - 1.0
                    ),
                    "compound_delta": float(
                        (1.0 + year_frame["candidate"]).prod()
                        - (1.0 + year_frame["baseline"]).prod()
                    ),
                }
            )
        years = sorted(set(aligned.index.year))
        for omitted_year in years:
            kept = aligned.loc[aligned.index.year != omitted_year]
            leave_one_year_rows.append(
                {
                    "sample": sample,
                    "omitted_year": int(omitted_year),
                    "annualized_relative_log_return": (
                        annualized_relative_log_return(
                            kept["candidate"],
                            kept["baseline"],
                        )
                    ),
                }
            )

        weights = pd.read_csv(
            Path("output") / directory / "weights.csv",
            index_col=0,
            parse_dates=True,
        )
        daily = pd.read_csv(
            Path("output") / directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )
        baseline_guard, _ = simulate_guard(
            weights,
            daily,
            opens,
            closes,
            OpenGapGuard("baseline"),
            start_date=start,
        )
        guard_daily, _ = simulate_guard(
            weights,
            daily,
            opens,
            closes,
            selected_configuration(20.0),
            start_date=start,
        )
        clusters = event_clusters(
            guard_daily,
            baseline_guard,
            "selected",
        )
        for _, cluster in clusters.iterrows():
            trigger_date = pd.Timestamp(cluster["trigger_date"])
            recovery_date = pd.Timestamp(cluster["recovery_date"])
            selected_event = aligned.loc[trigger_date:recovery_date]
            event_rows.append(
                {
                    "sample": sample,
                    "trigger_date": trigger_date.date().isoformat(),
                    "recovery_date": recovery_date.date().isoformat(),
                    "sessions": len(selected_event),
                    "candidate_return": float(
                        (1.0 + selected_event["candidate"]).prod() - 1.0
                    ),
                    "baseline_return": float(
                        (1.0 + selected_event["baseline"]).prod() - 1.0
                    ),
                    "compound_delta": float(
                        (1.0 + selected_event["candidate"]).prod()
                        - (1.0 + selected_event["baseline"]).prod()
                    ),
                }
            )
            kept = aligned.loc[
                (aligned.index < trigger_date)
                | (aligned.index > recovery_date)
            ]
            leave_one_event_rows.append(
                {
                    "sample": sample,
                    "omitted_trigger_date": (
                        trigger_date.date().isoformat()
                    ),
                    "annualized_relative_log_return": (
                        annualized_relative_log_return(
                            kept["candidate"],
                            kept["baseline"],
                        )
                    ),
                }
            )
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(OUTPUT / "seed_metrics.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(
        OUTPUT / "annual_returns.csv",
        index=False,
    )
    leave_year = pd.DataFrame(leave_one_year_rows)
    leave_year.to_csv(
        OUTPUT / "leave_one_year_out.csv",
        index=False,
    )
    pd.DataFrame(event_rows).to_csv(
        OUTPUT / "event_contributions.csv",
        index=False,
    )
    leave_event = pd.DataFrame(leave_one_event_rows)
    leave_event.to_csv(
        OUTPUT / "leave_one_event_out.csv",
        index=False,
    )
    complete_periods = {
        "normal": "complete_2015_2026",
        "proxy": "complete_2006_2026",
    }
    complete = seed_metrics.loc[
        seed_metrics.apply(
            lambda row: row["period"] == complete_periods[row["sample"]],
            axis=1,
        )
    ]
    print("Seed robustness:")
    print(complete.round(6).to_string(index=False))
    print("\nLeave-one-year-out minima:")
    print(
        leave_year.groupby("sample")[
            "annualized_relative_log_return"
        ]
        .agg(["min", "median", "max"])
        .round(6)
        .to_string()
    )
    print("\nLeave-one-event-out minima:")
    print(
        leave_event.groupby("sample")[
            "annualized_relative_log_return"
        ]
        .agg(["min", "median", "max"])
        .round(6)
        .to_string()
    )
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
