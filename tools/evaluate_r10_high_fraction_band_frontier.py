from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.evaluate_r10_combined_tail_capital import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    metrics_for_period,
    selected_guard,
)
from tools.evaluate_r10_gde_capital_efficiency import (
    join_live_gde,
    load_adjusted_open_close,
    simulate_gde_substitution,
)
from tools.evaluate_r10_gde_tracking_uncertainty import (
    ADJUSTED_CLOSE,
    bootstrap_tracking_uncertainty,
    gde_implementation_residual,
)
from tools.evaluate_smh_dynamic_guard import (
    OpenGapGuard,
    load_inputs as load_normal_guard_inputs,
    simulate as simulate_guard,
)
from tools.evaluate_smh_long_proxy_validation import (
    load_inputs as load_proxy_guard_inputs,
)


OUTPUT = Path("output/r10_high_fraction_band_frontier")
FRACTIONS = (0.40, 0.50, 0.60)
BANDS = (0.0025, 0.0100, 0.0200)


def select_highest_feasible(
    metrics: pd.DataFrame,
    tracking: pd.DataFrame,
) -> tuple[float, float]:
    full_periods = {
        "normal_synthetic": "complete_2015_2026",
        "proxy_synthetic": "complete_2006_2026",
        "normal_live": "live_2022_2026",
    }
    feasible: list[tuple[float, float]] = []
    candidates = (
        metrics[["substitution_fraction", "gde_no_trade_band"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for fraction_value, band_value in candidates:
        fraction = float(fraction_value)
        band = float(band_value)
        valid = True
        for sample, period in full_periods.items():
            primary = metrics.loc[
                metrics["sample"].eq(sample)
                & metrics["period"].eq(period)
                & metrics["scenario"].eq("current_liquidity")
                & metrics["substitution_fraction"].eq(fraction)
                & metrics["gde_no_trade_band"].eq(band)
            ].iloc[0]
            stress = metrics.loc[
                metrics["sample"].eq(sample)
                & metrics["period"].eq(period)
                & metrics["scenario"].eq("cost_stress")
                & metrics["substitution_fraction"].eq(fraction)
                & metrics["gde_no_trade_band"].eq(band)
            ].iloc[0]
            valid &= float(primary["cagr_delta"]) > 0.0
            valid &= float(primary["sharpe_delta"]) >= 0.0
            valid &= float(primary["candidate_max_drawdown"]) >= -0.18
            valid &= float(stress["cagr_delta"]) > 0.0
            valid &= float(stress["candidate_max_drawdown"]) >= -0.18
        track = tracking.loc[
            tracking["substitution_fraction"].eq(fraction)
            & tracking["gde_no_trade_band"].eq(band)
        ].iloc[0]
        valid &= float(track["probability_all_objectives"]) >= 0.95
        valid &= float(track["max_drawdown_p05"]) >= -0.18
        if valid:
            feasible.append((fraction, band))
    if not feasible:
        raise RuntimeError("No high-fraction candidate is feasible")
    maximum_fraction = max(fraction for fraction, _ in feasible)
    minimum_band = min(
        band
        for fraction, band in feasible
        if fraction == maximum_fraction
    )
    return maximum_fraction, minimum_band


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
    normal_weights, normal_daily, _, _ = load_normal_guard_inputs()
    proxy_weights, proxy_daily, _, _ = load_proxy_guard_inputs(
        Path("output") / PROXY_DIRECTORY
    )
    samples = {
        "normal_synthetic": {
            "directory": NORMAL_DIRECTORY,
            "weights": normal_weights,
            "daily": normal_daily,
            "opens": normal_opens,
            "closes": normal_closes,
            "mode": "synthetic",
            "start": "2015-01-01",
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
            "periods": {
                "live_2022_2026": ("2022-03-17", "2026-12-31"),
                "live_2022": ("2022-03-17", "2022-12-31"),
                "live_2023_2024": ("2023-01-01", "2024-12-31"),
                "live_2025_2026": ("2025-01-01", "2026-12-31"),
            },
        },
    }
    scenarios = {
        "current_liquidity": (20.0, 40.0),
        "cost_stress": (50.0, 75.0),
    }
    rows: list[dict[str, float | int | str]] = []
    current_candidates: dict[
        tuple[str, float, float], pd.DataFrame
    ] = {}
    baselines: dict[str, pd.DataFrame] = {}
    for sample, settings in samples.items():
        weights = settings["weights"]
        daily = settings["daily"]
        opens = settings["opens"]
        closes = settings["closes"]
        periods = settings["periods"]
        start = str(settings["start"])
        assert isinstance(weights, pd.DataFrame)
        assert isinstance(daily, pd.DataFrame)
        assert isinstance(opens, pd.DataFrame)
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)
        baseline_guard, _ = simulate_guard(
            weights,
            daily,
            opens,
            closes,
            OpenGapGuard("baseline"),
            start_date=start,
        )
        baseline = simulate_gde_substitution(
            str(settings["directory"]),
            opens,
            closes,
            substitution_fraction=0.0,
            gate_mode="always",
            gde_return_mode=str(settings["mode"]),
            start_date=start,
            end_date=None,
            gde_one_way_cost_bps=40.0,
        )
        common_baseline = baseline.index.intersection(
            baseline_guard.index
        )
        reconstruction_error = float(
            (
                baseline.loc[common_baseline, "net_return"]
                - baseline_guard.loc[common_baseline, "net_return"]
            )
            .abs()
            .max()
        )
        baselines[sample] = baseline
        for scenario, (
            emergency_slippage_bps,
            gde_cost_bps,
        ) in scenarios.items():
            guard_daily, guard_weights = simulate_guard(
                weights,
                daily,
                opens,
                closes,
                selected_guard(
                    emergency_slippage_bps,
                    post_trigger_cap=0.15,
                    relative_gap_trigger=-0.005,
                ),
                start_date=start,
            )
            for fraction in FRACTIONS:
                for band in BANDS:
                    candidate = simulate_gde_substitution(
                        str(settings["directory"]),
                        opens,
                        closes,
                        substitution_fraction=fraction,
                        gate_mode="growth_linked",
                        gde_return_mode=str(settings["mode"]),
                        start_date=start,
                        end_date=None,
                        gde_one_way_cost_bps=gde_cost_bps,
                        weights_override=guard_weights,
                        daily_override=guard_daily,
                        extra_slippage=guard_daily["slippage_cost"],
                        gde_no_trade_band=band,
                    )
                    common = baseline.index.intersection(candidate.index)
                    for period, (period_start, period_end) in periods.items():
                        selected = common[
                            (common >= period_start)
                            & (common <= period_end)
                        ]
                        rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario,
                                "substitution_fraction": fraction,
                                "gde_no_trade_band": band,
                                "period": period,
                                "baseline_reconstruction_error": (
                                    reconstruction_error
                                ),
                                **metrics_for_period(
                                    baseline.loc[selected, "net_return"],
                                    candidate.loc[selected, "net_return"],
                                ),
                            }
                        )
                    if scenario == "current_liquidity":
                        current_candidates[(sample, fraction, band)] = (
                            candidate
                        )
                        fraction_label = int(round(fraction * 100))
                        band_label = int(round(band * 10_000))
                        candidate.to_csv(
                            OUTPUT
                            / (
                                f"{sample}_fraction{fraction_label}"
                                f"_band{band_label}bp_daily.csv"
                            ),
                            index_label="date",
                        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    prices = pd.read_csv(
        ADJUSTED_CLOSE,
        index_col=0,
        parse_dates=True,
    )
    residual = gde_implementation_residual(prices)
    proxy_baseline = baselines["proxy_synthetic"]["net_return"]
    tracking_rows: list[dict[str, float | int | str]] = []
    for fraction in FRACTIONS:
        for band in BANDS:
            tracking_rows.append(
                {
                    "substitution_fraction": fraction,
                    "gde_no_trade_band": band,
                    **bootstrap_tracking_uncertainty(
                        proxy_baseline,
                        current_candidates[
                            ("proxy_synthetic", fraction, band)
                        ],
                        residual,
                    ),
                }
            )
    tracking = pd.DataFrame(tracking_rows)
    tracking.to_csv(OUTPUT / "tracking_frontier.csv", index=False)
    selected_fraction, selected_band = select_highest_feasible(
        metrics,
        tracking,
    )
    pd.Series(
        {
            "selected_fraction": selected_fraction,
            "selected_band": selected_band,
            "selection_rule": (
                "highest_fraction_then_smallest_band_meeting_primary_"
                "cagr_sharpe_mdd_stress_cagr_mdd_and_tracking_budget"
            ),
        },
        name="value",
    ).to_csv(OUTPUT / "selection.csv")
    full_periods = {
        "complete_2015_2026",
        "complete_2006_2026",
        "live_2022_2026",
    }
    print("Full-period metrics:")
    print(
        metrics.loc[metrics["period"].isin(full_periods)]
        [
            [
                "sample",
                "scenario",
                "substitution_fraction",
                "gde_no_trade_band",
                "cagr_delta",
                "sharpe_delta",
                "candidate_max_drawdown",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nTracking frontier:")
    print(tracking.round(6).to_string(index=False))
    print(
        f"\nSelected: fraction={selected_fraction:.0%}, "
        f"band={selected_band:.2%}"
    )
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
