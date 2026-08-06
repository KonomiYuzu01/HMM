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


OUTPUT = Path("output/r10_gde_no_trade_band")
SUBSTITUTION_FRACTION = 0.40
BANDS = (0.0, 0.0025, 0.0050, 0.0100, 0.0150, 0.0200)
CURRENT_CAGR_TOLERANCE = 0.0005
MINIMUM_OBJECTIVE_PROBABILITY = 0.95


def select_smallest_effective_band(
    metrics: pd.DataFrame,
    tracking: pd.DataFrame,
) -> float:
    full_periods = {
        "normal_synthetic": "complete_2015_2026",
        "proxy_synthetic": "complete_2006_2026",
        "normal_live": "live_2022_2026",
    }
    eligible: list[float] = []
    for band in sorted(
        float(value)
        for value in metrics["gde_no_trade_band"].unique()
        if float(value) > 0.0
    ):
        valid = True
        for sample, period in full_periods.items():
            subset = metrics.loc[
                metrics["sample"].eq(sample)
                & metrics["period"].eq(period)
            ]
            no_band_current = subset.loc[
                subset["scenario"].eq("current_liquidity")
                & subset["gde_no_trade_band"].eq(0.0)
            ].iloc[0]
            band_current = subset.loc[
                subset["scenario"].eq("current_liquidity")
                & subset["gde_no_trade_band"].eq(band)
            ].iloc[0]
            no_band_stress = subset.loc[
                subset["scenario"].eq("cost_stress")
                & subset["gde_no_trade_band"].eq(0.0)
            ].iloc[0]
            band_stress = subset.loc[
                subset["scenario"].eq("cost_stress")
                & subset["gde_no_trade_band"].eq(band)
            ].iloc[0]
            valid &= (
                float(band_current["candidate_cagr"])
                >= float(no_band_current["candidate_cagr"])
                - CURRENT_CAGR_TOLERANCE
            )
            valid &= (
                float(band_stress["candidate_cagr"])
                >= float(no_band_stress["candidate_cagr"])
            )
            valid &= (
                float(band_stress["candidate_sharpe"])
                >= float(no_band_stress["candidate_sharpe"])
            )
        track = tracking.loc[
            tracking["gde_no_trade_band"].eq(band)
        ].iloc[0]
        valid &= (
            float(track["probability_all_objectives"])
            >= MINIMUM_OBJECTIVE_PROBABILITY
        )
        valid &= float(track["max_drawdown_p05"]) >= -0.18
        if valid:
            eligible.append(band)
    return min(eligible) if eligible else 0.0


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
        tuple[str, float], pd.DataFrame
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
            for band in BANDS:
                candidate = simulate_gde_substitution(
                    str(settings["directory"]),
                    opens,
                    closes,
                    substitution_fraction=SUBSTITUTION_FRACTION,
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
                years = len(common) / 252.0
                for period, (period_start, period_end) in periods.items():
                    selected = common[
                        (common >= period_start)
                        & (common <= period_end)
                    ]
                    rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario,
                            "gde_no_trade_band": band,
                            "period": period,
                            "baseline_reconstruction_error": (
                                reconstruction_error
                            ),
                            **metrics_for_period(
                                baseline.loc[selected, "net_return"],
                                candidate.loc[selected, "net_return"],
                            ),
                            "full_sample_gde_annualized_turnover": (
                                float(candidate.loc[common, "gde_trade"].sum())
                                / years
                            ),
                            "full_sample_annualized_trading_cost": (
                                float(
                                    candidate.loc[
                                        common,
                                        "trading_cost",
                                    ].sum()
                                )
                                / years
                            ),
                        }
                    )
                if scenario == "current_liquidity":
                    current_candidates[(sample, band)] = candidate
                    label = int(round(band * 10_000))
                    candidate.to_csv(
                        OUTPUT
                        / f"{sample}_band{label}bp_daily.csv",
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
    for band in BANDS:
        tracking_rows.append(
            {
                "gde_no_trade_band": band,
                **bootstrap_tracking_uncertainty(
                    proxy_baseline,
                    current_candidates[("proxy_synthetic", band)],
                    residual,
                ),
            }
        )
    tracking = pd.DataFrame(tracking_rows)
    tracking.to_csv(OUTPUT / "tracking_by_band.csv", index=False)
    selected_band = select_smallest_effective_band(metrics, tracking)
    pd.Series(
        {
            "selection_rule": (
                "smallest_nonzero_band_preserving_current_cagr_and_"
                "improving_all_cost_stress_full_period_metrics"
            ),
            "selected_band": selected_band,
            "current_cagr_tolerance": CURRENT_CAGR_TOLERANCE,
            "minimum_objective_probability": (
                MINIMUM_OBJECTIVE_PROBABILITY
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
                "gde_no_trade_band",
                "cagr_delta",
                "sharpe_delta",
                "candidate_max_drawdown",
                "full_sample_gde_annualized_turnover",
                "full_sample_annualized_trading_cost",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nTracking by band:")
    print(tracking.round(6).to_string(index=False))
    print(f"\nSelected band: {selected_band:.2%}")
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
