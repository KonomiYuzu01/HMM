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


OUTPUT = Path("output/r10_calm_boost_capital")
GDE_NO_TRADE_BAND = 0.02
CANDIDATES = (
    ("fixed_40", 0.40, 0.40),
    ("fixed_50", 0.50, 0.50),
    ("fixed_60", 0.60, 0.60),
    ("calm_40_to_75", 0.40, 0.75),
    ("calm_40_to_100", 0.40, 1.00),
    ("calm_50_to_75", 0.50, 0.75),
    ("calm_50_to_100", 0.50, 1.00),
    ("calm_60_to_100", 0.60, 1.00),
)


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
    baselines: dict[str, pd.DataFrame] = {}
    current_candidates: dict[tuple[str, str], pd.DataFrame] = {}
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
            for candidate_name, minimum_fraction, maximum_fraction in (
                CANDIDATES
            ):
                candidate = simulate_gde_substitution(
                    str(settings["directory"]),
                    opens,
                    closes,
                    substitution_fraction=maximum_fraction,
                    gate_mode="base_plus_calm",
                    gde_return_mode=str(settings["mode"]),
                    start_date=start,
                    end_date=None,
                    gde_one_way_cost_bps=gde_cost_bps,
                    minimum_substitution_fraction=minimum_fraction,
                    weights_override=guard_weights,
                    daily_override=guard_daily,
                    extra_slippage=guard_daily["slippage_cost"],
                    gde_no_trade_band=GDE_NO_TRADE_BAND,
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
                            "candidate": candidate_name,
                            "minimum_fraction": minimum_fraction,
                            "maximum_fraction": maximum_fraction,
                            "gde_no_trade_band": GDE_NO_TRADE_BAND,
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
                    current_candidates[(sample, candidate_name)] = candidate
                    candidate.to_csv(
                        OUTPUT / f"{sample}_{candidate_name}_daily.csv",
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
    tracking_rows = []
    for candidate_name, _, _ in CANDIDATES:
        tracking_rows.append(
            {
                "candidate": candidate_name,
                **bootstrap_tracking_uncertainty(
                    proxy_baseline,
                    current_candidates[
                        ("proxy_synthetic", candidate_name)
                    ],
                    residual,
                ),
            }
        )
    tracking = pd.DataFrame(tracking_rows)
    tracking.to_csv(OUTPUT / "tracking_uncertainty.csv", index=False)

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
                "candidate",
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
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
