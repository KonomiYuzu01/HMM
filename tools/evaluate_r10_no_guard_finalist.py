from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.evaluate_r10_combined_tail_capital import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    metrics_for_period,
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
    load_inputs as load_normal_inputs,
)
from tools.evaluate_smh_long_proxy_validation import (
    load_inputs as load_proxy_inputs,
)


OUTPUT = Path("output/r10_no_guard_finalist")
SUBSTITUTION_FRACTION = 0.50
GDE_NO_TRADE_BAND = 0.02


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
    normal_weights, normal_daily, _, _ = load_normal_inputs()
    proxy_weights, proxy_daily, _, _ = load_proxy_inputs(
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
        "current_liquidity": 40.0,
        "cost_stress": 75.0,
    }
    rows: list[dict[str, float | int | str]] = []
    current_candidates: dict[str, pd.DataFrame] = {}
    baselines: dict[str, pd.DataFrame] = {}
    for sample, settings in samples.items():
        opens = settings["opens"]
        closes = settings["closes"]
        weights = settings["weights"]
        daily = settings["daily"]
        periods = settings["periods"]
        start = str(settings["start"])
        assert isinstance(opens, pd.DataFrame)
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(weights, pd.DataFrame)
        assert isinstance(daily, pd.DataFrame)
        assert isinstance(periods, dict)
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
            weights_override=weights,
            daily_override=daily,
        )
        baselines[sample] = baseline
        for scenario, cost_bps in scenarios.items():
            candidate = simulate_gde_substitution(
                str(settings["directory"]),
                opens,
                closes,
                substitution_fraction=SUBSTITUTION_FRACTION,
                gate_mode="growth_linked",
                gde_return_mode=str(settings["mode"]),
                start_date=start,
                end_date=None,
                gde_one_way_cost_bps=cost_bps,
                weights_override=weights,
                daily_override=daily,
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
                        "period": period,
                        **metrics_for_period(
                            baseline.loc[selected, "net_return"],
                            candidate.loc[selected, "net_return"],
                        ),
                    }
                )
            if scenario == "current_liquidity":
                current_candidates[sample] = candidate
                candidate.to_csv(
                    OUTPUT / f"{sample}_daily.csv",
                    index_label="date",
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    prices = pd.read_csv(
        ADJUSTED_CLOSE,
        index_col=0,
        parse_dates=True,
    )
    tracking = pd.DataFrame(
        [
            {
                "candidate": "no_guard_fraction50_band200bp",
                **bootstrap_tracking_uncertainty(
                    baselines["proxy_synthetic"]["net_return"],
                    current_candidates["proxy_synthetic"],
                    gde_implementation_residual(prices),
                ),
            }
        ]
    )
    tracking.to_csv(OUTPUT / "tracking_bootstrap.csv", index=False)
    print("Metrics:")
    print(
        metrics[
            [
                "sample",
                "scenario",
                "period",
                "cagr_delta",
                "sharpe_delta",
                "candidate_max_drawdown",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nTracking:")
    print(tracking.round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
