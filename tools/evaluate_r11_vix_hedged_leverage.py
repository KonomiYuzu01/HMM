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
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    load_strategy_inputs,
    relative_log_return,
    scale_non_cash_weights,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)


OUTPUT = Path("output/r11_vix_hedged_leverage")
RISK_MULTIPLIER = 1.05
GDE_FRACTION = 0.40
HEDGE_STRATEGIES = {
    "hedge0": {
        "normal": NORMAL_DIRECTORY,
        "proxy": PROXY_DIRECTORY,
    },
    "hedge2": {
        "normal": "experiment_r11_r9_vixhedge2",
        "proxy": "experiment_r11_r9_vixhedge2_20y_proxy",
    },
    "hedge4": {
        "normal": "experiment_r11_r9_vixhedge4",
        "proxy": "experiment_r11_r9_vixhedge4_20y_proxy",
    },
}


def metric_delta(
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict[str, float]:
    base = performance_metrics(baseline)
    trial = performance_metrics(candidate)
    return {
        **{f"baseline_{key}": value for key, value in base.items()},
        **{f"candidate_{key}": value for key, value in trial.items()},
        "cagr_delta": trial["cagr"] - base["cagr"],
        "sharpe_delta": trial["sharpe"] - base["sharpe"],
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
    base_normal_weights, base_normal_daily = load_strategy_inputs(
        NORMAL_DIRECTORY
    )
    base_proxy_weights, base_proxy_daily = load_strategy_inputs(
        PROXY_DIRECTORY
    )
    samples = {
        "normal_synthetic": {
            "baseline_directory": NORMAL_DIRECTORY,
            "base_weights": base_normal_weights,
            "base_daily": base_normal_daily,
            "strategy_key": "normal",
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
            "baseline_directory": PROXY_DIRECTORY,
            "base_weights": base_proxy_weights,
            "base_daily": base_proxy_daily,
            "strategy_key": "proxy",
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
            "baseline_directory": NORMAL_DIRECTORY,
            "base_weights": base_normal_weights,
            "base_daily": base_normal_daily,
            "strategy_key": "normal",
            "opens": live_opens,
            "closes": live_closes,
            "mode": "live",
            "start": "2022-03-17",
            "periods": {
                "live_2022": ("2022-03-17", "2022-12-31"),
                "live_2023_2024": ("2023-01-01", "2024-12-31"),
                "live_2025_2026": ("2025-01-01", "2026-12-31"),
                "live_2022_2026": ("2022-03-17", "2026-12-31"),
            },
        },
    }
    rows: list[dict[str, float | str]] = []
    current_paths: dict[str, dict[str, pd.Series]] = {}
    current_baselines: dict[str, pd.Series] = {}

    for sample, settings in samples.items():
        opens = settings["opens"]
        closes = settings["closes"]
        periods = settings["periods"]
        assert isinstance(opens, pd.DataFrame)
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)
        sample_paths: dict[str, pd.Series] = {}
        for scenario in COST_SCENARIOS:
            baseline = simulate_gde_substitution(
                str(settings["baseline_directory"]),
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
            if scenario.name == "current_liquidity":
                current_baselines[sample] = baseline["net_return"]
            for candidate_name, directories in HEDGE_STRATEGIES.items():
                strategy_directory = directories[
                    str(settings["strategy_key"])
                ]
                weights, daily = load_strategy_inputs(strategy_directory)
                levered_weights = scale_non_cash_weights(
                    weights,
                    RISK_MULTIPLIER,
                    unscaled_assets=("VIX_HEDGE",),
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
                trial = simulate_gde_substitution(
                    strategy_directory,
                    opens,
                    closes,
                    substitution_fraction=GDE_FRACTION,
                    gate_mode="growth_linked",
                    gde_return_mode=str(settings["mode"]),
                    start_date=str(settings["start"]),
                    end_date=None,
                    base_one_way_cost_bps=(
                        scenario.base_one_way_cost_bps
                    ),
                    gde_one_way_cost_bps=(
                        scenario.gde_one_way_cost_bps
                    ),
                    financing_spread_bps=(
                        scenario.financing_spread_bps
                    ),
                    weights_override=guard_weights,
                    daily_override=guard_daily,
                    extra_slippage=guard_daily["slippage_cost"],
                    gde_no_trade_band=GDE_NO_TRADE_BAND,
                )
                common = baseline.index.intersection(trial.index)
                for period, (period_start, period_end) in periods.items():
                    selected = common[
                        (common >= period_start)
                        & (common <= period_end)
                    ]
                    rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "candidate": candidate_name,
                            "period": period,
                            "vix_hedge_active_days": int(
                                guard_weights.loc[
                                    selected,
                                    "VIX_HEDGE",
                                ]
                                .gt(0.0)
                                .sum()
                            ),
                            **metric_delta(
                                baseline.loc[selected, "net_return"],
                                trial.loc[selected, "net_return"],
                            ),
                        }
                    )
                if scenario.name == "current_liquidity":
                    sample_paths[candidate_name] = trial["net_return"]
                    trial.to_csv(
                        OUTPUT
                        / f"{sample}_{candidate_name}_daily.csv",
                        index_label="date",
                    )
        current_paths[sample] = sample_paths

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    family_rows: list[dict[str, float | int | str]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        baseline = current_baselines[sample]
        names = list(current_paths[sample])
        matrix = np.column_stack(
            [
                relative_log_return(
                    baseline,
                    current_paths[sample][name],
                )
                for name in names
            ]
        )
        for selected_index, name in enumerate(names):
            for block_days in (21, 63, 126):
                family_rows.append(
                    {
                        "sample": sample,
                        "candidate": name,
                        "declared_family_size": len(names),
                        **circular_family_reality_check(
                            matrix,
                            selected_index,
                            block_days,
                        ),
                    }
                )
    family = pd.DataFrame(family_rows)
    family.to_csv(OUTPUT / "declared_family_reality_check.csv", index=False)

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
                "vix_hedge_active_days",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nDeclared-family Reality Check:")
    print(
        family[
            [
                "sample",
                "candidate",
                "block_days",
                "selected_annualized_relative_log_return",
                "familywise_reality_check_p_value",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
