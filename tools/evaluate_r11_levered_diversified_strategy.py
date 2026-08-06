from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import (
    OpenGapGuard,
    simulate as simulate_guard,
)
from regime_strategy.report import performance_metrics
from tools.evaluate_r10_combined_tail_capital import selected_guard
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


OUTPUT = Path("output/r11_levered_diversified_strategy")
RISK_MULTIPLIERS = (1.025, 1.050, 1.075)
GDE_FRACTIONS = (0.40, 0.50)
GDE_NO_TRADE_BAND = 0.02


@dataclass(frozen=True)
class CostScenario:
    name: str
    base_one_way_cost_bps: float
    emergency_slippage_bps: float
    gde_one_way_cost_bps: float
    financing_spread_bps: float


COST_SCENARIOS = (
    CostScenario("current_liquidity", 7.5, 20.0, 40.0, 100.0),
    CostScenario("cost_stress", 15.0, 50.0, 75.0, 150.0),
)


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


def guard_for_multiplier(
    multiplier: float,
    emergency_slippage_bps: float,
) -> OpenGapGuard:
    return selected_guard(
        emergency_slippage_bps,
        post_trigger_cap=0.15,
        relative_gap_trigger=-0.005,
        minimum_semis_weight=0.60 * multiplier,
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
                "live_2022": ("2022-03-17", "2022-12-31"),
                "live_2023_2024": ("2023-01-01", "2024-12-31"),
                "live_2025_2026": ("2025-01-01", "2026-12-31"),
                "live_2022_2026": ("2022-03-17", "2026-12-31"),
            },
        },
    }
    rows: list[dict[str, float | str]] = []
    family_rows: list[dict[str, float | int | str]] = []
    current_paths: dict[str, dict[str, pd.Series]] = {}
    current_baselines: dict[str, pd.Series] = {}

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
        sample_paths: dict[str, pd.Series] = {}
        for scenario in COST_SCENARIOS:
            baseline = simulate_gde_substitution(
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
            if scenario.name == "current_liquidity":
                current_baselines[sample] = baseline["net_return"]
            for multiplier in RISK_MULTIPLIERS:
                levered_weights = scale_non_cash_weights(
                    weights,
                    multiplier,
                )
                guard_daily, guard_weights = simulate_guard(
                    levered_weights,
                    daily,
                    opens,
                    closes,
                    guard_for_multiplier(
                        multiplier,
                        scenario.emergency_slippage_bps,
                    ),
                    cost_bps=scenario.base_one_way_cost_bps,
                    start_date=str(settings["start"]),
                )
                for gde_fraction in GDE_FRACTIONS:
                    candidate_name = (
                        f"risk{multiplier:.3f}"
                        f"_gde{int(round(gde_fraction * 100))}"
                    )
                    candidate = simulate_gde_substitution(
                        str(settings["directory"]),
                        opens,
                        closes,
                        substitution_fraction=gde_fraction,
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
                    common = baseline.index.intersection(candidate.index)
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
                                "risk_multiplier": multiplier,
                                "gde_fraction": gde_fraction,
                                "period": period,
                                "guard_triggers": int(
                                    guard_daily.loc[
                                        selected,
                                        "triggered",
                                    ].sum()
                                ),
                                **metric_delta(
                                    baseline.loc[selected, "net_return"],
                                    candidate.loc[selected, "net_return"],
                                ),
                            }
                        )
                    if scenario.name == "current_liquidity":
                        sample_paths[candidate_name] = candidate[
                            "net_return"
                        ]
                        candidate.to_csv(
                            OUTPUT
                            / f"{sample}_{candidate_name}_daily.csv",
                            index_label="date",
                        )
        current_paths[sample] = sample_paths

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

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
