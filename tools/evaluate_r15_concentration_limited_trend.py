from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
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
    metric_delta,
    relative_log_return,
    scale_non_cash_weights_by_series,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r11_post_shock_cooldown import sample_definitions
from tools.evaluate_r12_volatility_managed_risk import (
    BASE_MULTIPLIER,
    GDE_FRACTION,
    robustness_rows,
    simulate_fixed_r11,
)
from tools.evaluate_r13_drawdown_budget import cap_weights_at_cash_floor
from tools.evaluate_r14_incremental_trend_permission import (
    TrendCandidate,
    causal_trend_permission,
)


OUTPUT = Path("output/r15_concentration_limited_trend")
ACTIVE_MULTIPLIER = 1.20
CASH_FLOOR = -0.30
CUMULATIVE_TRIALS = 59
TRADING_EPSILON = 1e-14


@dataclass(frozen=True)
class CombinedCandidate:
    name: str
    trend_mode: str
    semis_cap: float


CANDIDATES = (
    CombinedCandidate("qqq_sma200_cap60", "qqq_sma200", 0.60),
    CombinedCandidate("both_sma200_cap60", "both_sma200", 0.60),
    CombinedCandidate("both_sma200_cap55", "both_sma200", 0.55),
)


def cap_semis_to_qqq(
    weights: pd.DataFrame,
    semis_cap: float,
) -> tuple[pd.DataFrame, pd.Series]:
    if not 0.0 <= semis_cap <= 1.0:
        raise ValueError("semis_cap must be in [0, 1]")
    adjusted = weights.copy()
    excess = (adjusted["SEMIS"] - semis_cap).clip(lower=0.0)
    adjusted["SEMIS"] -= excess
    adjusted["QQQ"] += excess
    return adjusted, excess


def combined_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    definition: CombinedCandidate,
    *,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if active_multiplier < 1.0:
        raise ValueError("active_multiplier cannot reduce R11")
    index = (
        weights.index.intersection(base_daily.index)
        .intersection(closes.index)
    )
    trend = TrendCandidate(definition.name, definition.trend_mode)
    permission = causal_trend_permission(
        closes.reindex(index),
        trend,
    )
    relative = pd.Series(
        np.where(permission, active_multiplier, 1.0),
        index=index,
        dtype=float,
    )
    absolute = BASE_MULTIPLIER * relative
    requested = scale_non_cash_weights_by_series(
        weights.loc[index],
        absolute,
    )
    concentrated, excess = cap_semis_to_qqq(
        requested,
        definition.semis_cap,
    )
    implemented, cash_scale = cap_weights_at_cash_floor(
        concentrated,
        cash_floor,
    )
    base_trade = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    state_change = permission.ne(permission.shift(1))
    binding = excess.gt(0.0)
    binding_start = binding & ~binding.shift(1, fill_value=False)
    update = base_trade | state_change | binding_start
    execution_daily = base_daily.loc[index].copy()
    execution_daily.loc[update, "turnover"] = np.maximum(
        execution_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = pd.DataFrame(
        {
            "trend_permission": permission,
            "relative_multiplier": relative,
            "absolute_multiplier": absolute,
            "semis_excess_to_qqq": excess,
            "semis_cap_binding": binding,
            "cash_cap_scale": cash_scale,
            "implemented_semis_weight": implemented["SEMIS"],
            "implemented_qqq_weight": implemented["QQQ"],
            "implemented_cash_weight": implemented["CASH"],
            "base_trade": base_trade,
            "state_change": state_change,
            "binding_start": binding_start,
            "execution_update": update,
        },
        index=index,
    )
    return implemented, execution_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    definition: CombinedCandidate,
    *,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, diagnostics = combined_schedule(
        weights,
        daily,
        closes,
        definition,
        active_multiplier=active_multiplier,
        cash_floor=cash_floor,
    )
    guard_daily, guard_weights = simulate_guard(
        scheduled,
        execution_daily,
        opens,
        closes,
        guard_for_multiplier(
            BASE_MULTIPLIER,
            scenario.emergency_slippage_bps,  # type: ignore[attr-defined]
        ),
        cost_bps=scenario.base_one_way_cost_bps,  # type: ignore[attr-defined]
        start_date=str(settings["start"]),
    )
    trial = simulate_gde_substitution(
        str(settings["directory"]),
        opens,
        closes,
        substitution_fraction=GDE_FRACTION,
        gate_mode="growth_linked",
        gde_return_mode=str(settings["mode"]),
        start_date=str(settings["start"]),
        end_date=None,
        base_one_way_cost_bps=(
            scenario.base_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        gde_one_way_cost_bps=(
            scenario.gde_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        financing_spread_bps=(
            scenario.financing_spread_bps  # type: ignore[attr-defined]
        ),
        weights_override=guard_weights,
        daily_override=guard_daily,
        extra_slippage=guard_daily["slippage_cost"],
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    diagnostics = diagnostics.reindex(trial.index)
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    diagnostics["guard_triggered"] = guard_daily.reindex(
        trial.index
    )["triggered"]
    return trial, diagnostics


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(
        PROXY_OPEN_CLOSE
    )
    live_opens, live_closes = join_live_gde(
        normal_opens,
        normal_closes,
    )
    normal_weights, normal_daily = load_strategy_inputs(
        NORMAL_DIRECTORY
    )
    proxy_weights, proxy_daily = load_strategy_inputs(
        PROXY_DIRECTORY
    )
    samples = sample_definitions(
        normal_weights,
        normal_daily,
        proxy_weights,
        proxy_daily,
        normal_opens,
        normal_closes,
        proxy_opens,
        proxy_closes,
        live_opens,
        live_closes,
    )

    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    year_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    paths: dict[str, dict[str, pd.Series]] = {}
    baselines: dict[str, pd.Series] = {}

    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        sample_paths: dict[str, pd.Series] = {}
        for scenario in COST_SCENARIOS:
            baseline, _, _ = simulate_fixed_r11(settings, scenario)
            if scenario.name == "current_liquidity":
                baselines[sample] = baseline["net_return"]
                baseline.to_csv(
                    OUTPUT / f"{sample}_r11_baseline_daily.csv",
                    index_label="date",
                )
            for definition in CANDIDATES:
                trial, diagnostics = simulate_candidate(
                    settings,
                    scenario,
                    definition,
                )
                common = baseline.index.intersection(trial.index)
                for period, (period_start, period_end) in periods.items():
                    selected = common[
                        (common >= period_start)
                        & (common <= period_end)
                    ]
                    metric_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "candidate": definition.name,
                            "period": period,
                            **metric_delta(
                                baseline.loc[selected, "net_return"],
                                trial.loc[selected, "net_return"],
                            ),
                        }
                    )
                diagnostic_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "candidate": definition.name,
                        "permission_share": float(
                            diagnostics["trend_permission"].mean()
                        ),
                        "semis_cap_days": int(
                            diagnostics["semis_cap_binding"].sum()
                        ),
                        "cash_cap_days": int(
                            diagnostics["cash_cap_scale"].lt(1.0).sum()
                        ),
                        "minimum_cash": float(
                            diagnostics["implemented_cash_weight"].min()
                        ),
                        "maximum_semis": float(
                            diagnostics["implemented_semis_weight"].max()
                        ),
                        "guard_triggers": int(
                            diagnostics["guard_triggered"]
                            .fillna(False)
                            .sum()
                        ),
                    }
                )
                if scenario.name == "current_liquidity":
                    sample_paths[definition.name] = trial[
                        "net_return"
                    ]
                    trial.to_csv(
                        OUTPUT
                        / f"{sample}_{definition.name}_daily.csv",
                        index_label="date",
                    )
                    diagnostics.to_csv(
                        OUTPUT
                        / f"{sample}_{definition.name}_diagnostics.csv",
                        index_label="date",
                    )
                    years, events = robustness_rows(
                        sample,
                        definition.name,
                        baseline["net_return"],
                        trial["net_return"],
                    )
                    year_rows.extend(years)
                    event_rows.extend(events)
        paths[sample] = sample_paths

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        OUTPUT / "constraint_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(year_rows).to_csv(
        OUTPUT / "leave_one_year_out.csv",
        index=False,
    )
    pd.DataFrame(event_rows).to_csv(
        OUTPUT / "leave_one_drawdown_event_out.csv",
        index=False,
    )

    family_rows: list[dict[str, object]] = []
    names = [definition.name for definition in CANDIDATES]
    for sample in ("normal_synthetic", "proxy_synthetic"):
        matrix = np.column_stack(
            [
                relative_log_return(
                    baselines[sample],
                    paths[sample][name],
                )
                for name in names
            ]
        )
        for selected_index, name in enumerate(names):
            for block_days in (21, 63, 126):
                check = circular_family_reality_check(
                    matrix,
                    selected_index,
                    block_days,
                )
                family_p = float(
                    check["familywise_reality_check_p_value"]
                )
                family_rows.append(
                    {
                        "sample": sample,
                        "candidate": name,
                        "declared_family_size": len(names),
                        "cumulative_trials": CUMULATIVE_TRIALS,
                        **check,
                        "cumulative_trial_adjusted_p_value": min(
                            1.0,
                            family_p * CUMULATIVE_TRIALS / len(names),
                        ),
                    }
                )
    pd.DataFrame(family_rows).to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    neighborhood_rows: list[dict[str, object]] = []
    neighborhoods = (
        ("active118", 1.18, CASH_FLOOR),
        ("active122", 1.22, CASH_FLOOR),
        ("cash25", ACTIVE_MULTIPLIER, -0.25),
        ("cash35", ACTIVE_MULTIPLIER, -0.35),
    )
    settings = samples["normal_synthetic"]
    scenario = COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    for definition in CANDIDATES:
        for label, active, cash_floor in neighborhoods:
            trial, _ = simulate_candidate(
                settings,
                scenario,
                definition,
                active_multiplier=active,
                cash_floor=cash_floor,
            )
            delta = metric_delta(
                baseline["net_return"],
                trial["net_return"],
            )
            neighborhood_rows.append(
                {
                    "candidate": definition.name,
                    "neighborhood": label,
                    "active_multiplier": active,
                    "cash_floor": cash_floor,
                    **delta,
                    "ridge_pass": bool(
                        delta["cagr_delta"] > 0.0
                        and delta["max_drawdown_delta"] >= 0.0
                    ),
                }
            )
    pd.DataFrame(neighborhood_rows).to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )

    print(
        metrics.loc[
            metrics["period"].isin(
                {
                    "complete_2015_2026",
                    "complete_2006_2026",
                    "live_2022_2026",
                }
            ),
            [
                "sample",
                "scenario",
                "candidate",
                "candidate_cagr",
                "candidate_max_drawdown",
                "cagr_delta",
                "max_drawdown_delta",
            ],
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
