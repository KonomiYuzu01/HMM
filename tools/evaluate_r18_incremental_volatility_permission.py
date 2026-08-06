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


OUTPUT = Path("output/r18_incremental_volatility_permission")
CASH_FLOOR = -0.30
CUMULATIVE_TRIALS = 66
TRADING_EPSILON = 1e-14


@dataclass(frozen=True)
class VolatilityCandidate:
    name: str
    active_multiplier: float


CANDIDATES = (
    VolatilityCandidate("active118_vol21le63_both", 1.18),
    VolatilityCandidate("active120_vol21le63_both", 1.20),
)


def causal_volatility_permission(
    closes: pd.DataFrame,
) -> pd.Series:
    missing = [
        asset for asset in ("QQQ", "SEMIS") if asset not in closes
    ]
    if missing:
        raise ValueError(f"Missing volatility assets: {missing}")
    log_returns = np.log(closes[["QQQ", "SEMIS"]]).diff().shift(1)
    vol21 = log_returns.rolling(21, min_periods=21).std(ddof=1)
    vol63 = log_returns.rolling(63, min_periods=63).std(ddof=1)
    sufficient = vol21.notna().all(axis=1) & vol63.notna().all(axis=1)
    permission = vol21.le(vol63).all(axis=1) & sufficient
    return permission.fillna(False).rename("volatility_permission")


def volatility_trend_schedule(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    active_multiplier: float,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if active_multiplier < 1.0:
        raise ValueError("active_multiplier cannot reduce R11")
    index = (
        weights.index.intersection(base_daily.index)
        .intersection(closes.index)
    )
    aligned_closes = closes.reindex(index)
    trend = causal_trend_permission(
        aligned_closes,
        TrendCandidate("both_sma200", "both_sma200"),
    )
    volatility = causal_volatility_permission(aligned_closes)
    desired = trend & volatility
    base_trade = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    trend_exit = trend.shift(1).fillna(False) & ~trend
    update = base_trade | trend_exit
    accepted = desired.where(update).ffill().fillna(False).astype(bool)
    relative = pd.Series(
        np.where(accepted, active_multiplier, 1.0),
        index=index,
        dtype=float,
    )
    absolute = BASE_MULTIPLIER * relative
    requested = scale_non_cash_weights_by_series(
        weights.loc[index],
        absolute,
    )
    capped, cap_scale = cap_weights_at_cash_floor(
        requested,
        cash_floor,
    )
    execution_daily = base_daily.loc[index].copy()
    execution_daily.loc[update, "turnover"] = np.maximum(
        execution_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = pd.DataFrame(
        {
            "trend_permission": trend,
            "raw_volatility_permission": volatility,
            "desired_incremental_permission": desired,
            "accepted_incremental_permission": accepted,
            "relative_multiplier": relative,
            "absolute_multiplier": absolute,
            "cash_cap_scale": cap_scale,
            "implemented_cash_weight": capped["CASH"],
            "base_trade": base_trade,
            "trend_exit": trend_exit,
            "execution_update": update,
        },
        index=index,
    )
    return capped, execution_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    candidate: VolatilityCandidate,
    *,
    active_multiplier: float | None = None,
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
    multiplier = (
        candidate.active_multiplier
        if active_multiplier is None
        else active_multiplier
    )
    scheduled, execution_daily, diagnostics = (
        volatility_trend_schedule(
            weights,
            daily,
            closes,
            active_multiplier=multiplier,
            cash_floor=cash_floor,
        )
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
            for candidate in CANDIDATES:
                trial, diagnostics = simulate_candidate(
                    settings,
                    scenario,
                    candidate,
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
                            "candidate": candidate.name,
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
                        "candidate": candidate.name,
                        "raw_volatility_permission_share": float(
                            diagnostics[
                                "raw_volatility_permission"
                            ].mean()
                        ),
                        "accepted_incremental_permission_share": float(
                            diagnostics[
                                "accepted_incremental_permission"
                            ].mean()
                        ),
                        "execution_updates": int(
                            diagnostics["execution_update"].sum()
                        ),
                        "cash_cap_days": int(
                            diagnostics["cash_cap_scale"].lt(1.0).sum()
                        ),
                        "minimum_target_cash": float(
                            diagnostics["implemented_cash_weight"].min()
                        ),
                        "minimum_post_guard_cash": float(
                            diagnostics["post_guard_cash_weight"].min()
                        ),
                        "guard_triggers": int(
                            diagnostics["guard_triggered"]
                            .fillna(False)
                            .sum()
                        ),
                    }
                )
                if scenario.name == "current_liquidity":
                    sample_paths[candidate.name] = trial["net_return"]
                    trial.to_csv(
                        OUTPUT
                        / f"{sample}_{candidate.name}_daily.csv",
                        index_label="date",
                    )
                    diagnostics.to_csv(
                        OUTPUT
                        / f"{sample}_{candidate.name}_diagnostics.csv",
                        index_label="date",
                    )
                    years, events = robustness_rows(
                        sample,
                        candidate.name,
                        baseline["net_return"],
                        trial["net_return"],
                    )
                    year_rows.extend(years)
                    event_rows.extend(events)
        paths[sample] = sample_paths

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    years = pd.DataFrame(year_rows)
    events = pd.DataFrame(event_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    diagnostics.to_csv(
        OUTPUT / "permission_diagnostics.csv",
        index=False,
    )
    years.to_csv(OUTPUT / "leave_one_year_out.csv", index=False)
    events.to_csv(
        OUTPUT / "leave_one_drawdown_event_out.csv",
        index=False,
    )

    names = [candidate.name for candidate in CANDIDATES]
    family_rows: list[dict[str, object]] = []
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
                            family_p
                            * CUMULATIVE_TRIALS
                            / len(names),
                        ),
                    }
                )
    family = pd.DataFrame(family_rows)
    family.to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    neighborhood_rows: list[dict[str, object]] = []
    settings = samples["normal_synthetic"]
    baseline, _, _ = simulate_fixed_r11(
        settings,
        COST_SCENARIOS[0],
    )
    for candidate in CANDIDATES:
        variants = (
            (
                "active_minus_001",
                candidate.active_multiplier - 0.01,
                CASH_FLOOR,
            ),
            (
                "active_plus_001",
                candidate.active_multiplier + 0.01,
                CASH_FLOOR,
            ),
            (
                "cash_floor_028",
                candidate.active_multiplier,
                -0.28,
            ),
        )
        for label, active, cash_floor in variants:
            trial, _ = simulate_candidate(
                settings,
                COST_SCENARIOS[0],
                candidate,
                active_multiplier=active,
                cash_floor=cash_floor,
            )
            delta = metric_delta(
                baseline["net_return"],
                trial["net_return"],
            )
            neighborhood_rows.append(
                {
                    "candidate": candidate.name,
                    "neighborhood": label,
                    "active_multiplier": active,
                    "cash_floor": cash_floor,
                    **delta,
                    "ridge_pass": bool(
                        delta["candidate_cagr"] >= 0.25
                        and delta["candidate_max_drawdown"]
                        >= delta["baseline_max_drawdown"]
                    ),
                }
            )
    neighborhoods = pd.DataFrame(neighborhood_rows)
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )

    acceptance_rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        name = candidate.name
        central = metrics.loc[
            (metrics["candidate"] == name)
            & (metrics["scenario"] == "current_liquidity")
        ]
        stress = metrics.loc[
            (metrics["candidate"] == name)
            & (metrics["scenario"] == "cost_stress")
        ]

        def select(
            frame: pd.DataFrame,
            sample: str,
            period: str,
        ) -> pd.Series:
            selected = frame.loc[
                (frame["sample"] == sample)
                & (frame["period"] == period)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one {name}/{sample}/{period} row"
                )
            return selected.iloc[0]

        complete = select(
            central,
            "normal_synthetic",
            "complete_2015_2026",
        )
        development = select(
            central,
            "normal_synthetic",
            "development_2015_2021",
        )
        holdout = select(
            central,
            "normal_synthetic",
            "holdout_2022_2025",
        )
        stress_complete = select(
            stress,
            "normal_synthetic",
            "complete_2015_2026",
        )
        proxy_early = select(
            central,
            "proxy_synthetic",
            "early_2006_2014",
        )
        proxy_late = select(
            central,
            "proxy_synthetic",
            "late_2015_2026",
        )
        proxy_complete = select(
            central,
            "proxy_synthetic",
            "complete_2006_2026",
        )
        family_pass = bool(
            family.loc[
                (family["sample"] == "normal_synthetic")
                & (family["candidate"] == name),
                "cumulative_trial_adjusted_p_value",
            ].le(0.05).all()
        )
        leave_out_pass = bool(
            years.loc[
                (years["sample"] == "normal_synthetic")
                & (years["candidate"] == name),
                "annualized_relative_log_return",
            ].gt(0.0).all()
            and events.loc[
                (events["sample"] == "normal_synthetic")
                & (events["candidate"] == name),
                "annualized_relative_log_return",
            ].gt(0.0).all()
        )
        neighborhood_passes = int(
            neighborhoods.loc[
                neighborhoods["candidate"] == name,
                "ridge_pass",
            ].sum()
        )
        diagnostic = diagnostics.loc[
            (diagnostics["sample"] == "normal_synthetic")
            & (diagnostics["scenario"] == "current_liquidity")
            & (diagnostics["candidate"] == name)
        ].iloc[0]
        economic_statistical_pass = bool(
            complete["candidate_cagr"] >= 0.25
            and complete["candidate_max_drawdown"]
            >= complete["baseline_max_drawdown"]
            and development["cagr_delta"] > 0.0
            and holdout["cagr_delta"] > 0.0
            and stress_complete["cagr_delta"] > 0.0
            and stress_complete["candidate_max_drawdown"]
            >= stress_complete["baseline_max_drawdown"]
            and proxy_early["cagr_delta"] > 0.0
            and proxy_late["cagr_delta"] > 0.0
            and proxy_complete["candidate_max_drawdown"]
            >= proxy_complete["baseline_max_drawdown"]
            and family_pass
            and leave_out_pass
            and neighborhood_passes >= 2
            and 0.20
            <= diagnostic["raw_volatility_permission_share"]
            <= 0.80
            and diagnostic["minimum_target_cash"] >= -0.30 - 1e-12
            and diagnostic["minimum_post_guard_cash"] >= -0.32
        )
        financing_hard_limit_enforced = False
        acceptance_rows.append(
            {
                "candidate": name,
                "complete_cagr": complete["candidate_cagr"],
                "complete_max_drawdown": (
                    complete["candidate_max_drawdown"]
                ),
                "r11_max_drawdown": (
                    complete["baseline_max_drawdown"]
                ),
                "development_cagr_delta": development["cagr_delta"],
                "holdout_cagr_delta": holdout["cagr_delta"],
                "stress_cagr": stress_complete["candidate_cagr"],
                "stress_max_drawdown": (
                    stress_complete["candidate_max_drawdown"]
                ),
                "proxy_complete_cagr": (
                    proxy_complete["candidate_cagr"]
                ),
                "proxy_complete_max_drawdown": (
                    proxy_complete["candidate_max_drawdown"]
                ),
                "proxy_r11_max_drawdown": (
                    proxy_complete["baseline_max_drawdown"]
                ),
                "volatility_permission_share": (
                    diagnostic["raw_volatility_permission_share"]
                ),
                "minimum_target_cash": (
                    diagnostic["minimum_target_cash"]
                ),
                "minimum_post_guard_cash": (
                    diagnostic["minimum_post_guard_cash"]
                ),
                "family_multiple_testing_pass": family_pass,
                "leave_one_out_pass": leave_out_pass,
                "neighborhood_pass_count": neighborhood_passes,
                "financing_hard_limit_enforced": (
                    financing_hard_limit_enforced
                ),
                "statistical_economic_pass": (
                    economic_statistical_pass
                ),
                "production_pass": bool(
                    economic_statistical_pass
                    and financing_hard_limit_enforced
                ),
            }
        )
    acceptance = pd.DataFrame(acceptance_rows)
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    print(
        acceptance[
            [
                "candidate",
                "complete_cagr",
                "complete_max_drawdown",
                "r11_max_drawdown",
                "proxy_complete_max_drawdown",
                "family_multiple_testing_pass",
                "statistical_economic_pass",
                "production_pass",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
