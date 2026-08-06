from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from tools.evaluate_r10_gde_capital_efficiency import (
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r12_volatility_managed_risk import (
    BASE_MULTIPLIER,
    GDE_FRACTION,
    simulate_fixed_r11,
)
from tools.evaluate_r13_drawdown_budget import cap_weights_at_cash_floor
from tools.evaluate_r14_incremental_trend_permission import (
    trend_permission_schedule,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r21_current_engine_industry_momentum import (
    ACTIVE_MULTIPLIER,
    DEFINITION,
    MAX_TILT,
    REBALANCE_DAYS,
)
import tools.evaluate_r23_high_volatility_tilt_gate as r23
from tools.evaluate_r23_high_volatility_tilt_gate import (
    HIGH_QUANTILE,
    apply_gated_industry_momentum,
)


OUTPUT = Path("output/r24_declared_cash_hard_limit")
CUMULATIVE_TRIALS = 93
DECLARED_CASH_FLOOR = -0.18
MAX_DRAWDOWN_TOLERANCE = 0.005


def enforce_daily_cash_target_limit(
    weights: pd.DataFrame,
    execution_daily: pd.DataFrame,
    *,
    cash_floor: float = DECLARED_CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "CASH" not in weights:
        raise ValueError("Cash hard limit requires a CASH column")
    index = weights.index.intersection(execution_daily.index)
    original = weights.loc[index].copy()
    limited, scale = cap_weights_at_cash_floor(original, cash_floor)
    active = scale.lt(1.0 - 1e-12)
    updated_daily = execution_daily.loc[index].copy()
    updated_daily.loc[active, "turnover"] = np.maximum(
        updated_daily.loc[active, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = pd.DataFrame(
        {
            "pre_hard_limit_cash_weight": original["CASH"],
            "hard_limit_cash_scale": scale,
            "hard_limit_active": active,
            "post_hard_limit_cash_weight": limited["CASH"],
            "hard_limit_execution_update": active,
        },
        index=index,
    )
    if limited["CASH"].lt(cash_floor - 1e-12).any():
        raise AssertionError("Cash hard limit was not enforced")
    if not np.allclose(
        limited.sum(axis=1),
        original.sum(axis=1),
        atol=1e-12,
    ):
        raise AssertionError("Cash hard limit changed total target weight")
    return limited, updated_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    cash_floor: float = DECLARED_CASH_FLOOR,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
    high_quantile: float = HIGH_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cash_floor != DECLARED_CASH_FLOOR:
        raise ValueError("The production cash hard limit is not tunable")
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, trend_diagnostics = (
        trend_permission_schedule(
            weights,
            daily,
            closes,
            DEFINITION,
            active_multiplier=active_multiplier,
            cash_floor=DECLARED_CASH_FLOOR,
        )
    )
    tilted, tilted_daily, momentum_diagnostics = (
        apply_gated_industry_momentum(
            scheduled,
            execution_daily,
            closes,
            rebalance_days=rebalance_days,
            max_tilt=max_tilt,
            high_quantile=high_quantile,
        )
    )
    guard_daily, guard_weights = simulate_guard(
        tilted,
        tilted_daily,
        opens,
        closes,
        guard_for_multiplier(
            BASE_MULTIPLIER,
            scenario.emergency_slippage_bps,  # type: ignore[attr-defined]
        ),
        cost_bps=scenario.base_one_way_cost_bps,  # type: ignore[attr-defined]
        start_date=str(settings["start"]),
    )
    limited, limited_daily, limit_diagnostics = (
        enforce_daily_cash_target_limit(
            guard_weights,
            guard_daily,
        )
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
        weights_override=limited,
        daily_override=limited_daily,
        extra_slippage=guard_daily["slippage_cost"],
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    diagnostics = trend_diagnostics.reindex(trial.index).join(
        momentum_diagnostics.reindex(trial.index),
        how="left",
        rsuffix="_momentum",
    ).join(
        limit_diagnostics.reindex(trial.index),
        how="left",
    )
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    diagnostics["pre_gde_cash_weight"] = limited.reindex(
        trial.index
    )["CASH"]
    diagnostics["post_guard_semis_weight"] = guard_weights.reindex(
        trial.index
    )["SEMIS"]
    diagnostics["guard_triggered"] = guard_daily.reindex(
        trial.index
    )["triggered"]
    return trial, diagnostics


def _append_gate_neighborhoods() -> None:
    samples = r21._build_samples()
    settings = samples["normal_synthetic"]
    scenario = COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    path = OUTPUT / "parameter_neighborhood.csv"
    neighborhood = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for label, quantile in (
        ("high_quantile85", 0.85),
        ("high_quantile95", 0.95),
    ):
        trial, _ = simulate_candidate(
            settings,
            scenario,
            high_quantile=quantile,
        )
        delta = metric_delta(
            baseline["net_return"],
            trial["net_return"],
        )
        rows.append(
            {
                "dimension": "high_volatility_quantile",
                "neighborhood": label,
                "active_multiplier": ACTIVE_MULTIPLIER,
                "rebalance_days": REBALANCE_DAYS,
                "max_tilt": MAX_TILT,
                "high_quantile": quantile,
                **delta,
                "economic_pass": bool(
                    delta["candidate_cagr"] >= 0.25
                    and delta["candidate_max_drawdown"]
                    >= (
                        delta["baseline_max_drawdown"]
                        - MAX_DRAWDOWN_TOLERANCE
                    )
                ),
            }
        )
    neighborhood = pd.concat(
        [neighborhood, pd.DataFrame(rows)],
        ignore_index=True,
    )
    neighborhood.to_csv(path, index=False)


def _rewrite_acceptance() -> None:
    acceptance_path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(acceptance_path)
    metrics = pd.read_csv(OUTPUT / "metrics_by_period.csv")
    neighborhood = pd.read_csv(OUTPUT / "parameter_neighborhood.csv")
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    required_dimensions = {
        "active_multiplier",
        "max_tilt",
        "rebalance_days",
        "high_volatility_quantile",
    }
    selected = neighborhood.loc[
        neighborhood["dimension"].isin(required_dimensions)
    ].copy()
    selected["tolerance_economic_pass"] = (
        selected["candidate_cagr"].ge(0.25)
        & selected["candidate_max_drawdown"].ge(
            selected["baseline_max_drawdown"]
            - MAX_DRAWDOWN_TOLERANCE
        )
    )
    neighborhood_pass = bool(
        set(selected["dimension"]) == required_dimensions
        and selected.groupby("dimension")[
            "tolerance_economic_pass"
        ].any().all()
    )
    budget_pass = bool(
        diagnostics["growth_budget_error"].fillna(0.0).le(1e-12).all()
    )
    hard_limit_pass = bool(
        diagnostics["implemented_cash_weight"]
        .ge(DECLARED_CASH_FLOOR - 1e-12)
        .all()
        and diagnostics["post_hard_limit_cash_weight"]
        .ge(DECLARED_CASH_FLOOR - 1e-12)
        .all()
        and diagnostics["pre_gde_cash_weight"]
        .ge(DECLARED_CASH_FLOOR - 1e-12)
        .all()
    )

    def row(sample: str, scenario: str, period: str) -> pd.Series:
        chosen = metrics.loc[
            (metrics["sample"] == sample)
            & (metrics["scenario"] == scenario)
            & (metrics["period"] == period)
        ]
        if len(chosen) != 1:
            raise ValueError(f"Expected one {sample}/{scenario}/{period}")
        return chosen.iloc[0]

    complete = row(
        "normal_synthetic", "current_liquidity", "complete_2015_2026"
    )
    development = row(
        "normal_synthetic",
        "current_liquidity",
        "development_2015_2021",
    )
    holdout = row(
        "normal_synthetic",
        "current_liquidity",
        "holdout_2022_2025",
    )
    stress = row(
        "normal_synthetic", "cost_stress", "complete_2015_2026"
    )
    proxy_complete = row(
        "proxy_synthetic",
        "current_liquidity",
        "complete_2006_2026",
    )
    proxy_early = row(
        "proxy_synthetic",
        "current_liquidity",
        "early_2006_2014",
    )
    proxy_late = row(
        "proxy_synthetic",
        "current_liquidity",
        "late_2015_2026",
    )
    existing = acceptance.iloc[0]

    def drawdown_within_tolerance(metric: pd.Series) -> bool:
        return bool(
            metric["candidate_max_drawdown"]
            >= (
                metric["baseline_max_drawdown"]
                - MAX_DRAWDOWN_TOLERANCE
            )
        )

    statistical_economic_pass = bool(
        complete["candidate_cagr"] >= 0.25
        and drawdown_within_tolerance(complete)
        and development["annualized_relative_log_return"] > 0.0
        and drawdown_within_tolerance(development)
        and holdout["annualized_relative_log_return"] > 0.0
        and drawdown_within_tolerance(holdout)
        and stress["annualized_relative_log_return"] > 0.0
        and drawdown_within_tolerance(stress)
        and proxy_complete["annualized_relative_log_return"] > 0.0
        and drawdown_within_tolerance(proxy_complete)
        and proxy_early["annualized_relative_log_return"] > 0.0
        and proxy_late["annualized_relative_log_return"] > 0.0
        and bool(existing["family_multiple_testing_pass"])
        and bool(existing["leave_one_out_pass"])
        and neighborhood_pass
        and bool(existing["year_breadth_pass"])
        and budget_pass
        and hard_limit_pass
    )
    acceptance["candidate"] = (
        "r23_with_declared_cash_hard_limit"
    )
    acceptance["parameter_neighborhood_pass"] = neighborhood_pass
    acceptance["growth_budget_conservation_pass"] = budget_pass
    acceptance["declared_cash_hard_limit_pass"] = hard_limit_pass
    acceptance["max_drawdown_tolerance"] = MAX_DRAWDOWN_TOLERANCE
    acceptance["statistical_economic_pass"] = statistical_economic_pass
    acceptance["production_pass"] = statistical_economic_pass
    acceptance.to_csv(acceptance_path, index=False)
    print(acceptance.round(6).to_string(index=False))


def main() -> None:
    r21.OUTPUT = OUTPUT
    r21.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r21.simulate_candidate = simulate_candidate
    r21.CASH_FLOOR = DECLARED_CASH_FLOOR
    r21.main()
    _append_gate_neighborhoods()
    _rewrite_acceptance()
    print(f"R24 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
