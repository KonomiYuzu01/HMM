from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from regime_strategy.portfolio import causal_realized_volatility_state
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
from tools.evaluate_r14_incremental_trend_permission import (
    trend_permission_schedule,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r21_current_engine_industry_momentum import (
    ACTIVE_MULTIPLIER,
    CASH_FLOOR,
    DEFINITION,
    MAX_TILT,
    REBALANCE_DAYS,
    causal_industry_momentum,
)


OUTPUT = Path("output/r23_high_volatility_tilt_gate")
CUMULATIVE_TRIALS = 92
VOLATILITY_WINDOW = 20
THRESHOLD_LOOKBACK = 756
MINIMUM_THRESHOLD_OBSERVATIONS = 504
LOW_QUANTILE = 0.45
HIGH_QUANTILE = 0.90
ALLOWED_STATES = {"insufficient", "low", "medium"}


def causal_gated_industry_momentum(
    closes: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    lookback_days: int = r21.MOMENTUM_DAYS,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
    high_quantile: float = HIGH_QUANTILE,
) -> pd.DataFrame:
    if not LOW_QUANTILE < high_quantile < 1.0:
        raise ValueError("high_quantile must exceed the fixed low quantile")
    diagnostics = causal_industry_momentum(
        closes,
        index,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        max_tilt=max_tilt,
    ).copy()
    pair_returns = closes[["QQQ", "SEMIS"]].pct_change(
        fill_method=None
    ).mean(axis=1)
    states = pd.Series("insufficient", index=index, dtype=object)
    realized = pd.Series(np.nan, index=index, dtype=float)
    update = diagnostics["momentum_update"].astype(bool)
    for date in index[update.to_numpy()]:
        state, current_volatility = causal_realized_volatility_state(
            pair_returns.loc[pair_returns.index < date],
            window_days=VOLATILITY_WINDOW,
            threshold_lookback_days=THRESHOLD_LOOKBACK,
            minimum_threshold_observations=(
                MINIMUM_THRESHOLD_OBSERVATIONS
            ),
            low_quantile=LOW_QUANTILE,
            high_quantile=high_quantile,
        )
        states.loc[date] = state
        realized.loc[date] = current_volatility
    states = states.where(update).ffill().fillna("insufficient")
    realized = realized.where(update).ffill()
    gate_active = ~states.isin(ALLOWED_STATES)
    update_share = diagnostics["raw_semis_growth_share"].where(update)
    update_share = update_share.where(~gate_active, 0.50)
    gated_share = update_share.ffill().fillna(0.50)
    diagnostics["volatility_state"] = states
    diagnostics["pair_realized_volatility"] = realized
    diagnostics["volatility_gate_active"] = gate_active
    diagnostics["ungated_semis_growth_share"] = diagnostics[
        "semis_growth_share"
    ]
    diagnostics["semis_growth_share"] = gated_share
    diagnostics["qqq_growth_share"] = 1.0 - gated_share
    diagnostics["gate_share_change"] = (
        diagnostics["semis_growth_share"]
        - diagnostics["ungated_semis_growth_share"]
    )
    return diagnostics


def apply_gated_industry_momentum(
    weights: pd.DataFrame,
    execution_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
    high_quantile: float = HIGH_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = (
        weights.index.intersection(execution_daily.index)
        .intersection(closes.index)
    )
    diagnostics = causal_gated_industry_momentum(
        closes,
        index,
        rebalance_days=rebalance_days,
        max_tilt=max_tilt,
        high_quantile=high_quantile,
    )
    adjusted = weights.loc[index].copy()
    growth_total = adjusted["QQQ"] + adjusted["SEMIS"]
    adjusted["SEMIS"] = (
        growth_total * diagnostics["semis_growth_share"]
    )
    adjusted["QQQ"] = growth_total - adjusted["SEMIS"]
    budget_error = (
        adjusted["QQQ"] + adjusted["SEMIS"] - growth_total
    ).abs()
    if float(budget_error.max()) > 1e-12:
        raise AssertionError("Volatility gate changed the growth budget")
    if adjusted[["QQQ", "SEMIS"]].lt(-1e-12).any().any():
        raise AssertionError("Volatility gate created short growth weight")
    updated_daily = execution_daily.loc[index].copy()
    update = diagnostics["momentum_update"].astype(bool)
    updated_daily.loc[update, "turnover"] = np.maximum(
        updated_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics["growth_budget_before"] = growth_total
    diagnostics["growth_budget_after"] = (
        adjusted["QQQ"] + adjusted["SEMIS"]
    )
    diagnostics["growth_budget_error"] = budget_error
    return adjusted, updated_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
    high_quantile: float = HIGH_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            cash_floor=cash_floor,
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
    diagnostics = trend_diagnostics.reindex(trial.index).join(
        momentum_diagnostics.reindex(trial.index),
        how="left",
        rsuffix="_momentum",
    )
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
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
                    >= delta["baseline_max_drawdown"]
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
    ]
    neighborhood_pass = bool(
        set(selected["dimension"]) == required_dimensions
        and selected.groupby("dimension")["economic_pass"].any().all()
    )
    budget_pass = bool(
        diagnostics["growth_budget_error"].fillna(0.0).le(1e-12).all()
    )
    active = diagnostics["volatility_gate_active"].astype(bool)
    high_volatility_neutral_pass = bool(
        np.allclose(
            diagnostics.loc[active, "semis_growth_share"],
            0.50,
            atol=1e-12,
        )
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
    statistical_economic_pass = bool(
        complete["candidate_cagr"] >= 0.25
        and complete["candidate_max_drawdown"]
        >= complete["baseline_max_drawdown"]
        and development["annualized_relative_log_return"] > 0.0
        and development["candidate_max_drawdown"]
        >= development["baseline_max_drawdown"]
        and holdout["annualized_relative_log_return"] > 0.0
        and holdout["candidate_max_drawdown"]
        >= holdout["baseline_max_drawdown"]
        and stress["candidate_cagr"] >= 0.25
        and stress["candidate_max_drawdown"]
        >= stress["baseline_max_drawdown"]
        and proxy_complete["annualized_relative_log_return"] > 0.0
        and proxy_complete["candidate_max_drawdown"]
        >= proxy_complete["baseline_max_drawdown"]
        and proxy_early["annualized_relative_log_return"] > 0.0
        and proxy_late["annualized_relative_log_return"] > 0.0
        and bool(existing["family_multiple_testing_pass"])
        and bool(existing["leave_one_out_pass"])
        and neighborhood_pass
        and bool(existing["year_breadth_pass"])
        and budget_pass
        and high_volatility_neutral_pass
    )
    acceptance["candidate"] = (
        "industry_momentum_126_high_volatility_neutral"
    )
    acceptance["parameter_neighborhood_pass"] = neighborhood_pass
    acceptance["growth_budget_conservation_pass"] = budget_pass
    acceptance["high_volatility_neutral_pass"] = (
        high_volatility_neutral_pass
    )
    acceptance["statistical_economic_pass"] = statistical_economic_pass
    acceptance["production_pass"] = bool(
        statistical_economic_pass
        and bool(existing["declared_cash_hard_limit_pass"])
    )
    acceptance.to_csv(acceptance_path, index=False)
    print(acceptance.round(6).to_string(index=False))


def main() -> None:
    r21.OUTPUT = OUTPUT
    r21.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r21.simulate_candidate = simulate_candidate
    r21.main()
    _append_gate_neighborhoods()
    _rewrite_acceptance()
    print(f"R23 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
