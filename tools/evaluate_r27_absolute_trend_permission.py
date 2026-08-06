from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r21_current_engine_industry_momentum import (
    MAX_TILT,
    REBALANCE_DAYS,
)
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r24_declared_cash_hard_limit as r24
from tools.evaluate_r24_declared_cash_hard_limit import (
    MAX_DRAWDOWN_TOLERANCE,
)
from tools.evaluate_r23_high_volatility_tilt_gate import HIGH_QUANTILE
from tools.evaluate_r26_pulse_relative_tilt_risk_veto import (
    MAXIMUM_HOLD_SESSIONS,
    causal_pulse_gated_industry_momentum,
)


OUTPUT = Path("output/r27_absolute_trend_permission")
CUMULATIVE_TRIALS = 98
ABSOLUTE_TREND_DAYS = 63


def causal_absolute_permitted_industry_momentum(
    closes: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    lookback_days: int = r21.MOMENTUM_DAYS,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
    high_quantile: float = HIGH_QUANTILE,
    absolute_trend_days: int = ABSOLUTE_TREND_DAYS,
) -> pd.DataFrame:
    if absolute_trend_days < 2:
        raise ValueError("absolute_trend_days must be at least two")
    missing = [
        asset for asset in ("QQQ", "SEMIS", "CASH") if asset not in closes
    ]
    if missing:
        raise ValueError(f"Missing absolute-trend assets: {missing}")
    diagnostics = causal_pulse_gated_industry_momentum(
        closes,
        index,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        max_tilt=max_tilt,
        high_quantile=high_quantile,
    ).copy()
    log_returns = np.log(
        closes[["QQQ", "SEMIS", "CASH"]]
        .astype(float)
        .div(closes[["QQQ", "SEMIS", "CASH"]].shift(1))
    )
    pair_excess = (
        0.50 * (log_returns["QQQ"] + log_returns["SEMIS"])
        - log_returns["CASH"]
    )
    trailing_excess = (
        pair_excess.rolling(
            absolute_trend_days,
            min_periods=absolute_trend_days,
        )
        .sum()
        .shift(1)
        .reindex(index)
    )
    update = diagnostics["momentum_update"].astype(bool)
    raw_permission = trailing_excess.gt(0.0)
    permission = (
        raw_permission.where(update).ffill().fillna(False).astype(bool)
    )
    raw_share = diagnostics["ungated_semis_growth_share"]
    permitted_share = raw_share.where(permission, 0.50)
    final_share = permitted_share.where(
        ~diagnostics["pulse_veto_active"].astype(bool),
        0.50,
    )
    diagnostics["raw_momentum_semis_growth_share"] = raw_share
    diagnostics["prior_pair_excess_log_return"] = trailing_excess
    diagnostics["raw_absolute_trend_permission"] = raw_permission
    diagnostics["absolute_trend_permission"] = permission
    diagnostics["absolute_permitted_semis_growth_share"] = permitted_share
    diagnostics["ungated_semis_growth_share"] = permitted_share
    diagnostics["semis_growth_share"] = final_share
    diagnostics["qqq_growth_share"] = 1.0 - final_share
    diagnostics["gate_share_change"] = final_share - raw_share
    return diagnostics


def apply_absolute_permitted_industry_momentum(
    weights: pd.DataFrame,
    execution_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
    high_quantile: float = HIGH_QUANTILE,
    absolute_trend_days: int = ABSOLUTE_TREND_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = (
        weights.index.intersection(execution_daily.index)
        .intersection(closes.index)
    )
    diagnostics = causal_absolute_permitted_industry_momentum(
        closes,
        index,
        rebalance_days=rebalance_days,
        max_tilt=max_tilt,
        high_quantile=high_quantile,
        absolute_trend_days=absolute_trend_days,
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
        raise AssertionError("Absolute permission changed growth budget")
    if adjusted[["QQQ", "SEMIS"]].lt(-1e-12).any().any():
        raise AssertionError("Absolute permission created short growth weight")
    updated_daily = execution_daily.loc[index].copy()
    update = (
        diagnostics["momentum_update"].astype(bool)
        | diagnostics["pulse_veto_state_change"].astype(bool)
    )
    updated_daily.loc[update, "turnover"] = np.maximum(
        updated_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics["risk_veto_execution_update"] = update
    diagnostics["growth_budget_before"] = growth_total
    diagnostics["growth_budget_after"] = (
        adjusted["QQQ"] + adjusted["SEMIS"]
    )
    diagnostics["growth_budget_error"] = budget_error
    return adjusted, updated_daily, diagnostics


def _append_absolute_trend_neighborhoods() -> None:
    samples = r21._build_samples()
    settings = samples["normal_synthetic"]
    scenario = COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    path = OUTPUT / "parameter_neighborhood.csv"
    neighborhood = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    central_apply = r24.apply_gated_industry_momentum
    for label, horizon in (
        ("absolute_trend42", 42),
        ("absolute_trend84", 84),
    ):
        r24.apply_gated_industry_momentum = partial(
            apply_absolute_permitted_industry_momentum,
            absolute_trend_days=horizon,
        )
        try:
            trial, _ = r24.simulate_candidate(settings, scenario)
        finally:
            r24.apply_gated_industry_momentum = central_apply
        delta = metric_delta(
            baseline["net_return"],
            trial["net_return"],
        )
        rows.append(
            {
                "dimension": "absolute_trend_days",
                "neighborhood": label,
                "active_multiplier": r21.ACTIVE_MULTIPLIER,
                "rebalance_days": REBALANCE_DAYS,
                "max_tilt": MAX_TILT,
                "absolute_trend_days": horizon,
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
    path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(path)
    neighborhood = pd.read_csv(OUTPUT / "parameter_neighborhood.csv")
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    required_dimensions = {
        "active_multiplier",
        "max_tilt",
        "rebalance_days",
        "high_volatility_quantile",
        "absolute_trend_days",
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
    active = diagnostics["pulse_veto_active"].astype(bool)
    groups = active.ne(active.shift()).cumsum()
    pulse_duration_pass = bool(
        diagnostics.groupby(groups)["pulse_veto_active"].sum().max()
        <= MAXIMUM_HOLD_SESSIONS
    )
    existing_pass = acceptance["statistical_economic_pass"].astype(bool)
    acceptance["candidate"] = (
        "r26_with_absolute_trend_permission"
    )
    acceptance["parameter_neighborhood_pass"] = neighborhood_pass
    acceptance["pulse_duration_pass"] = pulse_duration_pass
    acceptance["absolute_trend_signal_causal_pass"] = True
    acceptance["statistical_economic_pass"] = (
        existing_pass & neighborhood_pass & pulse_duration_pass
    )
    acceptance["production_pass"] = acceptance[
        "statistical_economic_pass"
    ]
    acceptance.to_csv(path, index=False)
    print(acceptance.round(6).to_string(index=False))


def main() -> None:
    r24.OUTPUT = OUTPUT
    r24.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r24.apply_gated_industry_momentum = (
        apply_absolute_permitted_industry_momentum
    )
    r24.main()
    _append_absolute_trend_neighborhoods()
    _rewrite_acceptance()
    print(f"R27 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
