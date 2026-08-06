from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r21_current_engine_industry_momentum import (
    MAX_TILT,
    REBALANCE_DAYS,
)
import tools.evaluate_r24_declared_cash_hard_limit as r24
from tools.evaluate_r23_high_volatility_tilt_gate import HIGH_QUANTILE
from tools.evaluate_r25_daily_relative_tilt_risk_veto import (
    causal_daily_gated_industry_momentum,
)


OUTPUT = Path("output/r26_pulse_relative_tilt_risk_veto")
CUMULATIVE_TRIALS = 95
RECOVERY_CONFIRMATIONS = 2
MAXIMUM_HOLD_SESSIONS = 4


def pulse_cooldown_state(
    high_volatility: pd.Series,
    prior_relative_return: pd.Series,
    *,
    recovery_confirmations: int = RECOVERY_CONFIRMATIONS,
    maximum_hold_sessions: int = MAXIMUM_HOLD_SESSIONS,
) -> pd.DataFrame:
    if recovery_confirmations < 1:
        raise ValueError("recovery_confirmations must be positive")
    if maximum_hold_sessions < 1:
        raise ValueError("maximum_hold_sessions must be positive")
    index = high_volatility.index.intersection(
        prior_relative_return.index
    )
    high = high_volatility.reindex(index).fillna(False).astype(bool)
    relative = prior_relative_return.reindex(index)
    active = False
    episode_handled = False
    remaining = 0
    confirmations = 0
    previous_active = False
    rows: list[dict[str, object]] = []
    for date in index:
        is_high = bool(high.loc[date])
        entered = False
        recovered = False
        expired = False
        if not is_high:
            active = False
            episode_handled = False
            remaining = 0
            confirmations = 0
        elif not episode_handled:
            active = True
            episode_handled = True
            remaining = maximum_hold_sessions
            confirmations = 0
            entered = True

        value = relative.loc[date]
        if active:
            if pd.notna(value) and float(value) > 0.0:
                confirmations += 1
            else:
                confirmations = 0
            if confirmations >= recovery_confirmations:
                active = False
                recovered = True
                remaining = 0
            else:
                remaining -= 1
                if remaining <= 0:
                    expired = True

        implemented_active = bool(
            active or (expired and not recovered)
        )
        state_change = implemented_active != previous_active
        rows.append(
            {
                "high_volatility_episode_entry": entered,
                "pulse_veto_active": implemented_active,
                "pulse_veto_recovered": recovered,
                "pulse_veto_expired": expired,
                "pulse_veto_state_change": state_change,
                "pulse_veto_remaining_sessions": max(remaining, 0),
                "pulse_veto_recovery_confirmations": confirmations,
            }
        )
        if expired:
            active = False
            remaining = 0
        previous_active = implemented_active
    return pd.DataFrame(rows, index=index)


def causal_pulse_gated_industry_momentum(
    closes: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    lookback_days: int = r21.MOMENTUM_DAYS,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
    high_quantile: float = HIGH_QUANTILE,
) -> pd.DataFrame:
    diagnostics = causal_daily_gated_industry_momentum(
        closes,
        index,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        max_tilt=max_tilt,
        high_quantile=high_quantile,
    ).copy()
    close_returns = closes[["QQQ", "SEMIS"]].pct_change(
        fill_method=None
    )
    prior_relative = (
        close_returns["SEMIS"] - close_returns["QQQ"]
    ).shift(1).reindex(index)
    pulse = pulse_cooldown_state(
        diagnostics["volatility_gate_active"],
        prior_relative,
    )
    ungated_share = diagnostics["ungated_semis_growth_share"]
    diagnostics["persistent_volatility_gate_active"] = diagnostics[
        "volatility_gate_active"
    ]
    diagnostics = diagnostics.drop(
        columns=[
            "semis_growth_share",
            "qqq_growth_share",
            "gate_share_change",
        ]
    ).join(pulse)
    diagnostics["prior_semis_minus_qqq_return"] = prior_relative
    diagnostics["volatility_gate_active"] = diagnostics[
        "pulse_veto_active"
    ]
    diagnostics["semis_growth_share"] = ungated_share.where(
        ~diagnostics["pulse_veto_active"],
        0.50,
    )
    diagnostics["qqq_growth_share"] = (
        1.0 - diagnostics["semis_growth_share"]
    )
    diagnostics["gate_share_change"] = (
        diagnostics["semis_growth_share"] - ungated_share
    )
    return diagnostics


def apply_pulse_gated_industry_momentum(
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
    diagnostics = causal_pulse_gated_industry_momentum(
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
        raise AssertionError("Pulse risk veto changed the growth budget")
    if adjusted[["QQQ", "SEMIS"]].lt(-1e-12).any().any():
        raise AssertionError("Pulse risk veto created short growth weight")
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


def main() -> None:
    r24.OUTPUT = OUTPUT
    r24.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r24.apply_gated_industry_momentum = (
        apply_pulse_gated_industry_momentum
    )
    r24.main()
    acceptance_path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(acceptance_path)
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    pulse_duration_pass = bool(
        diagnostics.groupby(
            diagnostics[
                "pulse_veto_active"
            ].astype(bool).ne(
                diagnostics["pulse_veto_active"]
                .astype(bool)
                .shift()
            ).cumsum()
        )["pulse_veto_active"].sum().max()
        <= MAXIMUM_HOLD_SESSIONS
    )
    acceptance["candidate"] = "r24_with_pulse_relative_tilt_risk_veto"
    acceptance["pulse_duration_pass"] = pulse_duration_pass
    acceptance["statistical_economic_pass"] = (
        acceptance["statistical_economic_pass"].astype(bool)
        & pulse_duration_pass
    )
    acceptance["production_pass"] = (
        acceptance["production_pass"].astype(bool)
        & pulse_duration_pass
    )
    acceptance.to_csv(acceptance_path, index=False)
    print(acceptance.round(6).to_string(index=False))
    print(f"R26 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
