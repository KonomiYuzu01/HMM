from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r21_current_engine_industry_momentum import (
    MAX_TILT,
    REBALANCE_DAYS,
    causal_industry_momentum,
)
import tools.evaluate_r24_declared_cash_hard_limit as r24
from tools.evaluate_r23_high_volatility_tilt_gate import (
    HIGH_QUANTILE,
    LOW_QUANTILE,
    MINIMUM_THRESHOLD_OBSERVATIONS,
    THRESHOLD_LOOKBACK,
    VOLATILITY_WINDOW,
)


OUTPUT = Path("output/r25_daily_relative_tilt_risk_veto")
CUMULATIVE_TRIALS = 94


def causal_daily_gated_industry_momentum(
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
    realized = (
        pair_returns.rolling(
            VOLATILITY_WINDOW,
            min_periods=VOLATILITY_WINDOW,
        )
        .std(ddof=1)
        .mul(np.sqrt(252.0))
    )
    prior_realized = realized.shift(1)
    low_threshold = (
        realized.rolling(
            THRESHOLD_LOOKBACK,
            min_periods=MINIMUM_THRESHOLD_OBSERVATIONS,
        )
        .quantile(LOW_QUANTILE)
        .shift(1)
    )
    high_threshold = (
        realized.rolling(
            THRESHOLD_LOOKBACK,
            min_periods=MINIMUM_THRESHOLD_OBSERVATIONS,
        )
        .quantile(high_quantile)
        .shift(1)
    )
    prior_realized = prior_realized.reindex(index)
    low_threshold = low_threshold.reindex(index)
    high_threshold = high_threshold.reindex(index)
    sufficient = (
        prior_realized.notna()
        & low_threshold.notna()
        & high_threshold.notna()
    )
    states = pd.Series("insufficient", index=index, dtype=object)
    states.loc[sufficient & prior_realized.le(low_threshold)] = "low"
    states.loc[
        sufficient
        & prior_realized.gt(low_threshold)
        & prior_realized.lt(high_threshold)
    ] = "medium"
    states.loc[sufficient & prior_realized.ge(high_threshold)] = "high"
    gate_active = states.eq("high")
    ungated_share = diagnostics["semis_growth_share"]
    gated_share = ungated_share.where(~gate_active, 0.50)
    gate_change = gate_active.ne(gate_active.shift(1)).fillna(
        gate_active
    )
    diagnostics["volatility_state"] = states
    diagnostics["pair_realized_volatility"] = prior_realized
    diagnostics["low_volatility_threshold"] = low_threshold
    diagnostics["high_volatility_threshold"] = high_threshold
    diagnostics["volatility_gate_active"] = gate_active
    diagnostics["volatility_gate_change"] = gate_change
    diagnostics["ungated_semis_growth_share"] = ungated_share
    diagnostics["semis_growth_share"] = gated_share
    diagnostics["qqq_growth_share"] = 1.0 - gated_share
    diagnostics["gate_share_change"] = gated_share - ungated_share
    return diagnostics


def apply_daily_gated_industry_momentum(
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
    diagnostics = causal_daily_gated_industry_momentum(
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
        raise AssertionError("Daily risk veto changed the growth budget")
    if adjusted[["QQQ", "SEMIS"]].lt(-1e-12).any().any():
        raise AssertionError("Daily risk veto created short growth weight")
    updated_daily = execution_daily.loc[index].copy()
    update = (
        diagnostics["momentum_update"].astype(bool)
        | diagnostics["volatility_gate_change"].astype(bool)
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
        apply_daily_gated_industry_momentum
    )
    r24.main()
    acceptance_path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(acceptance_path)
    acceptance["candidate"] = "r24_with_daily_relative_tilt_risk_veto"
    acceptance.to_csv(acceptance_path, index=False)
    print(f"R25 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
