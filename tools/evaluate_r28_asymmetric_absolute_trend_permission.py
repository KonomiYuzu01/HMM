from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r21_current_engine_industry_momentum import (
    MAX_TILT,
    REBALANCE_DAYS,
)
from tools.evaluate_r23_high_volatility_tilt_gate import HIGH_QUANTILE
from tools.evaluate_r27_absolute_trend_permission import (
    ABSOLUTE_TREND_DAYS,
)
import tools.evaluate_r27_absolute_trend_permission as r27
from tools.evaluate_r26_pulse_relative_tilt_risk_veto import (
    causal_pulse_gated_industry_momentum,
)


OUTPUT = Path("output/r28_asymmetric_absolute_trend_permission")
CUMULATIVE_TRIALS = 101


def asymmetric_absolute_permission(
    prior_excess_trend: pd.Series,
    scheduled_update: pd.Series,
) -> pd.DataFrame:
    index = prior_excess_trend.index.intersection(
        scheduled_update.index
    )
    excess = prior_excess_trend.reindex(index)
    scheduled = (
        scheduled_update.reindex(index).fillna(False).astype(bool)
    )
    permitted = False
    previous = False
    rows: list[dict[str, object]] = []
    for date in index:
        value = excess.loc[date]
        revoked = False
        entered = False
        if pd.isna(value) or float(value) <= 0.0:
            revoked = permitted
            permitted = False
        elif bool(scheduled.loc[date]) and not permitted:
            permitted = True
            entered = True
        state_change = permitted != previous
        rows.append(
            {
                "absolute_trend_permission": permitted,
                "absolute_trend_permission_entered": entered,
                "absolute_trend_permission_revoked": revoked,
                "absolute_trend_permission_state_change": state_change,
            }
        )
        previous = permitted
    return pd.DataFrame(rows, index=index)


def causal_asymmetric_absolute_industry_momentum(
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
    permission = asymmetric_absolute_permission(
        trailing_excess,
        diagnostics["momentum_update"],
    )
    raw_share = diagnostics["ungated_semis_growth_share"]
    permitted_share = raw_share.where(
        permission["absolute_trend_permission"],
        0.50,
    )
    final_share = permitted_share.where(
        ~diagnostics["pulse_veto_active"].astype(bool),
        0.50,
    )
    diagnostics["raw_momentum_semis_growth_share"] = raw_share
    diagnostics["prior_pair_excess_log_return"] = trailing_excess
    diagnostics["raw_absolute_trend_permission"] = (
        trailing_excess.gt(0.0)
    )
    diagnostics = diagnostics.join(permission)
    diagnostics["absolute_permitted_semis_growth_share"] = (
        permitted_share
    )
    diagnostics["ungated_semis_growth_share"] = permitted_share
    diagnostics["semis_growth_share"] = final_share
    diagnostics["qqq_growth_share"] = 1.0 - final_share
    diagnostics["gate_share_change"] = final_share - raw_share
    return diagnostics


def apply_asymmetric_absolute_industry_momentum(
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
    diagnostics = causal_asymmetric_absolute_industry_momentum(
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
        raise AssertionError("Asymmetric permission changed growth budget")
    if adjusted[["QQQ", "SEMIS"]].lt(-1e-12).any().any():
        raise AssertionError(
            "Asymmetric permission created short growth weight"
        )
    updated_daily = execution_daily.loc[index].copy()
    update = (
        diagnostics["momentum_update"].astype(bool)
        | diagnostics["pulse_veto_state_change"].astype(bool)
        | diagnostics[
            "absolute_trend_permission_state_change"
        ].astype(bool)
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
    r27.OUTPUT = OUTPUT
    r27.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r27.apply_absolute_permitted_industry_momentum = (
        apply_asymmetric_absolute_industry_momentum
    )
    r27.main()
    acceptance_path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(acceptance_path)
    acceptance["candidate"] = (
        "r26_with_asymmetric_absolute_trend_permission"
    )
    acceptance.to_csv(acceptance_path, index=False)
    print(f"R28 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
