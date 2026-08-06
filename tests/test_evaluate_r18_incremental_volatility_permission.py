from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r10_gde_capital_efficiency import ASSETS
from evaluate_r18_incremental_volatility_permission import (
    causal_volatility_permission,
    volatility_trend_schedule,
)


def _weights(index: pd.DatetimeIndex) -> pd.DataFrame:
    target = {asset: 0.0 for asset in ASSETS}
    target["QQQ"] = 0.50
    target["SEMIS"] = 0.30
    target["GOLD"] = 0.20
    return pd.DataFrame([target] * len(index), index=index)


def test_volatility_permission_uses_only_prior_close() -> None:
    index = pd.bdate_range("2023-01-01", periods=80)
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, (len(index), 2))
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=index,
        columns=["QQQ", "SEMIS"],
    )
    original = causal_volatility_permission(closes)
    changed = closes.copy()
    changed.loc[index[-1], ["QQQ", "SEMIS"]] = [1.0, 1000.0]

    revised = causal_volatility_permission(changed)

    assert revised.iloc[-1] == original.iloc[-1]


def test_both_assets_must_have_calm_short_volatility() -> None:
    index = pd.bdate_range("2023-01-01", periods=100)
    calm = np.full(len(index), 0.001)
    semis = np.concatenate(
        [np.full(75, 0.001), np.tile([0.08, -0.08], 13)[:25]]
    )
    closes = pd.DataFrame(
        {
            "QQQ": 100.0 * np.exp(np.cumsum(calm)),
            "SEMIS": 100.0 * np.exp(np.cumsum(semis)),
        },
        index=index,
    )

    permission = causal_volatility_permission(closes)

    assert not bool(permission.iloc[-1])


def test_volatility_change_waits_for_base_trade() -> None:
    index = pd.bdate_range("2022-01-01", periods=240)
    qqq = np.linspace(100.0, 200.0, len(index))
    semis = np.linspace(100.0, 220.0, len(index))
    closes = pd.DataFrame(
        {"QQQ": qqq, "SEMIS": semis},
        index=index,
    )
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    daily.loc[index[210], "turnover"] = 1.0

    _, _, diagnostics = volatility_trend_schedule(
        _weights(index),
        daily,
        closes,
        active_multiplier=1.18,
    )

    changes = diagnostics[
        "raw_volatility_permission"
    ].ne(
        diagnostics["raw_volatility_permission"].shift(1)
    )
    non_trade_changes = changes & ~diagnostics["base_trade"]
    if non_trade_changes.any():
        dates = diagnostics.index[non_trade_changes]
        for date in dates:
            location = diagnostics.index.get_loc(date)
            if location == 0:
                continue
            assert (
                diagnostics.loc[date, "accepted_incremental_permission"]
                == diagnostics.iloc[location - 1][
                    "accepted_incremental_permission"
                ]
            )


def test_cash_floor_is_enforced_on_target() -> None:
    index = pd.bdate_range("2022-01-01", periods=240)
    closes = pd.DataFrame(
        {
            "QQQ": np.linspace(100.0, 200.0, len(index)),
            "SEMIS": np.linspace(100.0, 220.0, len(index)),
        },
        index=index,
    )
    daily = pd.DataFrame({"turnover": 1.0}, index=index)

    implemented, _, _ = volatility_trend_schedule(
        _weights(index),
        daily,
        closes,
        active_multiplier=1.20,
        cash_floor=-0.30,
    )

    assert implemented["CASH"].min() >= -0.30 - 1e-12


def test_invalid_multiplier_is_rejected() -> None:
    index = pd.bdate_range("2022-01-01", periods=80)
    closes = pd.DataFrame(
        {
            "QQQ": np.linspace(100.0, 110.0, len(index)),
            "SEMIS": np.linspace(100.0, 112.0, len(index)),
        },
        index=index,
    )
    daily = pd.DataFrame({"turnover": 1.0}, index=index)

    with pytest.raises(ValueError):
        volatility_trend_schedule(
            _weights(index),
            daily,
            closes,
            active_multiplier=0.99,
        )
