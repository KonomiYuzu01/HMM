from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from tools.evaluate_r10_gde_capital_efficiency import ASSETS
from tools.evaluate_r38_targeted_crowding_hysteresis import (
    apply_targeted_crowding_cap,
    causal_hysteresis_crowding_state,
)


def _inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    index = pd.bdate_range("2014-01-02", periods=1_700)
    semis_returns = np.full(len(index), 0.0004)
    semis_returns[-220:-40] = 0.003
    semis_returns[-40:] = -0.004
    closes = pd.DataFrame(
        {
            "QQQ": 100.0
            * np.exp(np.cumsum(np.full(len(index), 0.0004))),
            "SEMIS": 100.0 * np.exp(np.cumsum(semis_returns)),
        },
        index=index,
    )
    full = pd.DataFrame(0.0, index=index, columns=ASSETS)
    full["QQQ"] = 0.45
    full["SEMIS"] = 0.45
    full["GOLD"] = 0.25
    full["USD"] = 0.05
    full["CASH"] = -0.20
    reference = full.copy()
    reference["QQQ"] = 0.38
    reference["SEMIS"] = 0.38
    reference["CASH"] = -0.06
    daily = pd.DataFrame({"turnover": 0.0}, index=index)
    return closes, full, reference, daily


def test_hysteresis_requires_confirmed_price_recovery() -> None:
    closes, _, _, _ = _inputs()
    diagnostics = causal_hysteresis_crowding_state(
        closes,
        closes.index,
        entry_percentile=0.90,
        exit_percentile=0.85,
    )
    active = diagnostics["targeted_crowding_active"]
    assert active.any()
    exits = active.shift(1).fillna(False) & ~active
    if exits.any():
        assert diagnostics.loc[exits, "confirmed_crowding_exit"].all()
        assert diagnostics.loc[exits, "price_recovered"].all()


def test_targeted_cap_preserves_non_growth_assets() -> None:
    closes, full, reference, daily = _inputs()
    adjusted, _, diagnostics = apply_targeted_crowding_cap(
        full,
        reference,
        daily,
        closes,
        entry_percentile=0.90,
        exit_percentile=0.85,
    )
    non_growth = [
        asset
        for asset in ASSETS
        if asset not in ("QQQ", "SEMIS", "CASH")
    ]
    assert np.allclose(adjusted[non_growth], full[non_growth])
    assert diagnostics["targeted_increase_error"].le(1e-12).all()
    assert diagnostics["targeted_conservation_error"].le(1e-12).all()


def test_disabled_targeted_cap_is_exact_identity() -> None:
    closes, full, reference, daily = _inputs()
    adjusted, _, _ = apply_targeted_crowding_cap(
        full,
        reference,
        daily,
        closes,
        targeted_cap_enabled=False,
    )
    assert np.allclose(adjusted[ASSETS], full[ASSETS])
