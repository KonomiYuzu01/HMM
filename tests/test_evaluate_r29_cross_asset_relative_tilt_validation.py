from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_r29_cross_asset_relative_tilt_validation import (
    first_ready_date,
    maximum_true_run,
    pair_diagnostics,
    simulate_open_portfolio,
)


def synthetic_opens(periods: int = 40) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    return pd.DataFrame(
        {
            "QQQ": 100.0 * np.cumprod(np.full(periods, 1.001)),
            "ASSET": 100.0 * np.cumprod(np.full(periods, 1.002)),
        },
        index=index,
    )


def test_simulation_preserves_budget_and_target_on_trade_dates() -> None:
    opens = synthetic_opens()
    target = pd.Series(0.80, index=opens.index)
    update = pd.Series(False, index=opens.index)
    update.iloc[[0, 21]] = True
    path = simulate_open_portfolio(
        opens,
        target,
        update,
        start=opens.index[0],
        cost_bps=7.5,
    )
    assert path["budget_error"].le(1e-12).all()
    assert np.allclose(
        path.loc[path["trade"], "implemented_asset_weight"],
        0.80,
    )


def test_higher_cost_cannot_raise_terminal_equity() -> None:
    opens = synthetic_opens()
    target = pd.Series(
        np.where(np.arange(len(opens)) % 2 == 0, 0.90, 0.10),
        index=opens.index,
    )
    update = pd.Series(True, index=opens.index)
    low = simulate_open_portfolio(
        opens,
        target,
        update,
        start=opens.index[0],
        cost_bps=7.5,
    )
    high = simulate_open_portfolio(
        opens,
        target,
        update,
        start=opens.index[0],
        cost_bps=15.0,
    )
    assert high["equity"].iloc[-1] < low["equity"].iloc[-1]


def test_no_update_allows_weights_to_drift_without_hidden_trading() -> None:
    opens = synthetic_opens()
    target = pd.Series(0.50, index=opens.index)
    update = pd.Series(False, index=opens.index)
    path = simulate_open_portfolio(
        opens,
        target,
        update,
        start=opens.index[0],
        cost_bps=7.5,
    )
    assert path["trade"].sum() == 1
    assert path["turnover"].iloc[1:].eq(0.0).all()
    assert path["implemented_asset_weight"].iloc[-1] > 0.50


def test_maximum_true_run_counts_only_consecutive_sessions() -> None:
    state = pd.Series(
        [False, True, True, False, True, True, True, False]
    )
    assert maximum_true_run(state) == 3


def test_current_close_cannot_change_current_open_target() -> None:
    index = pd.bdate_range("2017-01-02", periods=1_300)
    rng = np.random.default_rng(29)
    common = rng.normal(0.0004, 0.012, len(index))
    panel = pd.DataFrame(
        {
            "close_QQQ": 100.0
            * np.cumprod(
                1.0 + common + rng.normal(0.0, 0.002, len(index))
            ),
            "close_XLK": 100.0
            * np.cumprod(
                1.0 + common + rng.normal(0.0002, 0.004, len(index))
            ),
        },
        index=index,
    )
    before = pair_diagnostics(panel, "XLK")
    changed = panel.copy()
    changed.iloc[-1, changed.columns.get_loc("close_XLK")] *= 0.50
    after = pair_diagnostics(changed, "XLK")
    assert np.isclose(
        before.iloc[-1]["semis_growth_share"],
        after.iloc[-1]["semis_growth_share"],
    )
    assert first_ready_date(before) == first_ready_date(after)
