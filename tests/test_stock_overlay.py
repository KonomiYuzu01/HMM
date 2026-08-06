from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_strategy.stock_overlay import (
    active_relative_strength_confirmed,
    allocate_joint_replacement,
    allocate_standalone_replacement,
    estimate_active_risk,
    estimate_basket_risk,
    regime_replacement_share,
)


def return_frame(seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=320)
    benchmark = pd.Series(
        rng.normal(0.0004, 0.012, len(index)),
        index=index,
        name="ETF",
    )
    stocks = pd.DataFrame(
        {
            "A": benchmark + rng.normal(0.0, 0.010, len(index)),
            "B": benchmark + rng.normal(0.0, 0.010, len(index)),
            "C": benchmark + rng.normal(0.0, 0.010, len(index)),
        },
        index=index,
    )
    return stocks, benchmark


def test_identical_benchmark_basket_has_one_times_risk() -> None:
    _, benchmark = return_frame()
    stocks = pd.concat(
        [benchmark.rename("A"), benchmark.rename("B")],
        axis=1,
    )
    estimate = estimate_basket_risk(stocks, benchmark)
    assert np.isclose(estimate.risk_multiple, 1.0)
    assert np.isclose(estimate.beta, 1.0)


def test_joint_sizing_recognizes_idiosyncratic_diversification() -> None:
    stocks, benchmark = return_frame()
    joint = allocate_joint_replacement(0.20, stocks, benchmark)
    standalone = allocate_standalone_replacement(0.20, stocks, benchmark)
    assert joint.actual_stock_weight > standalone.actual_stock_weight
    assert joint.actual_stock_weight <= 0.20


def test_joint_sizing_respects_name_and_basket_caps() -> None:
    stocks, benchmark = return_frame()
    allocation = allocate_joint_replacement(
        0.50,
        stocks,
        benchmark,
        maximum_name_weight=0.04,
        maximum_basket_weight=0.10,
    )
    assert allocation.stock_weights.max() <= 0.04 + 1.0e-12
    assert allocation.actual_stock_weight <= 0.10 + 1.0e-12
    assert np.isclose(
        allocation.cash_reserve_weight,
        allocation.risk_budget - allocation.actual_stock_weight,
    )


def test_reentry_uses_etf_before_individual_stocks() -> None:
    assert regime_replacement_share("quiet_bull", 0.20) == 0.0
    assert regime_replacement_share("quiet_bull", 0.35) == 0.0
    assert regime_replacement_share("quiet_bull", 0.60) == 0.50
    assert regime_replacement_share("fragile_bull", 0.60) == 0.25
    assert regime_replacement_share("crisis_recovery", 0.60) == 0.0


def test_regime_replacement_share_accepts_candidate_state_map() -> None:
    shares = {
        "quiet_bull": 0.50,
        "normal_bull": 0.25,
        "fragile_bull": 0.0,
        "rebound": 0.0,
    }
    assert (
        regime_replacement_share(
            "normal_bull",
            0.60,
            replacement_shares=shares,
        )
        == 0.25
    )
    assert (
        regime_replacement_share(
            "rebound",
            0.60,
            replacement_shares=shares,
        )
        == 0.0
    )


def test_active_relative_strength_uses_only_supplied_history() -> None:
    _, benchmark = return_frame(seed=12)
    stronger = pd.DataFrame({"A": benchmark + 0.0005})
    weaker = pd.DataFrame({"A": benchmark - 0.0005})

    assert active_relative_strength_confirmed(
        stronger,
        benchmark,
        windows=(63, 126),
    )
    assert not active_relative_strength_confirmed(
        weaker,
        benchmark,
        windows=(63, 126),
    )
    assert not active_relative_strength_confirmed(
        stronger.iloc[:60],
        benchmark.iloc[:60],
        windows=(63,),
    )


def test_active_risk_is_zero_for_an_etf_clone() -> None:
    _, benchmark = return_frame(seed=9)
    stocks = pd.DataFrame({"CLONE": benchmark})

    estimate = estimate_active_risk(stocks, benchmark)

    assert estimate.long_tracking_error == pytest.approx(0.0, abs=1.0e-12)
    assert estimate.short_tracking_error == pytest.approx(0.0, abs=1.0e-12)
    assert estimate.basket_beta == pytest.approx(1.0, rel=1.0e-10)
