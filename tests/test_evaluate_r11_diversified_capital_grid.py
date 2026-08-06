import pandas as pd
import pytest

from tools.evaluate_r11_diversified_capital_grid import (
    cap_semis_to_qqq,
    scale_non_cash_weights,
    scale_non_cash_weights_by_series,
)


def test_cap_semis_to_qqq_preserves_account_weight() -> None:
    weights = pd.DataFrame(
        {
            "SPX": [0.0],
            "QQQ": [0.10],
            "SEMIS": [0.80],
            "BOND": [0.0],
            "GOLD": [0.20],
            "OIL": [0.0],
            "USD": [0.0],
            "CASH": [-0.10],
            "VIX_HEDGE": [0.0],
        }
    )

    adjusted = cap_semis_to_qqq(weights, 0.60)

    assert adjusted.loc[0, "SEMIS"] == pytest.approx(0.60)
    assert adjusted.loc[0, "QQQ"] == pytest.approx(0.30)
    assert adjusted.loc[0].sum() == pytest.approx(weights.loc[0].sum())


def test_cap_semis_to_qqq_does_not_raise_lower_weights() -> None:
    weights = pd.DataFrame(
        {
            "SPX": [0.0],
            "QQQ": [0.40],
            "SEMIS": [0.30],
            "BOND": [0.0],
            "GOLD": [0.20],
            "OIL": [0.0],
            "USD": [0.0],
            "CASH": [0.10],
            "VIX_HEDGE": [0.0],
        }
    )

    adjusted = cap_semis_to_qqq(weights, 0.60)

    pd.testing.assert_frame_equal(adjusted, weights)


def test_cap_semis_to_qqq_validates_cap() -> None:
    weights = pd.DataFrame()

    with pytest.raises(ValueError, match="semis_cap"):
        cap_semis_to_qqq(weights, 1.10)


def test_scale_non_cash_weights_uses_cash_as_financing_leg() -> None:
    weights = pd.DataFrame(
        {
            "SPX": [0.0],
            "QQQ": [0.40],
            "SEMIS": [0.40],
            "BOND": [0.0],
            "GOLD": [0.20],
            "OIL": [0.0],
            "USD": [0.0],
            "CASH": [0.0],
            "VIX_HEDGE": [0.0],
        }
    )

    adjusted = scale_non_cash_weights(weights, 1.05)

    assert adjusted.loc[0, "QQQ"] == pytest.approx(0.42)
    assert adjusted.loc[0, "SEMIS"] == pytest.approx(0.42)
    assert adjusted.loc[0, "GOLD"] == pytest.approx(0.21)
    assert adjusted.loc[0, "CASH"] == pytest.approx(-0.05)
    assert adjusted.loc[0].sum() == pytest.approx(1.0)


def test_scale_non_cash_weights_validates_multiplier() -> None:
    with pytest.raises(ValueError, match="risk_multiplier"):
        scale_non_cash_weights(pd.DataFrame(), -0.01)


def test_scale_non_cash_weights_can_preserve_hedge_share() -> None:
    weights = pd.DataFrame(
        {
            "SPX": [0.0],
            "QQQ": [0.40],
            "SEMIS": [0.36],
            "BOND": [0.0],
            "GOLD": [0.20],
            "OIL": [0.0],
            "USD": [0.0],
            "CASH": [0.0],
            "VIX_HEDGE": [0.04],
        }
    )

    adjusted = scale_non_cash_weights(
        weights,
        1.05,
        unscaled_assets=("VIX_HEDGE",),
    )

    assert adjusted.loc[0, "VIX_HEDGE"] == pytest.approx(0.04)
    assert adjusted.loc[0, "CASH"] == pytest.approx(-0.048)
    assert adjusted.loc[0].sum() == pytest.approx(1.0)


def test_scale_non_cash_weights_by_series_allows_causal_gating() -> None:
    index = pd.date_range("2025-01-01", periods=2)
    weights = pd.DataFrame(
        {
            "SPX": [0.0, 0.0],
            "QQQ": [0.40, 0.20],
            "SEMIS": [0.40, 0.20],
            "BOND": [0.0, 0.0],
            "GOLD": [0.20, 0.20],
            "OIL": [0.0, 0.0],
            "USD": [0.0, 0.0],
            "CASH": [0.0, 0.40],
            "VIX_HEDGE": [0.0, 0.0],
        },
        index=index,
    )
    multipliers = pd.Series([1.05, 1.00], index=index)

    adjusted = scale_non_cash_weights_by_series(weights, multipliers)

    assert adjusted.loc[index[0], "CASH"] == pytest.approx(-0.05)
    assert adjusted.loc[index[1], "CASH"] == pytest.approx(0.40)


def test_scale_non_cash_weights_by_series_requires_full_index() -> None:
    weights = pd.DataFrame(
        columns=[
            "SPX",
            "QQQ",
            "SEMIS",
            "BOND",
            "GOLD",
            "OIL",
            "USD",
            "CASH",
            "VIX_HEDGE",
        ],
        index=pd.date_range("2025-01-01", periods=2),
    )

    with pytest.raises(ValueError, match="cover"):
        scale_non_cash_weights_by_series(
            weights,
            pd.Series([1.0], index=weights.index[:1]),
        )
