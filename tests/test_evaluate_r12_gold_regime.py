from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_r12_gold_regime import (
    GoldRegimeCandidate,
    apply_gold_regime,
    candidate_gold_multiplier,
    candidate_family,
    causal_gold_signals,
    gold_drawdown_episodes,
    gold_multiplier,
)
from tools.fetch_gold_macro_inputs import parse_fred_csv


def test_candidate_family_is_unique_and_pre_registered() -> None:
    candidates = candidate_family()

    assert len(candidates) == 45
    assert len({candidate.name for candidate in candidates}) == 45


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("graded", [0.25, 0.50, 0.75, 1.00]),
        ("majority", [0.25, 0.25, 1.00, 1.00]),
        ("veto", [0.25, 1.00, 1.00, 1.00]),
    ],
)
def test_gold_multiplier_mapping(mode: str, expected: list[float]) -> None:
    votes = pd.Series([0.0, 1.0, 2.0, 3.0, np.nan])

    result = gold_multiplier(votes, floor_fraction=0.25, mapping_mode=mode)

    assert result.iloc[:4].tolist() == pytest.approx(expected)
    assert result.iloc[4] == pytest.approx(1.0)


def test_causal_gold_signals_ignore_same_day_information() -> None:
    index = pd.bdate_range("2024-01-01", periods=90)
    closes = pd.DataFrame(
        {
            "GOLD": np.linspace(100.0, 130.0, len(index)),
            "CASH": np.linspace(100.0, 101.0, len(index)),
        },
        index=index,
    )
    macro = pd.DataFrame(
        {
            "DFII10": np.linspace(2.0, 1.0, len(index)),
            "DTWEXBGS": np.linspace(110.0, 100.0, len(index)),
            "T10YIE": np.linspace(2.0, 2.2, len(index)),
        },
        index=index,
    )
    original = causal_gold_signals(closes, macro, horizon_days=21)

    changed_closes = closes.copy()
    changed_macro = macro.copy()
    changed_closes.loc[index[-1], "GOLD"] = 1_000.0
    changed_macro.loc[index[-1], "DFII10"] = 99.0
    changed_macro.loc[index[-1], "DTWEXBGS"] = 999.0
    changed = causal_gold_signals(
        changed_closes,
        changed_macro,
        horizon_days=21,
    )

    pd.testing.assert_series_equal(
        original.loc[index[-1]],
        changed.loc[index[-1]],
    )


def test_gold_regime_only_moves_risk_on_gold_to_cash() -> None:
    index = pd.bdate_range("2024-01-01", periods=2)
    weights = pd.DataFrame(
        {
            "SPX": [0.10, 0.05],
            "QQQ": [0.30, 0.10],
            "SEMIS": [0.30, 0.10],
            "GOLD": [0.20, 0.20],
            "BONDS": [0.00, 0.35],
            "COMMODITIES": [0.00, 0.05],
            "CASH": [0.10, 0.15],
        },
        index=index,
    )
    signals = pd.DataFrame({"support_votes": [0.0, 0.0]}, index=index)
    candidate = GoldRegimeCandidate(63, 0.25, "majority")

    adjusted, diagnostics = apply_gold_regime(
        weights,
        signals,
        candidate,
    )

    assert adjusted.loc[index[0], "GOLD"] == pytest.approx(0.05)
    assert adjusted.loc[index[0], "CASH"] == pytest.approx(0.25)
    assert adjusted.loc[index[1], "GOLD"] == pytest.approx(0.20)
    assert adjusted.loc[index[1], "CASH"] == pytest.approx(0.15)
    pd.testing.assert_frame_equal(
        adjusted[["SPX", "QQQ", "SEMIS", "BONDS", "COMMODITIES"]],
        weights[["SPX", "QQQ", "SEMIS", "BONDS", "COMMODITIES"]],
    )
    assert adjusted.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert diagnostics["risk_on"].tolist() == [1, 0]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("trend_tiered", [1.00, 1.00, 0.625, 0.25]),
        ("trend_confirmed", [1.00, 1.00, 0.25, 0.25]),
    ],
)
def test_hierarchical_rules_require_adverse_gold_trend(
    mode: str,
    expected: list[float],
) -> None:
    signals = pd.DataFrame(
        {
            "gold_support": [1.0, 0.0, 0.0, 0.0],
            "real_yield_support": [0.0, 1.0, 1.0, 0.0],
            "dollar_support": [0.0, 1.0, 0.0, 0.0],
            "support_votes": [1.0, 2.0, 1.0, 0.0],
        }
    )
    candidate = GoldRegimeCandidate(126, 0.25, mode)

    result = candidate_gold_multiplier(signals, candidate)

    assert result.tolist() == pytest.approx(expected)


def test_gold_drawdown_episodes_records_recovered_and_open_events() -> None:
    prices = pd.Series(
        [100.0, 80.0, 101.0, 100.0, 84.0],
        index=pd.bdate_range("2024-01-01", periods=5),
    )

    episodes = gold_drawdown_episodes(prices, threshold=-0.15)

    assert len(episodes) == 2
    assert episodes[0]["gold_drawdown"] == pytest.approx(-0.20)
    assert episodes[1]["gold_drawdown"] == pytest.approx(84.0 / 101.0 - 1.0)


def test_parse_fred_csv_validates_columns_and_values() -> None:
    content = b"observation_date,DFII10\n2024-01-01,1.23\n2024-01-02,.\n"

    series = parse_fred_csv(content, "DFII10")

    assert series.name == "DFII10"
    assert series.iloc[0] == pytest.approx(1.23)
    assert pd.isna(series.iloc[1])

    with pytest.raises(ValueError, match="Unexpected FRED columns"):
        parse_fred_csv(content, "DTWEXBGS")
