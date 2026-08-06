import pandas as pd
import pytest

from regime_strategy.forward_monitoring import append_immutable


def test_append_immutable_is_idempotent(tmp_path) -> None:
    path = tmp_path / "signals.csv"
    row = {"strategy": "candidate", "date": "2026-07-22", "weight": 0.25}
    append_immutable(path, row, ["strategy", "date"])
    append_immutable(path, row, ["strategy", "date"])
    stored = pd.read_csv(path)
    assert len(stored) == 1
    assert stored.iloc[0]["weight"] == 0.25


def test_append_immutable_rejects_changed_historical_signal(tmp_path) -> None:
    path = tmp_path / "signals.csv"
    append_immutable(
        path,
        {"strategy": "candidate", "date": "2026-07-22", "weight": 0.25},
        ["strategy", "date"],
    )
    with pytest.raises(RuntimeError, match="Immutable forward signal conflicts"):
        append_immutable(
            path,
            {"strategy": "candidate", "date": "2026-07-22", "weight": 0.30},
            ["strategy", "date"],
        )


def test_append_immutable_accepts_round_trip_float_noise(tmp_path) -> None:
    path = tmp_path / "signals.csv"
    append_immutable(
        path,
        {
            "strategy": "candidate",
            "date": "2026-07-22",
            "weight": 0.4252530189781724,
        },
        ["strategy", "date"],
    )
    append_immutable(
        path,
        {
            "strategy": "candidate",
            "date": "2026-07-22",
            "weight": 0.42525301942525867,
        },
        ["strategy", "date"],
    )
    assert len(pd.read_csv(path)) == 1
