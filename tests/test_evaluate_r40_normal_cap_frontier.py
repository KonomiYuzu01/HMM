import pytest

from tools.evaluate_r40_normal_cap_frontier import effective_non_cash_cap


def test_normal_state_uses_declared_extension() -> None:
    assert effective_non_cash_cap(1.0, 1.08) == pytest.approx(1.08)


def test_controlled_state_never_uses_normal_extension() -> None:
    assert effective_non_cash_cap(0.65, 1.20) == pytest.approx(0.65)


def test_rejects_undeclared_leverage() -> None:
    with pytest.raises(ValueError, match="normal_cap"):
        effective_non_cash_cap(1.0, 1.21)
