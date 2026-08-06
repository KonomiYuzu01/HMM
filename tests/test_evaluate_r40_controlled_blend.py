from __future__ import annotations

import pandas as pd
import pytest

from tools.evaluate_r40_controlled_blend import blend_controlled_targets


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2026-01-01", periods=2)
    r11 = pd.DataFrame(
        {"QQQ": [0.4, 0.3], "GOLD": [0.2, 0.3], "CASH": [0.4, 0.4]},
        index=index,
    )
    staged = pd.DataFrame(
        {"QQQ": [0.6, 0.5], "GOLD": [0.2, 0.2], "CASH": [0.2, 0.3]},
        index=index,
    )
    return r11, staged


def test_endpoints_reproduce_source_targets() -> None:
    r11, staged = frames()
    pd.testing.assert_frame_equal(blend_controlled_targets(r11, staged, 0.0), r11)
    pd.testing.assert_frame_equal(blend_controlled_targets(r11, staged, 1.0), staged)


def test_midpoint_is_convex_and_sums_to_one() -> None:
    r11, staged = frames()
    result = blend_controlled_targets(r11, staged, 0.5)
    pd.testing.assert_frame_equal(result, (r11 + staged) / 2.0)
    assert result.sum(axis=1).eq(1.0).all()


@pytest.mark.parametrize("share", [-0.01, 1.01])
def test_invalid_share_is_rejected(share: float) -> None:
    r11, staged = frames()
    with pytest.raises(ValueError):
        blend_controlled_targets(r11, staged, share)
