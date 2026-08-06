import numpy as np
import pytest

from tools.evaluate_hierarchical_regime_portfolios import ASSETS
from tools.evaluate_r10_crisis_decline_overlay import crisis_tilt_target


def test_crisis_tilt_preserves_cash_vix_and_total_weight() -> None:
    anchor = np.zeros(len(ASSETS))
    anchor[ASSETS.index("QQQ")] = 0.40
    anchor[ASSETS.index("SEMIS")] = 0.30
    anchor[ASSETS.index("GOLD")] = 0.10
    anchor[ASSETS.index("CASH")] = 0.16
    anchor[ASSETS.index("VIX_HEDGE")] = 0.04
    target = crisis_tilt_target(anchor, 0.50)
    assert target[ASSETS.index("CASH")] == pytest.approx(0.16)
    assert target[ASSETS.index("VIX_HEDGE")] == pytest.approx(0.04)
    assert target.sum() == pytest.approx(anchor.sum())
    assert target[ASSETS.index("GOLD")] > anchor[ASSETS.index("GOLD")]
    assert (
        target[ASSETS.index("QQQ")]
        + target[ASSETS.index("SEMIS")]
        < anchor[ASSETS.index("QQQ")]
        + anchor[ASSETS.index("SEMIS")]
    )
