from __future__ import annotations

import pandas as pd
import pytest

from tools.evaluate_student_t_hmm import stressed_return


def test_double_cost_subtracts_one_more_copy_of_trading_cost() -> None:
    frame = pd.DataFrame(
        {"net_return": [0.01, -0.02], "trading_cost": [0.001, 0.002]}
    )
    assert stressed_return(frame).tolist() == pytest.approx([0.009, -0.022])
