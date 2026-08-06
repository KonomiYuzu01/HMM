from datetime import date

import pandas as pd

from regime_strategy.vx import build_front_month_curve, monthly_vix_expiry


def test_standard_monthly_vix_expiry_is_thirty_days_before_third_friday() -> None:
    assert monthly_vix_expiry(2015, 1) == date(2015, 1, 21)
    assert monthly_vix_expiry(2024, 12) == date(2024, 12, 18)


def test_front_curve_excludes_a_contract_on_its_expiry_date() -> None:
    records = pd.DataFrame(
        {
            "Trade Date": [
                "2015-01-20",
                "2015-01-20",
                "2015-01-20",
                "2015-01-21",
                "2015-01-21",
                "2015-01-21",
            ],
            "Expiry": [
                "2015-01-21",
                "2015-02-18",
                "2015-03-18",
                "2015-01-21",
                "2015-02-18",
                "2015-03-18",
            ],
            "Settle": [20.0, 21.0, 22.0, 19.0, 20.5, 21.5],
        }
    )
    curve = build_front_month_curve(records)
    assert curve.loc["2015-01-20", "VX1"] == 20.0
    assert curve.loc["2015-01-20", "VX2"] == 21.0
    assert curve.loc["2015-01-21", "VX1"] == 20.5
    assert curve.loc["2015-01-21", "VX2"] == 21.5
