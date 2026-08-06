import pandas as pd

from regime_strategy.schedule import anchored_business_day_position


def test_anchored_position_does_not_depend_on_available_history() -> None:
    date = pd.Timestamp("2026-07-24")
    full_history = pd.bdate_range("2008-01-01", date)
    shortened_history = pd.bdate_range("2009-09-18", date)

    assert full_history.get_loc(date) != shortened_history.get_loc(date)
    assert anchored_business_day_position(full_history[-1]) == (
        anchored_business_day_position(shortened_history[-1])
    )


def test_anchored_position_advances_once_per_weekday() -> None:
    friday = anchored_business_day_position(pd.Timestamp("2026-07-24"))
    monday = anchored_business_day_position(pd.Timestamp("2026-07-27"))
    assert monday == friday + 1
