from __future__ import annotations

import calendar
from datetime import date, timedelta

import pandas as pd


def monthly_vix_expiry(year: int, month: int) -> date:
    """Return the standard monthly VX expiry before holiday adjustment."""
    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")
    next_year = year + (month == 12)
    next_month = 1 if month == 12 else month + 1
    month_calendar = calendar.monthcalendar(next_year, next_month)
    fridays = [week[calendar.FRIDAY] for week in month_calendar if week[calendar.FRIDAY]]
    third_friday = date(next_year, next_month, fridays[2])
    return third_friday - timedelta(days=30)


def build_front_month_curve(records: pd.DataFrame) -> pd.DataFrame:
    """Select the first two unexpired monthly VX settlements on every trade date."""
    required = {"Trade Date", "Expiry", "Settle"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"VX contract records are missing columns: {sorted(missing)}")

    clean = records.loc[:, ["Trade Date", "Expiry", "Settle"]].copy()
    clean["Trade Date"] = pd.to_datetime(clean["Trade Date"], errors="coerce")
    clean["Expiry"] = pd.to_datetime(clean["Expiry"], errors="coerce")
    clean["Settle"] = pd.to_numeric(clean["Settle"], errors="coerce")
    clean = clean.dropna().loc[lambda frame: frame["Settle"] > 0.0]
    clean = clean.loc[clean["Expiry"] > clean["Trade Date"]]
    clean = clean.sort_values(["Trade Date", "Expiry"]).drop_duplicates(
        ["Trade Date", "Expiry"], keep="last"
    )

    rows: list[dict[str, object]] = []
    for trade_date, group in clean.groupby("Trade Date", sort=True):
        front = group.iloc[:2]
        if len(front) < 2:
            continue
        rows.append(
            {
                "date": trade_date,
                "VX1": float(front.iloc[0]["Settle"]),
                "VX2": float(front.iloc[1]["Settle"]),
                "VX1_EXPIRY": front.iloc[0]["Expiry"],
                "VX2_EXPIRY": front.iloc[1]["Expiry"],
            }
        )
    if not rows:
        raise ValueError("No dates contain two valid unexpired VX contracts")
    curve = pd.DataFrame(rows).set_index("date")
    curve["VX_RATIO"] = curve["VX1"] / curve["VX2"]
    return curve
