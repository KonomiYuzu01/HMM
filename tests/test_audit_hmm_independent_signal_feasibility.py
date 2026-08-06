from pathlib import Path

import pandas as pd
import pytest

from tools.audit_hmm_independent_signal_feasibility import summarize_columns


def test_summarize_columns_reports_complete_common_history(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2026-07-27", "2026-07-28", "2026-07-29"],
            "A": [1.0, 2.0, 3.0],
            "B": [None, 4.0, 5.0],
        }
    ).to_csv(path, index=False)

    summary = summarize_columns(path, ["A", "B"])

    assert summary["complete_observations"] == 2
    assert summary["first_complete_date"] == "2026-07-28"
    assert summary["last_complete_date"] == "2026-07-29"
    assert summary["business_day_lag_to_reference"] == 1
    assert summary["missing_values_by_column"] == {"A": 0, "B": 1}


def test_summarize_columns_fails_closed_on_duplicate_dates(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {"date": ["2026-07-29", "2026-07-29"], "A": [1.0, 2.0]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="unique and increasing"):
        summarize_columns(path, ["A"])
