from __future__ import annotations

import numpy as np
import pandas as pd


_BUSINESS_DAY_ANCHOR = np.datetime64("1970-01-01", "D")


def anchored_business_day_position(date: pd.Timestamp) -> int:
    """Return a stable weekday count that is independent of data-history start."""
    normalized = np.datetime64(pd.Timestamp(date).date(), "D")
    return int(np.busday_count(_BUSINESS_DAY_ANCHOR, normalized))
