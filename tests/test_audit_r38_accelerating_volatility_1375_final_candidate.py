from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from tools.audit_r38_accelerating_volatility_1375_final_candidate import (
    path_digest,
    validate_market_frame,
)


def test_path_digest_is_stable_below_rounding_precision() -> None:
    values = pd.Series([0.1, -0.2, 0.3])
    changed = values.copy()
    changed.iloc[0] += 1e-16
    assert path_digest(values) == path_digest(changed)


def test_market_frame_audit_detects_duplicate_and_missing() -> None:
    index = pd.to_datetime(
        ["2026-01-02", "2026-01-02", "2026-01-05"]
    )
    frame = pd.DataFrame(
        {"QQQ": [1.0, np.nan, 2.0]},
        index=index,
    )
    result = validate_market_frame(frame)
    assert result["duplicate_dates"] == 1
    assert result["missing_values"] == 1

