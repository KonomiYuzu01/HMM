from pathlib import Path

import pandas as pd

from tools.evaluate_hmm_ninth_wave import _average_member_switch_rows


def _write_regimes(directory: Path, name: str, values: list[int]) -> None:
    member = directory / "members" / name
    member.mkdir(parents=True)
    pd.DataFrame(
        {"paper_risk_on_candidate": values},
        index=pd.bdate_range("2024-01-01", periods=len(values)),
    ).to_csv(member / "regimes.csv")


def test_multi_lookback_switches_compare_per_member_averages(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_regimes(baseline, "seed_7", [0, 1, 1])
    _write_regimes(baseline, "seed_42", [0, 1, 0])
    _write_regimes(candidate, "member_a", [0, 0, 0])
    _write_regimes(candidate, "member_b", [0, 1, 1])
    _write_regimes(candidate, "member_c", [0, 1, 0])

    row = _average_member_switch_rows("normal", baseline, candidate)[0]

    assert row["selected_order_switches"] == 1.5
    assert row["candidate_switches"] == 1.0
