from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("output/paper_core_zero_entry_growth_reallocation_ensemble")
DESTINATION = Path("output/model_disagreement_validation")
GROWTH_ASSETS = ["QQQ", "SEMIS"]


def forward_compound(returns: pd.Series, horizon: int) -> pd.Series:
    return (1.0 + returns).rolling(horizon).apply(np.prod, raw=True).shift(
        -(horizon - 1)
    ) - 1.0


def summarize(group: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"state": label, "observations": len(group)}
    for horizon in (5, 21):
        values = group[f"forward_{horizon}d"].dropna()
        row[f"mean_{horizon}d"] = float(values.mean())
        row[f"median_{horizon}d"] = float(values.median())
        row[f"loss_probability_{horizon}d"] = float((values < 0.0).mean())
        row[f"worst_{horizon}d"] = float(values.min())
    return row


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    member_dirs = sorted((SOURCE / "members").glob("seed_*"))
    growth = {}
    scheduled_indices = []
    for member_dir in member_dirs:
        weights = pd.read_csv(
            member_dir / "weights.csv", index_col=0, parse_dates=True
        )
        growth[member_dir.name] = weights[GROWTH_ASSETS].sum(axis=1)
        regimes = pd.read_csv(
            member_dir / "regimes.csv", index_col=0, parse_dates=True
        )
        scheduled_indices.append(regimes.index)
    scheduled = scheduled_indices[0]
    for index in scheduled_indices[1:]:
        scheduled = scheduled.intersection(index)

    growth_frame = pd.DataFrame(growth).reindex(scheduled)
    signal = pd.DataFrame(index=scheduled)
    signal["minimum_growth_exposure"] = growth_frame.min(axis=1)
    signal["mean_growth_exposure"] = growth_frame.mean(axis=1)
    signal["maximum_growth_exposure"] = growth_frame.max(axis=1)
    signal["growth_exposure_range"] = (
        signal["maximum_growth_exposure"] - signal["minimum_growth_exposure"]
    )
    active = growth_frame > 0.10
    signal["risk_on_votes"] = active.sum(axis=1)
    signal["state"] = np.select(
        [signal["risk_on_votes"] == 0, signal["risk_on_votes"] == len(member_dirs)],
        ["unanimous_off", "unanimous_on"],
        default="disagreement",
    )

    returns = pd.read_csv(
        SOURCE / "daily_returns.csv", index_col=0, parse_dates=True
    )["net_return"]
    for horizon in (5, 21):
        signal[f"forward_{horizon}d"] = forward_compound(returns, horizon).reindex(
            scheduled
        )
    signal.to_csv(DESTINATION / "scheduled_signals.csv")

    state_rows = [summarize(group, str(state)) for state, group in signal.groupby("state")]
    state_summary = pd.DataFrame(state_rows).set_index("state")
    state_summary.to_csv(DESTINATION / "state_forward_returns.csv")

    on = signal.loc[signal["state"] == "unanimous_on"].copy()
    on["dispersion_quartile"] = pd.qcut(
        on["growth_exposure_range"],
        4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    )
    dispersion_rows = [
        summarize(group, str(quartile))
        for quartile, group in on.groupby("dispersion_quartile", observed=True)
    ]
    dispersion_summary = pd.DataFrame(dispersion_rows).set_index("state")
    dispersion_summary.to_csv(DESTINATION / "risk_on_dispersion_forward_returns.csv")

    print("State-conditioned forward returns:")
    print(state_summary.round(6).to_string())
    print("\nUnanimous risk-on dispersion quartiles:")
    print(dispersion_summary.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
