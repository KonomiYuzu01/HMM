from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output")
DESTINATION = OUTPUT / "dual_reentry_candidate_validation"
SCENARIOS = {
    "normal_2015_2025": (
        "paper_core_growth_gold20_daily_risk_netted_ensemble",
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble",
        "2015",
        "2025",
    ),
    "proxy_2006_2026": (
        "past_20y_drawdown_backtest_netted",
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble_20y_proxy",
        "2006",
        "2026",
    ),
}
CYCLE_STATES = {
    -1: "insufficient",
    0: "bull",
    1: "correction",
    2: "bear",
    3: "rebound",
}
VOLATILITY_STATES = {
    -1: "insufficient",
    0: "low",
    1: "medium",
    2: "high",
}


def frame(directory: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT / directory / filename,
        index_col=0,
        parse_dates=True,
    )


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, (
        production_directory,
        candidate_directory,
        start,
        end,
    ) in SCENARIOS.items():
        production = frame(
            production_directory,
            "daily_returns.csv",
        )["net_return"].loc[start:end]
        candidate = frame(
            candidate_directory,
            "daily_returns.csv",
        )["net_return"].loc[start:end]
        diagnostics = frame(
            f"{candidate_directory}/members/seed_7",
            "daily_returns.csv",
        ).loc[start:end]
        production_weights = frame(
            production_directory,
            "weights.csv",
        ).loc[start:end]
        candidate_weights = frame(
            candidate_directory,
            "weights.csv",
        ).loc[start:end]
        dates = (
            production.index.intersection(candidate.index)
            .intersection(diagnostics.index)
            .intersection(production_weights.index)
            .intersection(candidate_weights.index)
        )
        sample = pd.DataFrame(
            {
                "production": production.reindex(dates),
                "candidate": candidate.reindex(dates),
                "cycle_code": diagnostics.reindex(dates)[
                    "turning_point_state"
                ].astype(int),
                "volatility_code": diagnostics.reindex(dates)[
                    "turning_point_volatility_state"
                ].astype(int),
                "zero_entry_active": diagnostics.reindex(dates)[
                    "turning_point_zero_entry_active"
                ].astype(int),
                "growth_exposure_delta": (
                    candidate_weights.reindex(dates)[["QQQ", "SEMIS"]].sum(
                        axis=1
                    )
                    - production_weights.reindex(dates)[
                        ["QQQ", "SEMIS"]
                    ].sum(axis=1)
                ),
            }
        ).dropna()
        sample["cycle_state"] = sample["cycle_code"].map(CYCLE_STATES)
        sample["volatility_state"] = sample["volatility_code"].map(
            VOLATILITY_STATES
        )
        sample["regime"] = (
            sample["cycle_state"] + "__" + sample["volatility_state"]
        )
        sample["relative_log_return"] = np.log1p(
            sample["candidate"]
        ) - np.log1p(sample["production"])
        regime_change = sample["regime"].ne(
            sample["regime"].shift()
        )
        sample["regime_episode"] = regime_change.cumsum()
        total_relative_log = float(sample["relative_log_return"].sum())
        for regime, group in sample.groupby("regime", sort=True):
            cycle_state, volatility_state = regime.split("__")
            contribution = float(group["relative_log_return"].sum())
            rows.append(
                {
                    "scenario": scenario,
                    "cycle_state": cycle_state,
                    "volatility_state": volatility_state,
                    "sessions": len(group),
                    "occupancy": len(group) / len(sample),
                    "episodes": int(group["regime_episode"].nunique()),
                    "zero_entry_active_sessions": int(
                        group["zero_entry_active"].sum()
                    ),
                    "mean_growth_exposure_delta": float(
                        group["growth_exposure_delta"].mean()
                    ),
                    "production_compound_return": float(
                        (1.0 + group["production"]).prod() - 1.0
                    ),
                    "candidate_compound_return": float(
                        (1.0 + group["candidate"]).prod() - 1.0
                    ),
                    "relative_log_contribution": contribution,
                    "share_of_total_relative_log": (
                        contribution / total_relative_log
                        if abs(total_relative_log) > 1e-12
                        else float("nan")
                    ),
                }
            )
    attribution = pd.DataFrame(rows).set_index(
        ["scenario", "cycle_state", "volatility_state"]
    )
    attribution.to_csv(DESTINATION / "regime_attribution.csv")
    print(
        attribution[
            [
                "sessions",
                "occupancy",
                "episodes",
                "zero_entry_active_sessions",
                "mean_growth_exposure_delta",
                "relative_log_contribution",
                "share_of_total_relative_log",
            ]
        ].round(6).to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
