from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output")
DESTINATION = OUTPUT / "regime_strategy_research_2026-07-25" / "r8_release"
PRODUCTION = "paper_core_growth_gold20_daily_risk_netted_ensemble"
CANDIDATE = (
    "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
    "inverse_momentum_netted_ensemble"
)
TRANCHES = 4
MAX_TRANCHE_ONE_WAY_TURNOVER = 0.10
TRANSACTION_COST_BPS = 15.0


def load_returns(directory: str) -> pd.Series:
    frame = pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    return frame["net_return"].rename(directory)


def load_target(directory: str) -> pd.Series:
    frame = pd.read_csv(
        OUTPUT / directory / "next_target_weights.csv",
        index_col=0,
    )
    return frame["ensemble_current_sleeve_weight"].rename(directory)


def cumulative_candidate_shares(
    tranches: int,
    spacing_sessions: int,
) -> np.ndarray:
    if tranches < 1:
        raise ValueError("tranches must be positive")
    if spacing_sessions < 1:
        raise ValueError("spacing_sessions must be positive")
    horizon = 1 + (tranches - 1) * spacing_sessions
    shares = np.empty(horizon, dtype=float)
    for offset in range(horizon):
        completed = min(offset // spacing_sessions + 1, tranches)
        shares[offset] = completed / tranches
    return shares


def staged_target_plan(
    production: pd.Series,
    candidate: pd.Series,
    tranches: int = TRANCHES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = production.index.union(candidate.index)
    baseline = production.reindex(assets, fill_value=0.0)
    destination = candidate.reindex(assets, fill_value=0.0)
    target_rows: list[pd.Series] = []
    trade_rows: list[pd.Series] = []
    previous = baseline
    for tranche in range(1, tranches + 1):
        share = tranche / tranches
        target = baseline + share * (destination - baseline)
        trade = target - previous
        target_rows.append(target.rename(tranche))
        trade_rows.append(trade.rename(tranche))
        previous = target
    targets = pd.DataFrame(target_rows)
    trades = pd.DataFrame(trade_rows)
    targets.index.name = "tranche"
    trades.index.name = "tranche"
    return targets, trades


def activation_path_results(
    production: pd.Series,
    candidate: pd.Series,
    *,
    tranches: int = TRANCHES,
    spacing_sessions: int,
    start: str,
    end: str,
) -> pd.DataFrame:
    aligned = pd.concat([production, candidate], axis=1).dropna().loc[start:end]
    aligned.columns = ["production", "candidate"]
    shares = cumulative_candidate_shares(tranches, spacing_sessions)
    horizon = len(shares)
    rows: list[dict[str, object]] = []
    for position in range(0, len(aligned) - horizon + 1):
        window = aligned.iloc[position : position + horizon]
        staged = (
            (1.0 - shares) * window["production"].to_numpy()
            + shares * window["candidate"].to_numpy()
        )
        instant = window["candidate"].to_numpy()
        staged_growth = float(np.prod(1.0 + staged))
        instant_growth = float(np.prod(1.0 + instant))
        rows.append(
            {
                "activation_date": window.index[0],
                "completion_date": window.index[-1],
                "horizon_sessions": horizon,
                "spacing_sessions": spacing_sessions,
                "staged_return": staged_growth - 1.0,
                "instant_return": instant_growth - 1.0,
                "relative_transition_return": (
                    staged_growth / instant_growth - 1.0
                ),
            }
        )
    return pd.DataFrame(rows).set_index("activation_date")


def summarize_activation_paths(paths: pd.DataFrame, label: str) -> pd.Series:
    relative = paths["relative_transition_return"]
    active = relative[relative.abs() > 1e-12]
    worst_position = relative.idxmin()
    return pd.Series(
        {
            "schedule": label,
            "historical_activation_dates": len(relative),
            "active_activation_dates": len(active),
            "horizon_sessions": int(paths["horizon_sessions"].iloc[0]),
            "mean_relative_transition_return": float(relative.mean()),
            "p05_relative_transition_return": float(relative.quantile(0.05)),
            "median_relative_transition_return": float(relative.median()),
            "p95_relative_transition_return": float(relative.quantile(0.95)),
            "worst_relative_transition_return": float(relative.min()),
            "best_relative_transition_return": float(relative.max()),
            "staged_underperformance_share": float(
                relative.lt(-1e-12).mean()
            ),
            "active_mean_relative_transition_return": float(active.mean()),
            "active_p05_relative_transition_return": float(
                active.quantile(0.05)
            ),
            "active_underperformance_share": float(active.lt(0.0).mean()),
            "worst_activation_date": pd.Timestamp(
                worst_position
            ).date().isoformat(),
            "worst_completion_date": pd.Timestamp(
                paths.loc[worst_position, "completion_date"]
            ).date().isoformat(),
        }
    )


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    production_returns = load_returns(PRODUCTION)
    candidate_returns = load_returns(CANDIDATE)
    production_target = load_target(PRODUCTION)
    candidate_target = load_target(CANDIDATE)

    targets, trades = staged_target_plan(production_target, candidate_target)
    targets.to_csv(DESTINATION / "staged_target_weights.csv")
    trades.to_csv(DESTINATION / "staged_trades.csv")

    one_way_turnover = 0.5 * trades.abs().sum(axis=1)
    if float(one_way_turnover.max()) > MAX_TRANCHE_ONE_WAY_TURNOVER + 1e-12:
        raise RuntimeError("A migration tranche exceeds the 10% turnover cap")
    if not np.allclose(
        targets.iloc[-1].to_numpy(),
        candidate_target.reindex(targets.columns, fill_value=0.0).to_numpy(),
    ):
        raise RuntimeError("The staged migration does not reach the candidate")

    path_frames: list[pd.DataFrame] = []
    summaries: list[pd.Series] = []
    for label, spacing in (("four_consecutive_sessions", 1), ("four_weeks", 5)):
        paths = activation_path_results(
            production_returns,
            candidate_returns,
            spacing_sessions=spacing,
            start="2015-01-01",
            end="2025-12-31",
        )
        paths["schedule"] = label
        path_frames.append(paths)
        summaries.append(summarize_activation_paths(paths, label))
    path_frame = pd.concat(path_frames)
    path_frame.to_csv(DESTINATION / "historical_activation_paths.csv")
    summary = pd.DataFrame(summaries).set_index("schedule")
    summary.to_csv(DESTINATION / "historical_activation_summary.csv")

    total_one_way_turnover = float(one_way_turnover.sum())
    explicit_cost_fraction = (
        total_one_way_turnover * TRANSACTION_COST_BPS / 10_000.0
    )
    migration_summary = {
        "production": PRODUCTION,
        "candidate": CANDIDATE,
        "tranches": TRANCHES,
        "maximum_tranche_one_way_turnover": float(one_way_turnover.max()),
        "total_one_way_turnover": total_one_way_turnover,
        "assumed_transaction_cost_bps": TRANSACTION_COST_BPS,
        "estimated_total_explicit_cost_fraction": explicit_cost_fraction,
        "estimated_total_explicit_cost_nav_bps": (
            explicit_cost_fraction * 10_000.0
        ),
        "turnover_gate_cap": MAX_TRANCHE_ONE_WAY_TURNOVER,
        "turnover_gate_pass": bool(
            one_way_turnover.max() <= MAX_TRANCHE_ONE_WAY_TURNOVER
        ),
    }
    (DESTINATION / "migration_summary.json").write_text(
        json.dumps(migration_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Per-tranche one-way turnover:")
    print(one_way_turnover.to_string())
    print("\nHistorical activation summary:")
    print(summary.to_string())
    print("\nMigration summary:")
    print(pd.Series(migration_summary).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
