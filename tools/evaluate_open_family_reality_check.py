from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_open_execution import load_open_close, simulate_open_execution
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/open_family_reality_check")
SELECTED = "jump_aware"
STRATEGIES = {
    "production_baseline": "paper_core_growth_gold20_daily_risk_ensemble",
    SELECTED: "paper_core_growth_gold20_jump_aware_daily_risk_ensemble",
    "jump_notrade2": "paper_core_growth_gold20_jump_daily_risk_notrade2_ensemble",
    "jump_notrade3": "paper_core_growth_gold20_jump_daily_risk_notrade3_ensemble",
    "target_vol14": "paper_core_growth_gold20_daily_risk_vol14_ensemble",
    "target_vol16": "paper_core_growth_gold20_daily_risk_vol16_ensemble",
    "target_vol18": "paper_core_growth_gold20_daily_risk_vol18_ensemble",
    "target_vol22": "paper_core_growth_gold20_daily_risk_vol22_ensemble",
    "symmetric_recovery": "paper_core_growth_gold20_symmetric_daily_risk_ensemble",
    "independent_reallocation": "paper_core_growth_gold20_daily_reallocation_ensemble",
    "leverage_110": "paper_core_growth_gold20_lev110_daily_risk_ensemble",
    "dynamic_defensive": "paper_core_growth_dynamic_defensive_jump_daily_risk_ensemble",
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    opens, closes = load_open_close(False)
    spy = closes["SPX"].pct_change(fill_method=None).loc["2015":"2025"]
    relative = {}
    eligibility_rows = []
    for name, directory in STRATEGIES.items():
        frame = simulate_open_execution(directory, opens, closes, cost_bps=7.5)
        metrics = performance_metrics(frame["net_return"])
        eligible = metrics["max_drawdown"] >= -0.18
        eligibility_rows.append(
            {
                "strategy": name,
                "eligible": int(eligible),
                **metrics,
            }
        )
        if eligible:
            aligned = pd.concat(
                [frame["net_return"].rename("candidate"), spy.rename("SPY")],
                axis=1,
                join="inner",
            ).dropna()
            relative[name] = np.log1p(aligned["candidate"]) - np.log1p(
                aligned["SPY"]
            )
    eligibility = pd.DataFrame(eligibility_rows).set_index("strategy")
    eligibility.to_csv(DESTINATION / "eligibility.csv")

    matrix_frame = pd.DataFrame(relative).dropna()
    observed = matrix_frame.mean(axis=0).to_numpy(dtype=float) * 252.0
    centered = matrix_frame.to_numpy(dtype=float).copy()
    centered -= centered.mean(axis=0, keepdims=True)
    selected_index = matrix_frame.columns.get_loc(SELECTED)
    selected_observed = float(observed[selected_index])
    block_days = 21
    simulations = 10_000
    count = len(centered)
    blocks = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(20260723)
    maximum_statistics = np.empty(simulations)
    selected_statistics = np.empty(simulations)
    for sample in range(simulations):
        starts = rng.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        statistic = centered[indices].mean(axis=0) * 252.0
        maximum_statistics[sample] = statistic.max()
        selected_statistics[sample] = statistic[selected_index]

    ranking = pd.DataFrame(
        {"annualized_relative_log_return": observed},
        index=matrix_frame.columns,
    ).sort_values("annualized_relative_log_return", ascending=False)
    ranking.to_csv(DESTINATION / "ranking.csv")
    summary = pd.Series(
        {
            "selected_annualized_relative_log_return": selected_observed,
            "nominal_one_sided_p_value": float(
                (selected_statistics >= selected_observed).mean()
            ),
            "familywise_reality_check_p_value": float(
                (maximum_statistics >= selected_observed).mean()
            ),
            "candidate_family_size": int(matrix_frame.shape[1]),
            "selected_rank": int(ranking.index.get_loc(SELECTED) + 1),
            "drawdown_constraint": -0.18,
            "block_days": block_days,
            "bootstrap_samples": simulations,
        },
        name=SELECTED,
    )
    summary.to_csv(DESTINATION / "summary.csv")

    print("Eligibility:")
    print(
        eligibility[["eligible", "cagr", "sharpe", "max_drawdown"]]
        .round(6)
        .to_string()
    )
    print("\nRisk-feasible ranking versus SPY:")
    print(ranking.round(6).to_string())
    print("\nReality Check:")
    print(summary.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
