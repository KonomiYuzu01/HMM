from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "recovery_phase_gate_validation"
DEVELOPMENT = slice("2015-01-01", "2021-12-31")
STRATEGIES = {
    "step2": "paper_core_growth_gold20_daily_risk_netted_ensemble",
    "floor40": (
        "paper_core_growth_gold20_floor40_daily_risk_netted_ensemble"
    ),
    "rebound20": (
        "paper_core_growth_gold20_recovery_rebound20_netted_ensemble"
    ),
    "positive20": (
        "paper_core_growth_gold20_recovery_positive20_netted_ensemble"
    ),
    "positive20_guarded": (
        "paper_core_growth_gold20_recovery_positive20_"
        "guarded_netted_ensemble"
    ),
    "rebound63": (
        "paper_core_growth_gold20_recovery_rebound63_netted_ensemble"
    ),
    "positive63": (
        "paper_core_growth_gold20_recovery_positive63_netted_ensemble"
    ),
    "correction_veto20_guarded": (
        "paper_core_growth_gold20_correction_veto20_"
        "guarded_netted_ensemble"
    ),
    "correction_veto63_guarded": (
        "paper_core_growth_gold20_correction_veto63_"
        "guarded_netted_ensemble"
    ),
    "slow_negative_guarded": (
        "paper_core_growth_gold20_slow_negative_guarded_netted_ensemble"
    ),
}


def load(directory: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT / directory / filename,
        index_col=0,
        parse_dates=True,
    )


def capture_ratio(
    strategy: pd.Series,
    benchmark: pd.Series,
    positive: bool,
) -> float:
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    mask = aligned["benchmark"] > 0.0 if positive else aligned["benchmark"] < 0.0
    return float(
        aligned.loc[mask, "strategy"].mean()
        / aligned.loc[mask, "benchmark"].mean()
    )


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        "data/prices_recovery_quality.csv",
        index_col=0,
        parse_dates=True,
    )
    growth = prices[["QQQ", "SEMIS"]].pct_change(
        fill_method=None
    ).mean(axis=1).loc[DEVELOPMENT]
    rows: list[dict[str, float | str]] = []
    for name, directory in STRATEGIES.items():
        daily = load(directory, "daily_returns.csv").loc[DEVELOPMENT]
        weights = load(directory, "weights.csv").reindex(daily.index)
        metrics = performance_metrics(daily["net_return"])
        up_capture = capture_ratio(
            daily["net_return"], growth.reindex(daily.index), True
        )
        down_capture = capture_ratio(
            daily["net_return"], growth.reindex(daily.index), False
        )
        rows.append(
            {
                "strategy": name,
                "cagr": float(metrics["cagr"]),
                "annual_volatility": float(metrics["annual_volatility"]),
                "sharpe": float(metrics["sharpe"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "annualized_cost_drag": float(
                    daily["cost"].mean() * 252.0
                ),
                "average_growth_exposure": float(
                    weights[["QQQ", "SEMIS"]].sum(axis=1).mean()
                ),
                "up_capture": up_capture,
                "down_capture": down_capture,
                "capture_spread": up_capture - down_capture,
            }
        )
    result = pd.DataFrame(rows).set_index("strategy")
    baseline = result.loc["step2"]
    for metric in (
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "annualized_cost_drag",
        "average_growth_exposure",
        "up_capture",
        "down_capture",
        "capture_spread",
    ):
        result[f"{metric}_delta_vs_step2"] = (
            result[metric] - float(baseline[metric])
        )
    result["development_cagr_pass"] = (
        result["cagr_delta_vs_step2"] >= 0.003
    ).astype(int)
    result["capture_spread_pass"] = (
        result["capture_spread_delta_vs_step2"] > 0.0
    ).astype(int)
    result["drawdown_pass"] = (
        result["max_drawdown"] >= -0.15
    ).astype(int)
    result["cost_pass"] = (
        result["annualized_cost_drag_delta_vs_step2"] <= 0.002
    ).astype(int)
    result["eligible"] = result[
        [
            "development_cagr_pass",
            "capture_spread_pass",
            "drawdown_pass",
            "cost_pass",
        ]
    ].all(axis=1).astype(int)
    initial_candidates = result.loc[
        ["rebound20", "positive20", "rebound63", "positive63"]
    ]
    initial_eligible = initial_candidates[
        initial_candidates["eligible"].eq(1)
    ]
    result["initial_selected"] = 0
    if not initial_eligible.empty:
        initial_selected = str(initial_eligible["cagr"].idxmax())
        result.loc[initial_selected, "initial_selected"] = 1
    stage2_candidates = result.loc[
        [
            "correction_veto20_guarded",
            "correction_veto63_guarded",
            "slow_negative_guarded",
        ]
    ]
    stage2_eligible = stage2_candidates[
        stage2_candidates["eligible"].eq(1)
    ]
    result["stage2_selected"] = 0
    if not stage2_eligible.empty:
        stage2_selected = str(stage2_eligible["cagr"].idxmax())
        result.loc[stage2_selected, "stage2_selected"] = 1
    result.to_csv(DESTINATION / "development_screen.csv")
    print(result.round(6).to_string())


if __name__ == "__main__":
    main()
