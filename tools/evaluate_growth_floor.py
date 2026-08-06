from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_relative_momentum import circular_block_bootstrap
from regime_strategy.report import performance_metrics


OUTPUT_ROOT = Path("output")
DESTINATION = OUTPUT_ROOT / "growth_floor_validation"
BASELINE = "paper_core_growth"
CANDIDATE = "paper_core_growth_floor"
PERIODS = {
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "full_2015_present": ("2015-01-01", None),
}


def load_daily(strategy: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_ROOT / strategy / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    daily = {BASELINE: load_daily(BASELINE), CANDIDATE: load_daily(CANDIDATE)}
    rows: list[dict[str, float | str]] = []
    for strategy, frame in daily.items():
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(frame.loc[start:end, "net_return"]),
                }
            )
    metrics = pd.DataFrame(rows).set_index(["strategy", "period"])
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")

    baseline_complete = metrics.loc[(BASELINE, "complete_2015_2025")]
    candidate_complete = metrics.loc[(CANDIDATE, "complete_2015_2025")]
    baseline_development = metrics.loc[(BASELINE, "development_2015_2021")]
    candidate_development = metrics.loc[(CANDIDATE, "development_2015_2021")]
    baseline_holdout = metrics.loc[(BASELINE, "holdout_2022_2025")]
    candidate_holdout = metrics.loc[(CANDIDATE, "holdout_2022_2025")]
    acceptance = pd.DataFrame(
        [
            {
                "complete_cagr": candidate_complete["cagr"],
                "cagr_delta_vs_baseline": candidate_complete["cagr"]
                - baseline_complete["cagr"],
                "complete_max_drawdown": candidate_complete["max_drawdown"],
                "complete_sharpe_delta": candidate_complete["sharpe"]
                - baseline_complete["sharpe"],
                "development_cagr_delta": candidate_development["cagr"]
                - baseline_development["cagr"],
                "holdout_cagr_delta": candidate_holdout["cagr"]
                - baseline_holdout["cagr"],
                "cagr_plus_1pp_pass": int(
                    candidate_complete["cagr"] >= baseline_complete["cagr"] + 0.01
                ),
                "drawdown_18pct_pass": int(
                    candidate_complete["max_drawdown"] >= -0.18
                ),
                "development_no_degradation_pass": int(
                    candidate_development["cagr"] >= baseline_development["cagr"]
                ),
                "holdout_no_degradation_pass": int(
                    candidate_holdout["cagr"] >= baseline_holdout["cagr"]
                ),
            }
        ],
        index=[CANDIDATE],
    )
    pass_columns = [
        "cagr_plus_1pp_pass",
        "drawdown_18pct_pass",
        "development_no_degradation_pass",
        "holdout_no_degradation_pass",
    ]
    acceptance["overall_pass"] = acceptance[pass_columns].all(axis=1).astype(int)
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    sample = slice("2015-01-01", "2025-12-31")
    bootstrap = pd.DataFrame(
        [
            circular_block_bootstrap(
                daily[CANDIDATE].loc[sample, "net_return"],
                daily[BASELINE].loc[sample, "net_return"],
            )
        ],
        index=[CANDIDATE],
    )
    bootstrap.to_csv(DESTINATION / "bootstrap.csv")

    aligned = pd.concat(
        [
            daily[CANDIDATE].loc[sample, "net_return"].rename("candidate"),
            daily[BASELINE].loc[sample, "net_return"].rename("baseline"),
        ],
        axis=1,
    ).dropna()
    qqq_returns = pd.read_csv(
        "data/prices_high_cagr.csv", index_col=0, parse_dates=True
    )["QQQ"].pct_change(fill_method=None).reindex(aligned.index)
    relative_log = np.log1p(aligned["candidate"]) - np.log1p(aligned["baseline"])
    years = len(aligned) / 252
    attribution = pd.DataFrame(
        [
            {
                "annual_relative_log_return": float(relative_log.sum() / years),
                "qqq_up_day_contribution": float(
                    relative_log.loc[qqq_returns > 0.0].sum() / years
                ),
                "qqq_down_day_contribution": float(
                    relative_log.loc[qqq_returns <= 0.0].sum() / years
                ),
                "incremental_annual_cost": float(
                    (
                        daily[CANDIDATE].loc[sample, "cost"]
                        - daily[BASELINE].loc[sample, "cost"]
                    ).mean()
                    * 252
                ),
            }
        ],
        index=[CANDIDATE],
    )
    attribution.to_csv(DESTINATION / "up_down_attribution.csv")

    annual = pd.concat(
        {
            "baseline": daily[BASELINE].loc[sample, "net_return"],
            "candidate": daily[CANDIDATE].loc[sample, "net_return"],
        },
        axis=1,
    )
    annual = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    annual["candidate_minus_baseline"] = annual["candidate"] - annual["baseline"]
    annual.to_csv(DESTINATION / "annual_return_comparison.csv")

    regimes = pd.read_csv(
        OUTPUT_ROOT / CANDIDATE / "regimes.csv", index_col=0, parse_dates=True
    )
    active = regimes["risk_off_growth_floor_share"] > 0.0
    weights = pd.read_csv(
        OUTPUT_ROOT / CANDIDATE / "weights.csv", index_col=0, parse_dates=True
    ).reindex(regimes.index)
    pair_exposure = weights["QQQ"] + weights["SEMIS"]
    diagnostics = pd.DataFrame(
        [
            {
                "rebalance_observations": len(regimes),
                "growth_floor_active_rebalances": int(active.sum()),
                "growth_floor_active_share": float(active.mean()),
                "mean_pair_exposure_when_floor_active": float(
                    pair_exposure.loc[active].mean()
                ),
                "annual_wins_vs_baseline": int(
                    (annual["candidate_minus_baseline"] > 0.0).sum()
                ),
                "annual_periods": len(annual),
                "gross_cagr": performance_metrics(
                    daily[CANDIDATE].loc[sample, "gross_return"]
                )["cagr"],
                "net_cagr": candidate_complete["cagr"],
                "annualized_average_cost": float(
                    daily[CANDIDATE].loc[sample, "cost"].mean() * 252
                ),
            }
        ],
        index=[CANDIDATE],
    )
    diagnostics.to_csv(DESTINATION / "signal_diagnostics.csv")

    print(metrics.round(4).to_string())
    print("\nAcceptance:")
    print(acceptance.round(4).to_string())
    print("\nBootstrap:")
    print(bootstrap.round(4).to_string())
    print("\nAttribution:")
    print(attribution.round(4).to_string())
    print("\nSignal diagnostics:")
    print(diagnostics.round(4).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
