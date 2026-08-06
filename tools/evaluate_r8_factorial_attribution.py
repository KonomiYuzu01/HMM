from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output")
DESTINATION = (
    OUTPUT
    / "regime_strategy_research_2026-07-25"
    / "r8_factorial_attribution"
)
NORMAL = {
    "production": "paper_core_growth_gold20_daily_risk_netted_ensemble",
    "dual": (
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble"
    ),
    "floor45": (
        "paper_core_growth_gold20_dual_reentry_floor45_inverse_momentum_"
        "netted_ensemble"
    ),
    "r8": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
        "inverse_momentum_netted_ensemble"
    ),
}
PROXY = {
    "production": "past_20y_drawdown_backtest_netted",
    "dual": (
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble_20y_proxy"
    ),
    "floor45": (
        "paper_core_growth_gold20_dual_reentry_floor45_inverse_momentum_"
        "netted_ensemble_20y_proxy"
    ),
    "r8": (
        "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
        "inverse_momentum_netted_ensemble_20y_proxy"
    ),
}
LAYERS = (
    ("dual_reentry", "production", "dual"),
    ("floor45", "dual", "floor45"),
    ("constructive_bridge", "floor45", "r8"),
)


def load_returns(directory: str) -> pd.Series:
    return pd.read_csv(
        OUTPUT / directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )["net_return"]


def attribution(
    directories: dict[str, str],
    scenario: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = {
        name: load_returns(directory).loc[start:end]
        for name, directory in directories.items()
    }
    aligned = pd.DataFrame(returns).dropna()
    log_returns = np.log1p(aligned)
    annual_rows: list[pd.Series] = []
    summary_rows: list[dict[str, object]] = []
    years = (aligned.index[-1] - aligned.index[0]).days / 365.25
    for layer, baseline, candidate in LAYERS:
        relative = log_returns[candidate] - log_returns[baseline]
        annual = relative.groupby(relative.index.year).sum()
        annual_rows.append(np.expm1(annual).rename(layer))
        summary_rows.append(
            {
                "scenario": scenario,
                "layer": layer,
                "annualized_relative_log_return": float(
                    relative.sum() / years
                ),
                "positive_year_share": float(annual.gt(0.0).mean()),
                "worst_year": int(annual.idxmin()),
                "worst_year_relative_return": float(
                    np.expm1(annual.min())
                ),
                "best_year": int(annual.idxmax()),
                "best_year_relative_return": float(
                    np.expm1(annual.max())
                ),
            }
        )
    annual_frame = pd.DataFrame(annual_rows).T
    annual_frame.index.name = "year"
    annual_frame["scenario"] = scenario
    summary = pd.DataFrame(summary_rows).set_index(["scenario", "layer"])
    return annual_frame, summary


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    normal_annual, normal_summary = attribution(
        NORMAL,
        "normal_2015_2025",
        "2015-01-01",
        "2025-12-31",
    )
    proxy_annual, proxy_summary = attribution(
        PROXY,
        "proxy_2006_2025",
        "2006-08-01",
        "2025-12-31",
    )
    annual = pd.concat([normal_annual, proxy_annual])
    summary = pd.concat([normal_summary, proxy_summary])
    annual.to_csv(DESTINATION / "annual_layer_attribution.csv")
    summary.to_csv(DESTINATION / "layer_summary.csv")
    print("Layer summary:")
    print(summary.round(6).to_string())
    print("\nSelected years:")
    print(
        proxy_annual.loc[
            proxy_annual.index.intersection([2008, 2011, 2019, 2022, 2025])
        ].round(6).to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
