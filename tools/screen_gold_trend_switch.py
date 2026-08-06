from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from regime_strategy.portfolio import apply_daily_growth_risk_reduction


SOURCE = Path("output/paper_core_growth_gold20_daily_risk_ensemble")
DESTINATION = Path("output/gold_trend_switch_screen")
PRICE_CACHE = Path("data/prices_high_cagr.csv")
COST_BPS = 7.5
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_2025": ("2015-01-01", "2025-12-31"),
}


def path_turnover(
    targets: pd.DataFrame, asset_returns: pd.DataFrame
) -> pd.Series:
    current = targets.iloc[0].copy()
    rows: list[float] = []
    for date, target in targets.iterrows():
        rows.append(0.5 * float((target - current).abs().sum()))
        gross_return = float(target @ asset_returns.loc[date])
        current = target * (1.0 + asset_returns.loc[date]) / (1.0 + gross_return)
    return pd.Series(rows, index=targets.index, name="turnover")


def simulate_independent_tsmom(
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    growth_risk_target: float | None = None,
) -> pd.DataFrame:
    assets = ["QQQ", "SEMIS", "GOLD", "CASH"]
    returns = prices[assets].pct_change(fill_method=None)
    lagged_excess_trend = pd.DataFrame(
        {
            asset: np.log(prices[asset] / prices["CASH"]).diff(252).shift(1)
            for asset in ("QQQ", "SEMIS", "GOLD")
        }
    )
    lagged_volatility = (
        returns[["QQQ", "SEMIS", "GOLD"]]
        .ewm(com=60.0, adjust=False)
        .std(bias=False)
        .shift(1)
        * np.sqrt(261.0)
    )
    base_weights = {"QQQ": 0.40, "SEMIS": 0.40, "GOLD": 0.20}
    current = pd.Series(0.0, index=assets)
    current["CASH"] = 1.0
    rows: list[dict[str, float]] = []
    full_index = prices.index
    for date in dates:
        target = current.copy()
        if int(full_index.get_loc(date)) % 21 == 0:
            target[:] = 0.0
            for asset, base_weight in base_weights.items():
                trend = float(lagged_excess_trend.loc[date, asset])
                volatility = float(lagged_volatility.loc[date, asset])
                if trend > 0.0 and np.isfinite(volatility) and volatility > 0.0:
                    target[asset] = base_weight * min(1.0, 0.40 / volatility)
            target["CASH"] = 1.0 - float(target.drop("CASH").sum())
        if growth_risk_target is not None:
            target_array, _, _ = apply_daily_growth_risk_reduction(
                target.to_numpy(dtype=float),
                returns.loc[returns.index < date, assets].dropna(how="any"),
                assets,
                ["QQQ", "SEMIS"],
                assets.index("CASH"),
                20,
                60,
                growth_risk_target,
                252,
            )
            target = pd.Series(target_array, index=assets)
        turnover = 0.5 * float((target - current).abs().sum())
        trading_cost = 2.0 * turnover * COST_BPS / 10_000.0
        day_returns = returns.loc[date, assets].fillna(0.0)
        gross_return = float(target @ day_returns)
        net_return = gross_return - trading_cost
        rows.append(
            {
                "net_return": net_return,
                "gross_return": gross_return,
                "cost": trading_cost,
                "turnover": turnover,
                **{f"weight_{asset}": float(target[asset]) for asset in assets},
            }
        )
        current = target * (1.0 + day_returns) / (1.0 + gross_return)
    return pd.DataFrame(rows, index=dates)


def main() -> None:
    weights = pd.read_csv(
        SOURCE / "weights.csv", index_col=0, parse_dates=True
    ).astype(float)
    daily = pd.read_csv(
        SOURCE / "daily_returns.csv", index_col=0, parse_dates=True
    ).astype(float)
    prices = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True).astype(float)
    weights, daily = weights.align(daily, join="inner", axis=0)
    asset_returns = prices[weights.columns].pct_change(fill_method=None).reindex(
        weights.index
    ).fillna(0.0)

    excess_trend = (
        np.log(prices["GOLD"] / prices["CASH"])
        .diff(252)
        .shift(1)
        .reindex(weights.index)
    )
    risk_on = (weights["QQQ"] + weights["SEMIS"]) > 0.50
    candidate = weights.copy()
    trend_off = excess_trend.le(0.0).fillna(False)
    gold_to_qqq = trend_off & risk_on
    gold_to_cash = trend_off & ~risk_on
    candidate.loc[gold_to_qqq, "QQQ"] += candidate.loc[gold_to_qqq, "GOLD"]
    candidate.loc[gold_to_cash, "CASH"] += candidate.loc[gold_to_cash, "GOLD"]
    candidate.loc[trend_off, "GOLD"] = 0.0

    baseline_gross = (weights * asset_returns).sum(axis=1)
    candidate_gross = (candidate * asset_returns).sum(axis=1)
    baseline_path_turnover = path_turnover(weights, asset_returns)
    candidate_path_turnover = path_turnover(candidate, asset_returns)
    incremental_cost = (
        2.0
        * (candidate_path_turnover - baseline_path_turnover)
        * COST_BPS
        / 10_000.0
    )
    candidate_net = (
        daily["net_return"]
        + candidate_gross
        - baseline_gross
        - incremental_cost
    ).rename("candidate")
    independent_tsmom = simulate_independent_tsmom(prices, weights.index)
    independent_tsmom_risk20 = simulate_independent_tsmom(
        prices, weights.index, growth_risk_target=0.20
    )

    rows: list[dict[str, float | str]] = []
    for period, (start, end) in PERIODS.items():
        for name, returns in {
            "production": daily["net_return"],
            "gold_trend_switch": candidate_net,
            "independent_tsmom": independent_tsmom["net_return"],
            "independent_tsmom_risk20": independent_tsmom_risk20["net_return"],
        }.items():
            rows.append(
                {
                    "period": period,
                    "strategy": name,
                    **performance_metrics(returns.loc[start:end]),
                }
            )
    metrics = pd.DataFrame(rows).set_index(["period", "strategy"])
    diagnostics = pd.Series(
        {
            "fraction_gold_trend_off": float(trend_off.mean()),
            "fraction_gold_to_qqq": float(gold_to_qqq.mean()),
            "fraction_gold_to_cash": float(gold_to_cash.mean()),
            "annualized_incremental_turnover": float(
                (candidate_path_turnover - baseline_path_turnover).mean() * 252.0
            ),
            "annualized_incremental_cost": float(incremental_cost.mean() * 252.0),
            "tsmom_annualized_turnover": float(
                independent_tsmom["turnover"].mean() * 252.0
            ),
            "tsmom_annualized_cost": float(
                independent_tsmom["cost"].mean() * 252.0
            ),
            "tsmom_risk20_annualized_turnover": float(
                independent_tsmom_risk20["turnover"].mean() * 252.0
            ),
            "tsmom_risk20_annualized_cost": float(
                independent_tsmom_risk20["cost"].mean() * 252.0
            ),
        },
        name="value",
    )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DESTINATION / "metrics.csv")
    diagnostics.to_csv(DESTINATION / "diagnostics.csv")
    pd.DataFrame(
        {
            "production": daily["net_return"],
            "gold_trend_switch": candidate_net,
            "gold_excess_trend_252d": excess_trend,
            "gold_trend_off": trend_off.astype(int),
            "gold_to_qqq": gold_to_qqq.astype(int),
            "gold_to_cash": gold_to_cash.astype(int),
            "incremental_cost": incremental_cost,
        }
    ).to_csv(DESTINATION / "daily.csv")
    independent_tsmom.to_csv(DESTINATION / "independent_tsmom_daily.csv")
    independent_tsmom_risk20.to_csv(
        DESTINATION / "independent_tsmom_risk20_daily.csv"
    )
    print(metrics.round(6).to_string())
    print("\nDiagnostics:")
    print(diagnostics.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
