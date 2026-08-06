from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


DATA_PATH = Path("data/prices_high_cagr.csv")
OUTPUT_DIR = Path("output/tactical_momentum_validation")
ASSETS = ["SPX", "QQQ", "SEMIS", "BOND", "GOLD", "OIL", "USD", "CASH"]
GROWTH_ASSETS = ["QQQ", "SEMIS"]
DEFENSIVE_ASSETS = ["BOND", "GOLD", "CASH"]
FORMATION_DAYS = 252
VOLATILITY_DAYS = 60
TARGET_VOLATILITY = 0.15
REBALANCE_DAYS = 21
COST_BPS = 7.5
ANNUALIZATION = 252


@dataclass(frozen=True)
class StrategySpec:
    name: str
    rotate_growth: bool
    volatility_scale: bool


SPECS = (
    StrategySpec("ROTATION_VOL15", rotate_growth=True, volatility_scale=True),
    StrategySpec("ROTATION_UNSCALED", rotate_growth=True, volatility_scale=False),
    StrategySpec("GROWTH_EQUAL_VOL15", rotate_growth=False, volatility_scale=True),
)


def _target_weights(
    historical_returns: pd.DataFrame,
    spec: StrategySpec,
) -> tuple[np.ndarray, str, bool, float, float]:
    formation = historical_returns.iloc[-FORMATION_DAYS:]
    momentum = np.log1p(formation.clip(lower=-0.999999)).sum(axis=0)
    cash_momentum = float(momentum["CASH"])
    growth_excess = momentum[GROWTH_ASSETS] - cash_momentum
    risk_on = bool(float(growth_excess.max()) > 0.0)

    unit = np.zeros(len(ASSETS), dtype=float)
    if risk_on:
        if spec.rotate_growth:
            selected = str(growth_excess.idxmax())
            unit[ASSETS.index(selected)] = 1.0
        else:
            selected = "QQQ+SEMIS"
            for asset in GROWTH_ASSETS:
                unit[ASSETS.index(asset)] = 0.5
    else:
        selected = str(momentum[DEFENSIVE_ASSETS].idxmax())
        unit[ASSETS.index(selected)] = 1.0

    risky = unit.copy()
    risky[ASSETS.index("CASH")] = 0.0
    risky_sum = float(risky.sum())
    realized_volatility = 0.0
    exposure = 1.0
    if risky_sum > 0.0:
        recent = historical_returns.iloc[-VOLATILITY_DAYS:][ASSETS]
        covariance = recent.cov().to_numpy() * ANNUALIZATION
        realized_volatility = float(
            np.sqrt(max(unit @ covariance @ unit, 0.0))
        )
        if spec.volatility_scale:
            exposure = min(1.0, TARGET_VOLATILITY / max(realized_volatility, 1e-12))

    target = exposure * unit
    target[ASSETS.index("CASH")] += 1.0 - float(target.sum())
    return target, selected, risk_on, exposure, realized_volatility


def run_strategy(
    returns: pd.DataFrame,
    spec: StrategySpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_position = max(FORMATION_DAYS, int(returns.index.searchsorted("2010-01-04")))
    dates = returns.index[start_position:]
    weights = np.zeros(len(ASSETS), dtype=float)
    weights[ASSETS.index("CASH")] = 1.0
    equity = 1.0
    peak = 1.0
    daily_rows: list[dict[str, float]] = []
    weight_rows: list[np.ndarray] = []
    signal_rows: list[dict[str, object]] = []

    for date in dates:
        trading_cost = 0.0
        turnover = 0.0
        calendar_position = int(returns.index.get_loc(date))
        if calendar_position % REBALANCE_DAYS == 0:
            history = returns.loc[returns.index < date, ASSETS]
            target, selected, risk_on, exposure, realized_volatility = _target_weights(
                history, spec
            )
            traded_notional = float(np.abs(target - weights).sum())
            trading_cost = traded_notional * COST_BPS / 10_000.0
            turnover = 0.5 * traded_notional
            weights = target
            signal_rows.append(
                {
                    "date": date,
                    "selected": selected,
                    "risk_on": int(risk_on),
                    "exposure": exposure,
                    "realized_volatility": realized_volatility,
                }
            )

        day_returns = returns.loc[date, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ day_returns)
        net_return = gross_return - trading_cost
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        daily_rows.append(
            {
                "gross_return": gross_return,
                "cost": trading_cost,
                "net_return": net_return,
                "turnover": turnover,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
            }
        )
        weight_rows.append(weights.copy())

        gross_growth = 1.0 + gross_return
        if gross_growth > 1e-12:
            weights = weights * (1.0 + day_returns) / gross_growth
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()

    return (
        pd.DataFrame(daily_rows, index=dates),
        pd.DataFrame(weight_rows, index=dates, columns=ASSETS),
        pd.DataFrame(signal_rows).set_index("date"),
    )


def _period_metrics(series: pd.Series) -> dict[str, float]:
    return performance_metrics(series, ANNUALIZATION)


def main() -> None:
    prices = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date")
    prices = prices[ASSETS].dropna(how="any")
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strategy_returns: dict[str, pd.Series] = {}
    for spec in SPECS:
        daily, weights, signals = run_strategy(returns, spec)
        daily.to_csv(OUTPUT_DIR / f"{spec.name.lower()}_daily.csv")
        weights.to_csv(OUTPUT_DIR / f"{spec.name.lower()}_weights.csv")
        signals.to_csv(OUTPUT_DIR / f"{spec.name.lower()}_signals.csv")
        strategy_returns[spec.name] = daily["net_return"]

    common_index = next(iter(strategy_returns.values())).index
    benchmark_returns = {
        "SPX": returns.loc[common_index, "SPX"],
        "QQQ": returns.loc[common_index, "QQQ"],
        "SEMIS": returns.loc[common_index, "SEMIS"],
        "GROWTH_EQUAL": returns.loc[common_index, GROWTH_ASSETS].mean(axis=1),
    }
    all_returns = {**strategy_returns, **benchmark_returns}
    pd.DataFrame(all_returns).to_csv(OUTPUT_DIR / "returns.csv")

    periods = {
        "predevelopment_2010_2014": ("2010-01-01", "2014-12-31"),
        "development_2015_2021": ("2015-01-01", "2021-12-31"),
        "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
        "complete_2015_2025": ("2015-01-01", "2025-12-31"),
        "full_2010_present": ("2010-01-01", None),
    }
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        for name, series in all_returns.items():
            sample = series.loc[start:end]
            rows.append(
                {"period": period, "strategy": name, **_period_metrics(sample)}
            )
    metrics = pd.DataFrame(rows).set_index(["period", "strategy"])
    metrics.to_csv(OUTPUT_DIR / "metrics_by_period.csv")
    print(metrics.round(4).to_string())


if __name__ == "__main__":
    main()
