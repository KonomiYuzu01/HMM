from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = {
    "GFC_2007_2009": ("2007-10-09", "2009-03-09"),
    "US_downgrade_Euro_2011": ("2011-04-29", "2011-10-03"),
    "China_oil_2015_2016": ("2015-07-20", "2016-02-11"),
    "Volmageddon_Q4_2018": ("2018-01-26", "2018-12-24"),
    "semis_trade_war_2019": ("2019-04-24", "2019-06-17"),
    "COVID_2020": ("2020-02-19", "2020-03-23"),
    "growth_bear_2022": ("2021-11-19", "2022-10-14"),
    "semis_correction_2024": ("2024-07-10", "2024-08-07"),
    "tariff_shock_2025": ("2025-02-19", "2025-04-07"),
    "cross_asset_shock_2026": ("2026-01-29", "2026-03-30"),
    "recent_semis_2026": ("2026-06-03", "2026-07-23"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure strategy and benchmark behavior in fixed crisis windows."
    )
    parser.add_argument("--daily", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--members-root")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def path_statistics(returns: pd.Series) -> dict[str, object]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {
            "return": np.nan,
            "max_drawdown": np.nan,
            "trough_date": pd.NaT,
            "worst_day": np.nan,
            "worst_day_date": pd.NaT,
            "annual_volatility": np.nan,
        }
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "trough_date": drawdown.idxmin(),
        "worst_day": float(clean.min()),
        "worst_day_date": clean.idxmin(),
        "annual_volatility": float(clean.std(ddof=1) * np.sqrt(252.0)),
    }


def member_max_drawdowns(
    root: Path | None,
    start: str,
    end: str,
) -> dict[str, float]:
    if root is None:
        return {}
    result: dict[str, float] = {}
    for seed in (7, 42, 123):
        path = root / f"seed_{seed}" / "daily_returns.csv"
        if not path.exists():
            continue
        daily = pd.read_csv(path, index_col=0, parse_dates=True)
        stats = path_statistics(daily.loc[start:end, "net_return"])
        result[f"seed_{seed}_max_drawdown"] = float(stats["max_drawdown"])
    return result


def main() -> None:
    args = parse_args()
    daily = pd.read_csv(args.daily, index_col=0, parse_dates=True)
    weights = pd.read_csv(args.weights, index_col=0, parse_dates=True)
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    benchmark_returns = prices[["SPX", "QQQ", "SEMIS"]].pct_change(
        fill_method=None
    )
    members_root = Path(args.members_root) if args.members_root else None

    rows: list[dict[str, object]] = []
    for event, (start, configured_end) in WINDOWS.items():
        available_end = min(
            pd.Timestamp(configured_end),
            daily.index.max(),
            weights.index.max(),
            prices.index.max(),
        )
        if available_end < pd.Timestamp(start):
            continue
        end = available_end.date().isoformat()
        if daily.loc[start:end].empty or weights.loc[start:end].empty:
            continue
        strategy_stats = path_statistics(daily.loc[start:end, "net_return"])
        window_weights = weights.loc[start:end]
        row: dict[str, object] = {
            "event": event,
            "start": start,
            "end": end,
            "strategy_return": strategy_stats["return"],
            "strategy_max_drawdown": strategy_stats["max_drawdown"],
            "strategy_trough_date": strategy_stats["trough_date"],
            "strategy_worst_day": strategy_stats["worst_day"],
            "strategy_worst_day_date": strategy_stats["worst_day_date"],
            "strategy_annual_volatility": strategy_stats["annual_volatility"],
            "average_growth_weight": float(
                window_weights[["QQQ", "SEMIS"]].sum(axis=1).mean()
            ),
            "minimum_growth_weight": float(
                window_weights[["QQQ", "SEMIS"]].sum(axis=1).min()
            ),
            "average_QQQ_weight": float(window_weights["QQQ"].mean()),
            "average_SEMIS_weight": float(window_weights["SEMIS"].mean()),
            "average_GOLD_weight": float(window_weights["GOLD"].mean()),
            "average_BOND_weight": float(window_weights["BOND"].mean()),
            "average_USD_weight": float(window_weights["USD"].mean()),
            "average_CASH_weight": float(window_weights["CASH"].mean()),
            "maximum_VIX_HEDGE_weight": float(
                window_weights.get(
                    "VIX_HEDGE",
                    pd.Series(0.0, index=window_weights.index),
                ).max()
            ),
        }
        for benchmark in benchmark_returns:
            stats = path_statistics(benchmark_returns.loc[start:end, benchmark])
            row[f"{benchmark}_return"] = stats["return"]
            row[f"{benchmark}_max_drawdown"] = stats["max_drawdown"]
        row.update(member_max_drawdowns(members_root, start, end))
        rows.append(row)

    result = pd.DataFrame(rows).set_index("event")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output)
    display_columns = [
        "strategy_return",
        "strategy_max_drawdown",
        "SPX_max_drawdown",
        "QQQ_max_drawdown",
        "SEMIS_max_drawdown",
        "average_growth_weight",
        "average_GOLD_weight",
        "average_CASH_weight",
        "maximum_VIX_HEDGE_weight",
    ]
    print(result[display_columns].round(4).to_string())


if __name__ == "__main__":
    main()
