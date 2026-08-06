from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract completed and ongoing drawdown episodes from daily returns."
    )
    parser.add_argument("--daily", required=True, help="CSV containing net_return.")
    parser.add_argument(
        "--benchmarks",
        help="Optional adjusted-price CSV with benchmark columns.",
    )
    parser.add_argument("--threshold", type=float, default=-0.05)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def drawdown_episodes(
    returns: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    clean = returns.dropna().astype(float)
    equity = (1.0 + clean).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0

    rows: list[dict[str, object]] = []
    in_episode = False
    peak_date: pd.Timestamp | None = None
    trough_date: pd.Timestamp | None = None
    trough_drawdown = 0.0

    for date, value in drawdown.items():
        if not in_episode and value < 0.0:
            in_episode = True
            prior = equity.loc[:date].iloc[:-1]
            peak_date = prior.idxmax() if not prior.empty else date
            trough_date = date
            trough_drawdown = float(value)
        elif in_episode and value < trough_drawdown:
            trough_date = date
            trough_drawdown = float(value)

        if in_episode and value >= -1.0e-12:
            if trough_drawdown <= threshold:
                assert peak_date is not None and trough_date is not None
                rows.append(
                    {
                        "peak_date": peak_date,
                        "trough_date": trough_date,
                        "recovery_date": date,
                        "max_drawdown": trough_drawdown,
                        "days_peak_to_trough": int(
                            clean.index.get_loc(trough_date)
                            - clean.index.get_loc(peak_date)
                        ),
                        "days_to_recovery": int(
                            clean.index.get_loc(date)
                            - clean.index.get_loc(peak_date)
                        ),
                        "ongoing": False,
                    }
                )
            in_episode = False
            peak_date = None
            trough_date = None
            trough_drawdown = 0.0

    if in_episode and trough_drawdown <= threshold:
        assert peak_date is not None and trough_date is not None
        rows.append(
            {
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_date": pd.NaT,
                "max_drawdown": trough_drawdown,
                "days_peak_to_trough": int(
                    clean.index.get_loc(trough_date) - clean.index.get_loc(peak_date)
                ),
                "days_to_recovery": pd.NA,
                "ongoing": True,
            }
        )
    return pd.DataFrame(rows)


def benchmark_episode_returns(
    episodes: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    result = episodes.copy()
    for asset in prices:
        values: list[float] = []
        for row in result.itertuples(index=False):
            window = prices.loc[row.peak_date : row.trough_date, asset].dropna()
            values.append(
                float(window.iloc[-1] / window.iloc[0] - 1.0)
                if len(window) >= 2
                else float("nan")
            )
        result[f"{asset}_peak_to_trough"] = values
    return result


def main() -> None:
    args = parse_args()
    daily = pd.read_csv(args.daily, index_col=0, parse_dates=True)
    if "net_return" not in daily:
        raise ValueError("Daily CSV must contain a net_return column")
    episodes = drawdown_episodes(daily["net_return"], float(args.threshold))
    if args.benchmarks:
        prices = pd.read_csv(args.benchmarks, index_col=0, parse_dates=True)
        episodes = benchmark_episode_returns(episodes, prices)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(output, index=False)
    print(episodes.to_string(index=False))


if __name__ == "__main__":
    main()
