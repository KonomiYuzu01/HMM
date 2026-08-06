from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .backtest import RegimeBacktester
from .data import load_prices
from .report import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the causal Wasserstein-HMM allocation")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--refresh", action="store_true", help="Refresh Yahoo adjusted prices")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prices = load_prices(config["data"], refresh=args.refresh)
    backtester = RegimeBacktester(config)
    result = backtester.run(prices)
    metrics = write_report(
        result,
        config["backtest"]["output_dir"],
        int(config["backtest"]["annualization"]),
    )
    print(metrics.round(4).to_string())
    target, diagnostics = backtester.recommend_next(prices, result)
    signal = pd.DataFrame(
        {
            "target_weight": target,
            "as_of_close": prices.index[-1].date().isoformat(),
            "signal_for": pd.Timestamp(target.name).date().isoformat(),
            "action": str(diagnostics["action"]),
            "estimated_rebalance_date": str(diagnostics["estimated_rebalance_date"]),
        }
    )
    output_dir = Path(config["backtest"]["output_dir"])
    signal.to_csv(output_dir / "next_target_weights.csv", index_label="asset")
    pd.Series(diagnostics, name="value").to_csv(output_dir / "next_signal_diagnostics.csv")
    print(
        f"\nNext-session action: {diagnostics['action']} "
        f"(using close {prices.index[-1].date()}, "
        f"estimated rebalance {diagnostics['estimated_rebalance_date']})"
    )
    print("Target/hold weights:")
    print(target.sort_values(ascending=False).round(4).to_string())
    print(f"\nArtifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
