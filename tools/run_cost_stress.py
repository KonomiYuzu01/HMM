from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from regime_strategy.backtest import RegimeBacktester
from regime_strategy.data import load_prices
from regime_strategy.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cached-data transaction-cost stress")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cost-bps", required=True, type=float)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--max-gross-leverage", type=float)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["portfolio"]["cost_bps_per_dollar_traded"] = args.cost_bps
    if args.random_seed is not None:
        config["model"]["random_seed"] = args.random_seed
    if args.restarts is not None:
        if args.restarts < 1:
            parser.error("--restarts must be at least 1")
        config["model"]["restarts"] = args.restarts
    if args.max_gross_leverage is not None:
        if args.max_gross_leverage <= 0.0:
            parser.error("--max-gross-leverage must be positive")
        config["portfolio"]["max_gross_leverage"] = args.max_gross_leverage
    config["backtest"]["output_dir"] = args.output_dir
    prices = load_prices(config["data"], refresh=False)
    result = RegimeBacktester(config).run(prices)
    metrics = write_report(
        result,
        args.output_dir,
        int(config["backtest"]["annualization"]),
    )
    print(metrics.round(4).to_string())


if __name__ == "__main__":
    main()
