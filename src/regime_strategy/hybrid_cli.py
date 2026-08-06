from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .backtest import RegimeBacktester
from .data import load_prices
from .hybrid import (
    TrendSleeveBacktester,
    combine_sleeves,
    write_hybrid_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HMM/trend model ensemble")
    parser.add_argument("--config", default="config/paper_hmm_trend_hybrid.yaml")
    parser.add_argument("--cost-bps", type=float)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    hybrid_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hmm_path = Path(hybrid_config["hmm_config"])
    hmm_config = yaml.safe_load(hmm_path.read_text(encoding="utf-8"))
    trend_config = dict(hybrid_config["trend"])
    outer_config = dict(hybrid_config["hybrid"])
    if bool(hybrid_config.get("disable_drawdown_overlay", False)):
        hmm_config["portfolio"]["drawdown_overlay"] = []
        trend_config["drawdown_overlay"] = []
    if args.cost_bps is not None:
        hmm_config["portfolio"]["cost_bps_per_dollar_traded"] = args.cost_bps
        trend_config["cost_bps_per_dollar_traded"] = args.cost_bps
        outer_config["cost_bps_per_dollar_traded"] = args.cost_bps
    output_dir = args.output_dir or str(hybrid_config["output_dir"])

    prices = load_prices(hmm_config["data"], refresh=False)
    hmm = RegimeBacktester(hmm_config).run(prices)
    trend = TrendSleeveBacktester(trend_config).run(prices)
    result = combine_sleeves(
        hmm,
        trend,
        float(outer_config["hmm_share"]),
        int(outer_config["rebalance_every_days"]),
        float(outer_config["no_trade_turnover"]),
        float(outer_config["cost_bps_per_dollar_traded"]),
        prices.index
        if bool(outer_config.get("calendar_anchored_schedule", False))
        else None,
    )
    metrics = write_hybrid_report(result, output_dir)
    print(metrics.round(4).to_string())
    print(f"\nArtifacts: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
