from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_open_execution import simulate_open_execution
from tools.evaluate_r10_gde_capital_efficiency import (
    load_adjusted_open_close,
    paired_block_bootstrap,
)


OUTPUT = Path("output/r10_full_no_trade_threshold")
SAMPLES = {
    "normal": {
        "open_close": Path("data/adjusted_open_close_2011_present.csv"),
        "start_date": "2015-01-01",
        "directories": {
            0.01: "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble",
            0.02: "experiment_r10_r9_notrade2",
            0.03: "experiment_r10_r9_notrade3",
            0.05: "experiment_r10_r9_notrade5",
        },
        "periods": {
            "development_2015_2021": ("2015-01-01", "2021-12-31"),
            "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
            "recent_2026": ("2026-01-01", "2026-12-31"),
            "complete_2015_2026": ("2015-01-01", "2026-12-31"),
        },
    },
    "proxy": {
        "open_close": Path("data/adjusted_open_close_20y_proxy.csv"),
        "start_date": "2006-08-01",
        "directories": {
            0.01: "experiment_r9_broad50_stage35_d10_20y_proxy",
            0.02: "experiment_r10_r9_notrade2_20y_proxy",
            0.03: "experiment_r10_r9_notrade3_20y_proxy",
            0.05: "experiment_r10_r9_notrade5_20y_proxy",
        },
        "periods": {
            "early_2006_2014": ("2006-08-01", "2014-12-31"),
            "late_2015_2026": ("2015-01-01", "2026-12-31"),
            "complete_2006_2026": ("2006-08-01", "2026-12-31"),
        },
    },
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    for sample, settings in SAMPLES.items():
        opens, closes = load_adjusted_open_close(settings["open_close"])
        start_date = str(settings["start_date"])
        directories = settings["directories"]
        periods = settings["periods"]
        assert isinstance(directories, dict)
        assert isinstance(periods, dict)
        for cost_bps in (7.5, 15.0, 30.0):
            frames = {
                threshold: simulate_open_execution(
                    directory,
                    opens,
                    closes,
                    start_date=start_date,
                    end_date=None,
                    cost_bps=cost_bps,
                )
                for threshold, directory in directories.items()
            }
            baseline = frames[0.01]
            for threshold, frame in frames.items():
                common = baseline.index.intersection(frame.index)
                for period, (start, end) in periods.items():
                    selected = common[
                        (common >= start) & (common <= end)
                    ]
                    baseline_metrics = performance_metrics(
                        baseline.loc[selected, "net_return"]
                    )
                    metrics = performance_metrics(
                        frame.loc[selected, "net_return"]
                    )
                    rows.append(
                        {
                            "sample": sample,
                            "cost_bps": cost_bps,
                            "no_trade_turnover": threshold,
                            "period": period,
                            **{
                                f"baseline_{key}": value
                                for key, value in baseline_metrics.items()
                            },
                            **metrics,
                            "cagr_delta": (
                                metrics["cagr"]
                                - baseline_metrics["cagr"]
                            ),
                            "sharpe_delta": (
                                metrics["sharpe"]
                                - baseline_metrics["sharpe"]
                            ),
                            "max_drawdown_delta": (
                                metrics["max_drawdown"]
                                - baseline_metrics["max_drawdown"]
                            ),
                            "annualized_one_way_turnover": float(
                                frame.loc[
                                    selected,
                                    "open_turnover",
                                ].mean()
                                * 252.0
                            ),
                            "trade_days": int(
                                frame.loc[
                                    selected,
                                    "open_turnover",
                                ].gt(1e-14).sum()
                            ),
                        }
                    )
                if cost_bps == 7.5 and threshold != 0.01:
                    for block_days in (21, 63, 126):
                        bootstrap_rows.append(
                            {
                                "sample": sample,
                                "no_trade_turnover": threshold,
                                **paired_block_bootstrap(
                                    baseline.loc[common, "net_return"],
                                    frame.loc[common, "net_return"],
                                    block_days=block_days,
                                ),
                            }
                        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT / "paired_bootstrap.csv",
        index=False,
    )
    columns = [
        "sample",
        "cost_bps",
        "no_trade_turnover",
        "period",
        "cagr_delta",
        "sharpe_delta",
        "max_drawdown",
        "annualized_one_way_turnover",
        "trade_days",
    ]
    print(metrics[columns].round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
