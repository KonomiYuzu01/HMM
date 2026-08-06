from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_r10_gde_tracking_uncertainty import (
    ADJUSTED_CLOSE,
    bootstrap_tracking_uncertainty,
    gde_implementation_residual,
)


OUTPUT = Path("output/r10_staged_rollout")
BASE = Path("output/r10_combined_tail_capital")
CANDIDATE = Path("output/r10_high_fraction_band_frontier")
ROLLOUT_SHARES = (0.25, 0.50, 1.00)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = {
        "normal_synthetic": {
            "baseline": BASE / "normal_synthetic_baseline_daily.csv",
            "candidate": (
                CANDIDATE
                / "normal_synthetic_fraction50_band200bp_daily.csv"
            ),
            "periods": {
                "development_2015_2021": (
                    "2015-01-01",
                    "2021-12-31",
                ),
                "holdout_2022_2025": (
                    "2022-01-01",
                    "2025-12-31",
                ),
                "recent_2026": ("2026-01-01", "2026-12-31"),
                "complete_2015_2026": (
                    "2015-01-01",
                    "2026-12-31",
                ),
            },
        },
        "proxy_synthetic": {
            "baseline": BASE / "proxy_synthetic_baseline_daily.csv",
            "candidate": (
                CANDIDATE
                / "proxy_synthetic_fraction50_band200bp_daily.csv"
            ),
            "periods": {
                "early_2006_2014": ("2006-08-01", "2014-12-31"),
                "late_2015_2026": ("2015-01-01", "2026-12-31"),
                "complete_2006_2026": (
                    "2006-08-01",
                    "2026-12-31",
                ),
            },
        },
        "normal_live": {
            "baseline": BASE / "normal_live_baseline_daily.csv",
            "candidate": (
                CANDIDATE
                / "normal_live_fraction50_band200bp_daily.csv"
            ),
            "periods": {
                "live_2022_2026": ("2022-03-17", "2026-12-31"),
                "live_2022": ("2022-03-17", "2022-12-31"),
                "live_2023_2024": ("2023-01-01", "2024-12-31"),
                "live_2025_2026": ("2025-01-01", "2026-12-31"),
            },
        },
    }
    rows: list[dict[str, float | str]] = []
    annual_rows: list[dict[str, float | int | str]] = []
    proxy_paths: dict[float, tuple[pd.Series, pd.DataFrame]] = {}
    for sample, settings in samples.items():
        baseline_frame = pd.read_csv(
            settings["baseline"],
            index_col=0,
            parse_dates=True,
        )
        candidate_frame = pd.read_csv(
            settings["candidate"],
            index_col=0,
            parse_dates=True,
        )
        common = baseline_frame.index.intersection(candidate_frame.index)
        baseline = baseline_frame.loc[common, "net_return"]
        candidate = candidate_frame.loc[common, "net_return"]
        for share in ROLLOUT_SHARES:
            staged = (1.0 - share) * baseline + share * candidate
            staged_frame = candidate_frame.loc[common].copy()
            staged_frame["net_return"] = staged
            staged_frame["gde_target"] *= share
            label = int(round(share * 100))
            staged_frame.to_csv(
                OUTPUT / f"{sample}_share{label}_daily.csv",
                index_label="date",
            )
            for period, (start, end) in settings["periods"].items():
                period_baseline = baseline.loc[start:end]
                period_staged = staged.loc[start:end]
                base_metrics = performance_metrics(period_baseline)
                staged_metrics = performance_metrics(period_staged)
                rows.append(
                    {
                        "sample": sample,
                        "rollout_share": share,
                        "period": period,
                        "baseline_cagr": base_metrics["cagr"],
                        "staged_cagr": staged_metrics["cagr"],
                        "cagr_delta": (
                            staged_metrics["cagr"]
                            - base_metrics["cagr"]
                        ),
                        "baseline_sharpe": base_metrics["sharpe"],
                        "staged_sharpe": staged_metrics["sharpe"],
                        "sharpe_delta": (
                            staged_metrics["sharpe"]
                            - base_metrics["sharpe"]
                        ),
                        "staged_max_drawdown": staged_metrics[
                            "max_drawdown"
                        ],
                    }
                )
            for year, year_returns in staged.groupby(staged.index.year):
                base_year = baseline.reindex(year_returns.index)
                annual_rows.append(
                    {
                        "sample": sample,
                        "rollout_share": share,
                        "year": int(year),
                        "r9_return": float(
                            (1.0 + base_year).prod() - 1.0
                        ),
                        "staged_return": float(
                            (1.0 + year_returns).prod() - 1.0
                        ),
                    }
                )
            if sample == "proxy_synthetic":
                proxy_paths[share] = (baseline, staged_frame)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(
        OUTPUT / "annual_returns.csv",
        index=False,
    )
    prices = pd.read_csv(
        ADJUSTED_CLOSE,
        index_col=0,
        parse_dates=True,
    )
    residual = gde_implementation_residual(prices)
    tracking_rows = []
    for share, (baseline, staged_frame) in proxy_paths.items():
        tracking_rows.append(
            {
                "rollout_share": share,
                **bootstrap_tracking_uncertainty(
                    baseline,
                    staged_frame,
                    residual,
                ),
            }
        )
    tracking = pd.DataFrame(tracking_rows)
    tracking.to_csv(OUTPUT / "tracking_by_rollout_share.csv", index=False)
    launch_gde_trade = 0.016592
    launch_gld_trade = 0.016592
    estimated_launch_cost = (
        launch_gde_trade * 40.0 / 10_000.0
        + launch_gld_trade * 7.5 / 10_000.0
    )
    pd.Series(
        {
            "launch_gde_account_weight_trade": launch_gde_trade,
            "launch_gld_account_weight_trade": launch_gld_trade,
            "estimated_launch_cost_account_rate": estimated_launch_cost,
            "estimated_launch_cost_bps": estimated_launch_cost * 10_000,
        },
        name="value",
    ).to_csv(OUTPUT / "launch_cost.csv")
    print("Metrics:")
    print(metrics.round(6).to_string(index=False))
    print("\nTracking:")
    print(tracking.round(6).to_string(index=False))
    print(
        "\nEstimated launch cost: "
        f"{estimated_launch_cost * 10_000:.3f} bps of managed capital"
    )
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
