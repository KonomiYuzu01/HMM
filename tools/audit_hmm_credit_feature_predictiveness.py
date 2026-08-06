from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output/hmm_credit_feature_predictiveness")
HORIZONS = (21, 63)
FORWARD_DAYS = 21
PERIODS = {
    "early_2008_2014": ("2008-01-01", "2014-12-31"),
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
}


def causal_rolling_percentile(series: pd.Series) -> pd.Series:
    return series.rolling(756, min_periods=252).rank(pct=True)


def forward_path_labels(log_returns: pd.Series, days: int = FORWARD_DAYS) -> pd.DataFrame:
    values = log_returns.to_numpy(dtype=float)
    total = np.full(len(values), np.nan)
    worst = np.full(len(values), np.nan)
    for index in range(len(values) - days):
        path = np.cumsum(values[index + 1 : index + days + 1])
        total[index] = float(path[-1])
        worst[index] = float(path.min())
    return pd.DataFrame(
        {"forward_log_return": total, "forward_worst_log_return": worst},
        index=log_returns.index,
    )


def group_differences(frame: pd.DataFrame, low_column: str) -> dict[str, float | int]:
    low = frame.loc[frame[low_column]]
    other = frame.loc[~frame[low_column]]
    return {
        "observations": len(frame),
        "low_credit_observations": len(low),
        "other_observations": len(other),
        "forward_return_difference": float(
            low["forward_log_return"].mean() - other["forward_log_return"].mean()
        ),
        "forward_worst_return_difference": float(
            low["forward_worst_log_return"].mean()
            - other["forward_worst_log_return"].mean()
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    credit = pd.read_csv(
        "data/sector_hmm_shadow_prices.csv", index_col="date", parse_dates=True
    )[["HYG", "IEF"]]
    market = pd.read_csv(
        "data/prices_recovery_quality.csv", index_col="date", parse_dates=True
    )[["QQQ", "SEMIS", "VIX", "VIX3M"]]
    prices = credit.join(market, how="inner").dropna()
    growth_log_return = (
        np.log(prices[["QQQ", "SEMIS"]]).diff().mean(axis=1)
    )
    labels = forward_path_labels(growth_log_return)
    vix_backwardation = prices["VIX"] / prices["VIX3M"] > 1.0
    volatility = growth_log_return.rolling(21).std() * np.sqrt(252.0)
    volatility_slow = growth_log_return.rolling(63).std() * np.sqrt(252.0)
    volatility_accelerating = volatility > volatility_slow
    rows: list[dict[str, object]] = []
    offset_rows: list[dict[str, object]] = []
    horizon_passes: dict[str, bool] = {}
    for horizon in HORIZONS:
        relative = np.log(prices["HYG"] / prices["IEF"]).diff(horizon)
        percentile = causal_rolling_percentile(relative)
        base = labels.copy()
        base["low_credit"] = percentile <= 0.20
        base["existing_risk_clear"] = ~(vix_backwardation | volatility_accelerating)
        base = base.dropna()
        period_results: list[dict[str, object]] = []
        for period, (start, end) in PERIODS.items():
            full_period = base.loc[start:end]
            period_frame = full_period.iloc[::FORWARD_DAYS].copy()
            all_result = group_differences(period_frame, "low_credit")
            clear = period_frame.loc[period_frame["existing_risk_clear"]]
            clear_result = group_differences(clear, "low_credit")
            row = {
                "horizon": horizon,
                "period": period,
                **{f"all_{key}": value for key, value in all_result.items()},
                **{f"risk_clear_{key}": value for key, value in clear_result.items()},
            }
            rows.append(row)
            period_results.append(row)
            for offset in range(FORWARD_DAYS):
                offset_frame = full_period.iloc[offset::FORWARD_DAYS].copy()
                offset_result = group_differences(offset_frame, "low_credit")
                offset_rows.append(
                    {
                        "horizon": horizon,
                        "period": period,
                        "offset": offset,
                        **offset_result,
                        "expected_direction": bool(
                            offset_result["low_credit_observations"] >= 10
                            and offset_result["forward_return_difference"] < 0.0
                            and offset_result["forward_worst_return_difference"] < 0.0
                        ),
                    }
                )
        directional = all(
            row["all_forward_return_difference"] < 0.0
            and row["all_forward_worst_return_difference"] < 0.0
            and row["all_low_credit_observations"] >= 10
            for row in period_results
        )
        independent_count = sum(
            row["risk_clear_forward_return_difference"] < 0.0
            for row in period_results
        )
        horizon_passes[str(horizon)] = bool(directional and independent_count >= 2)
    details = pd.DataFrame(rows)
    details.to_csv(OUTPUT / "period_metrics.csv", index=False)
    offsets = pd.DataFrame(offset_rows)
    offsets.to_csv(OUTPUT / "offset_metrics.csv", index=False)
    offset_summary = (
        offsets.groupby(["horizon", "period"], sort=False)["expected_direction"]
        .mean()
        .rename("expected_direction_fraction")
        .reset_index()
    )
    offset_summary.to_csv(OUTPUT / "offset_summary.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_predictive_screen",
        "production_changed": False,
        "orders_generated": False,
        "signal_horizons": list(HORIZONS),
        "forward_days": FORWARD_DAYS,
        "non_overlapping_stride": FORWARD_DAYS,
        "horizon_passes": horizon_passes,
        "offset_expected_direction_fraction": {
            f"{int(row.horizon)}:{row.period}": float(
                row.expected_direction_fraction
            )
            for row in offset_summary.itertuples(index=False)
        },
        "any_horizon_passes": any(horizon_passes.values()),
        "next_action": (
            "eligible_for_separate_preregistered_shadow_design"
            if any(horizon_passes.values())
            else "stop_without_hmm_or_portfolio_changes"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(details.to_string(index=False))


if __name__ == "__main__":
    main()
