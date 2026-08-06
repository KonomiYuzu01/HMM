from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


DESTINATION = Path("output/return_engine_validation")


def regression(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    aligned = pd.concat([y.rename("strategy"), x], axis=1, join="inner").dropna()
    design = np.column_stack(
        [np.ones(len(aligned)), aligned[x.columns].to_numpy(dtype=float)]
    )
    coefficients = np.linalg.lstsq(
        design, aligned["strategy"].to_numpy(dtype=float), rcond=None
    )[0]
    fitted = design @ coefficients
    residual = aligned["strategy"].to_numpy(dtype=float) - fitted
    total = aligned["strategy"].to_numpy(dtype=float) - float(
        aligned["strategy"].mean()
    )
    result = {
        "annualized_intercept": float(coefficients[0] * 252.0),
        "r_squared": float(1.0 - residual @ residual / (total @ total)),
    }
    result.update(
        {f"beta_{name}": float(value) for name, value in zip(x, coefficients[1:])}
    )
    return pd.Series(result)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    strategy = pd.read_csv(
        "output/open_execution_validation/gold_20_daily_asset_cap_daily.csv",
        index_col=0,
        parse_dates=True,
    )["net_return"]
    prices = pd.read_csv(
        "data/adjusted_open_close_2011_present.csv",
        index_col=0,
        parse_dates=True,
    )
    benchmarks = prices[
        ["close_SPX", "close_QQQ", "close_SEMIS", "close_GOLD"]
    ].pct_change(fill_method=None)
    benchmarks.columns = ["SPY", "QQQ", "SMH", "GLD"]
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmarks], axis=1, join="inner"
    ).dropna()

    regression_rows = {
        "SPY_single_factor": regression(aligned["strategy"], aligned[["SPY"]]),
        "growth_gold_factors": regression(
            aligned["strategy"], aligned[["QQQ", "SMH", "GLD"]]
        ),
    }
    regressions = pd.DataFrame(regression_rows).T
    regressions.to_csv(DESTINATION / "factor_regressions.csv")

    capture_rows = []
    for benchmark in ("SPY", "QQQ", "SMH"):
        for direction, mask in (
            ("up", aligned[benchmark] > 0.0),
            ("down", aligned[benchmark] < 0.0),
        ):
            capture_rows.append(
                {
                    "benchmark": benchmark,
                    "direction": direction,
                    "capture_ratio": float(
                        aligned.loc[mask, "strategy"].mean()
                        / aligned.loc[mask, benchmark].mean()
                    ),
                    "observations": int(mask.sum()),
                }
            )
    capture = pd.DataFrame(capture_rows).set_index(["benchmark", "direction"])
    capture.to_csv(DESTINATION / "capture_ratios.csv")

    return_rows = []
    for name in aligned:
        metrics = performance_metrics(aligned[name])
        return_rows.append(
            {
                "asset": name,
                **metrics,
                "annualized_arithmetic_return": float(aligned[name].mean() * 252.0),
                "variance_drag_approximation": float(
                    0.5 * aligned[name].var(ddof=1) * 252.0
                ),
            }
        )
    return_profile = pd.DataFrame(return_rows).set_index("asset")
    return_profile.to_csv(DESTINATION / "return_profile.csv")

    weights = pd.read_csv(
        "output/paper_core_growth_gold20_daily_risk_ensemble/weights.csv",
        index_col=0,
        parse_dates=True,
    ).loc["2015":"2025"]
    exposure = pd.DataFrame(
        {
            "average_weight": weights.mean(),
            "median_weight": weights.median(),
            "ten_percentile": weights.quantile(0.10),
            "ninety_percentile": weights.quantile(0.90),
        }
    )
    exposure.loc["QQQ_PLUS_SMH"] = {
        "average_weight": float((weights["QQQ"] + weights["SEMIS"]).mean()),
        "median_weight": float((weights["QQQ"] + weights["SEMIS"]).median()),
        "ten_percentile": float((weights["QQQ"] + weights["SEMIS"]).quantile(0.10)),
        "ninety_percentile": float((weights["QQQ"] + weights["SEMIS"]).quantile(0.90)),
    }
    exposure.to_csv(DESTINATION / "exposure_profile.csv")

    diagnostics = pd.Series(
        {
            "fraction_days_growth_exposure_below_50pct": float(
                ((weights["QQQ"] + weights["SEMIS"]) < 0.50).mean()
            ),
            "spy_up_capture": float(capture.loc[("SPY", "up"), "capture_ratio"]),
            "spy_down_capture": float(capture.loc[("SPY", "down"), "capture_ratio"]),
            "timing_capture_spread": float(
                capture.loc[("SPY", "up"), "capture_ratio"]
                - capture.loc[("SPY", "down"), "capture_ratio"]
            ),
        },
        name="value",
    )
    diagnostics.to_csv(DESTINATION / "diagnostics.csv")

    print("Factor regressions:")
    print(regressions.round(6).to_string())
    print("\nCapture ratios:")
    print(capture.round(6).to_string())
    print("\nReturn profile:")
    print(
        return_profile[
            [
                "cagr",
                "annualized_arithmetic_return",
                "annual_volatility",
                "variance_drag_approximation",
                "max_drawdown",
            ]
        ]
        .round(6)
        .to_string()
    )
    print("\nExposure profile:")
    print(exposure.round(6).to_string())
    print("\nDiagnostics:")
    print(diagnostics.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
