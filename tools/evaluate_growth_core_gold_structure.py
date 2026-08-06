from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "growth_core_gold_structure_validation"
VARIANTS = {
    "fixed_gold00": "experiment_r9_growth_core_gold00_full_defense",
    "fixed_gold05": "experiment_r9_growth_core_gold05_full_defense",
    "fixed_gold10": "experiment_r9_growth_core_gold10_full_defense",
    "fixed_gold20": "experiment_r9_growth_core_gold20_full_defense",
    "gold_trend10": "experiment_r9_growth_core_gold_trend10",
    "gold_trend20": "experiment_r9_growth_core_gold_trend20",
    "multi_trend10": "experiment_r9_growth_core_multi_trend10",
    "multi_trend20": "experiment_r9_growth_core_multi_trend20",
    "gold05_multi_trend10": "experiment_r9_growth_core_gold05_multi_trend10",
    "gold05_multi_trend15": "experiment_r9_growth_core_gold05_multi_trend15",
    "gold10_multi_trend10": "experiment_r9_growth_core_gold10_multi_trend10",
    "gold20_own_trend": "experiment_r9_growth_core_gold20_own_trend",
    "gold10_plus10_own_trend": (
        "experiment_r9_growth_core_gold10_plus10_own_trend"
    ),
}
SAMPLES = {
    "normal": {
        "suffix": "",
        "price_path": Path("data/prices_recovery_quality.csv"),
        "periods": {
            "development_2015_2021": ("2015-01-01", "2021-12-31"),
            "holdout_2022_2026": ("2022-01-01", "2026-12-31"),
            "complete_2015_2026": ("2015-01-01", "2026-12-31"),
        },
    },
    "extended": {
        "suffix": "_2012",
        "price_path": Path("data/prices_recovery_quality.csv"),
        "periods": {
            "early_2012_2014": ("2012-01-01", "2014-12-31"),
            "late_2015_2026": ("2015-01-01", "2026-12-31"),
            "complete_2012_2026": ("2012-01-01", "2026-12-31"),
        },
    },
    "proxy": {
        "suffix": "_20y_proxy",
        "price_path": Path("data/prices_20y_proxy.csv"),
        "periods": {
            "early_2006_2014": ("2006-08-01", "2014-12-31"),
            "late_2015_2026": ("2015-01-01", "2026-12-31"),
            "complete_2006_2026": ("2006-08-01", "2026-12-31"),
        },
    },
}
FIXED_VARIANTS = {
    key: value for key, value in VARIANTS.items() if key.startswith("fixed_")
}
CROSS_SAMPLE_VARIANTS = {
    **FIXED_VARIANTS,
    "gold05_multi_trend15": VARIANTS["gold05_multi_trend15"],
    "gold10_multi_trend10": VARIANTS["gold10_multi_trend10"],
    "gold20_own_trend": VARIANTS["gold20_own_trend"],
}


def load_output(directory: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT / directory / filename,
        index_col=0,
        parse_dates=True,
    )


def capture_ratio(
    strategy: pd.Series,
    benchmark: pd.Series,
    positive: bool,
) -> float:
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    mask = aligned["benchmark"] > 0.0 if positive else aligned["benchmark"] < 0.0
    return float(
        aligned.loc[mask, "strategy"].mean()
        / aligned.loc[mask, "benchmark"].mean()
    )


def block_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    block_days: int = 21,
    samples: int = 10_000,
    seed: int = 20_260_725,
) -> dict[str, float]:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    relative = (
        np.log1p(aligned["candidate"]) - np.log1p(aligned["baseline"])
    ).to_numpy(dtype=float)
    count = len(relative)
    block_count = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for sample_index in range(samples):
        starts = rng.integers(0, count, size=block_count)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        estimates[sample_index] = float(relative[indices].mean() * 252.0)
    return {
        "annualized_relative_log_return": float(relative.mean() * 252.0),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "probability_positive": float((estimates > 0.0).mean()),
    }


def available_variants(sample: str) -> dict[str, str]:
    return VARIANTS if sample == "normal" else CROSS_SAMPLE_VARIANTS


def metrics_and_capture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample, settings in SAMPLES.items():
        suffix = str(settings["suffix"])
        prices = pd.read_csv(
            settings["price_path"],
            index_col=0,
            parse_dates=True,
        )
        growth = prices[["QQQ", "SEMIS"]].pct_change(
            fill_method=None
        ).mean(axis=1)
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for variant, directory in available_variants(sample).items():
            daily = load_output(f"{directory}{suffix}", "daily_returns.csv")
            for period, bounds in periods.items():
                start, end = bounds
                frame = daily.loc[start:end]
                strategy = frame["net_return"]
                aligned_growth = growth.reindex(strategy.index)
                rows.append(
                    {
                        "sample": sample,
                        "variant": variant,
                        "period": period,
                        **performance_metrics(strategy),
                        "up_capture": capture_ratio(
                            strategy,
                            aligned_growth,
                            True,
                        ),
                        "down_capture": capture_ratio(
                            strategy,
                            aligned_growth,
                            False,
                        ),
                        "annualized_cost_drag": float(
                            frame["cost"].mean() * 252.0
                        ),
                    }
                )
    return pd.DataFrame(rows).set_index(["sample", "variant", "period"])


def weight_summary() -> pd.DataFrame:
    assets = ["QQQ", "SEMIS", "GOLD", "BOND", "OIL", "USD", "CASH"]
    rows: list[dict[str, object]] = []
    for sample, settings in SAMPLES.items():
        suffix = str(settings["suffix"])
        for variant, directory in available_variants(sample).items():
            weights = load_output(f"{directory}{suffix}", "weights.csv")
            means = weights.reindex(columns=assets, fill_value=0.0).mean()
            rows.append(
                {
                    "sample": sample,
                    "variant": variant,
                    **{f"mean_{asset.lower()}": float(means[asset]) for asset in assets},
                    "mean_growth": float(means["QQQ"] + means["SEMIS"]),
                    "mean_diversifier": float(
                        means[["GOLD", "BOND", "OIL", "USD"]].sum()
                    ),
                }
            )
    return pd.DataFrame(rows).set_index(["sample", "variant"])


def bootstrap_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample, settings in SAMPLES.items():
        suffix = str(settings["suffix"])
        periods = settings["periods"]
        assert isinstance(periods, dict)
        complete_period = next(
            period for period in periods if period.startswith("complete_")
        )
        start, end = periods[complete_period]
        baseline = load_output(
            f"{FIXED_VARIANTS['fixed_gold20']}{suffix}",
            "daily_returns.csv",
        ).loc[start:end, "net_return"]
        for variant, directory in available_variants(sample).items():
            if variant == "fixed_gold20":
                continue
            candidate = load_output(
                f"{directory}{suffix}",
                "daily_returns.csv",
            ).loc[start:end, "net_return"]
            rows.append(
                {
                    "sample": sample,
                    "variant": variant,
                    **block_bootstrap(candidate, baseline),
                }
            )
    return pd.DataFrame(rows).set_index(["sample", "variant"])


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metrics = metrics_and_capture()
    weights = weight_summary()
    bootstrap = bootstrap_summary()
    metrics.to_csv(DESTINATION / "metrics_and_capture.csv")
    weights.to_csv(DESTINATION / "average_weights.csv")
    bootstrap.to_csv(DESTINATION / "bootstrap_vs_fixed_gold20.csv")
    print(metrics.round(4).to_string())
    print("\nAverage weights:")
    print(weights.round(4).to_string())
    print("\nBootstrap versus fixed 20% gold:")
    print(bootstrap.round(4).to_string())


if __name__ == "__main__":
    main()
