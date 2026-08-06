from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "probabilistic_reentry_validation"
HORIZON_DAYS = 20
MINIMUM_TRAINING_OBSERVATIONS = 756
REGULARIZATION_C = 0.25
PRIMARY_CAP = 0.20
SENSITIVITY_CAP = 0.10
COST_BPS = 7.5
FEATURE_COLUMNS = [
    "growth_excess_log_5",
    "growth_excess_log_20",
    "growth_excess_log_63",
    "growth_excess_log_252",
    "growth_volatility_20",
    "growth_volatility_63",
    "growth_downside_variation_share_20",
    "growth_drawdown_252",
    "spx_log_return_20",
    "vix_term_ratio_minus_one",
]
SAMPLES = {
    "extended": {
        "prices": Path("data/prices_recovery_quality.csv"),
        "directory": "experiment_r9_growth_core_gold20_full_defense_2012",
        "periods": {
            "development_2012_2021": ("2012-01-01", "2021-12-31"),
            "holdout_2022_present": ("2022-01-01", None),
            "complete_2012_present": ("2012-01-01", None),
        },
    },
    "proxy": {
        "prices": Path("data/prices_20y_proxy.csv"),
        "directory": (
            "experiment_r9_growth_core_gold20_full_defense_20y_proxy"
        ),
        "periods": {
            "early_2006_2014": ("2006-08-01", "2014-12-31"),
            "late_2015_present": ("2015-01-01", None),
            "complete_2006_present": ("2006-08-01", None),
        },
    },
}


def future_sum(series: pd.Series, horizon_days: int) -> pd.Series:
    return series.rolling(horizon_days).sum().shift(-(horizon_days - 1))


def build_features_and_labels(
    prices: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    returns = prices.pct_change(fill_method=None)
    growth_return = returns[["QQQ", "SEMIS"]].mean(axis=1)
    growth_log_return = np.log1p(growth_return.clip(lower=-0.999999))
    cash_log_return = np.log1p(
        returns["CASH"].clip(lower=-0.999999)
    )
    growth_excess_log = growth_log_return - cash_log_return

    features = pd.DataFrame(index=prices.index)
    for days in (5, 20, 63, 252):
        features[f"growth_excess_log_{days}"] = (
            growth_excess_log.rolling(days).sum().shift(1)
        )
    for days in (20, 63):
        features[f"growth_volatility_{days}"] = (
            growth_return.rolling(days).std(ddof=1).shift(1)
            * np.sqrt(252.0)
        )
    downside_square = growth_return.clip(upper=0.0).pow(2)
    total_square = growth_return.pow(2)
    features["growth_downside_variation_share_20"] = (
        downside_square.rolling(20).sum()
        / total_square.rolling(20).sum().replace(0.0, np.nan)
    ).shift(1)
    growth_equity = np.exp(growth_log_return.fillna(0.0).cumsum())
    features["growth_drawdown_252"] = (
        growth_equity
        / growth_equity.rolling(252).max()
        - 1.0
    ).shift(1)
    features["spx_log_return_20"] = (
        np.log1p(returns["SPX"].clip(lower=-0.999999))
        .rolling(20)
        .sum()
        .shift(1)
    )
    features["vix_term_ratio_minus_one"] = (
        prices["VIX"] / prices["VIX3M"] - 1.0
    ).shift(1)
    features = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    future_excess_log_return = future_sum(
        growth_excess_log,
        horizon_days,
    )
    label = future_excess_log_return.gt(0.0).astype(float)
    label[future_excess_log_return.isna()] = np.nan
    label.name = "future_growth_outperforms_cash"
    future_excess_log_return.name = "future_excess_log_return"
    return features, label, future_excess_log_return


def walkforward_probabilities(
    features: pd.DataFrame,
    label: pd.Series,
    prediction_index: pd.DatetimeIndex,
    horizon_days: int = HORIZON_DAYS,
    minimum_training_observations: int = MINIMUM_TRAINING_OBSERVATIONS,
    regularization_c: float = REGULARIZATION_C,
) -> pd.DataFrame:
    aligned = features.join(label)
    rows: list[dict[str, object]] = []
    current_month: tuple[int, int] | None = None
    model: Pipeline | None = None
    base_rate = float("nan")
    training_count = 0
    training_end_date = pd.NaT

    for date in prediction_index:
        timestamp = pd.Timestamp(date)
        month = (timestamp.year, timestamp.month)
        if month != current_month:
            current_month = month
            position = int(features.index.searchsorted(timestamp))
            cutoff_position = position - horizon_days
            available = (
                aligned.iloc[: cutoff_position + 1]
                if cutoff_position >= 0
                else aligned.iloc[:0]
            )
            training = available.dropna()
            training_count = len(training)
            training_end_date = (
                training.index[-1] if training_count else pd.NaT
            )
            model = None
            base_rate = float("nan")
            if training_count >= minimum_training_observations:
                training_label = training[label.name].astype(int)
                if training_label.nunique() == 2:
                    base_rate = float(training_label.mean())
                    model = Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                LogisticRegression(
                                    C=regularization_c,
                                    max_iter=2_000,
                                    random_state=20_260_725,
                                ),
                            ),
                        ]
                    )
                    model.fit(
                        training[FEATURE_COLUMNS],
                        training_label,
                    )

        probability = float("nan")
        if model is not None and timestamp in features.index:
            row = features.loc[[timestamp], FEATURE_COLUMNS]
            if row.notna().all(axis=None):
                probability = float(model.predict_proba(row)[0, 1])
        rows.append(
            {
                "date": timestamp,
                "probability": probability,
                "base_rate": base_rate,
                "training_count": training_count,
                "training_end_date": training_end_date,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def simulate_probe(
    base_weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    asset_returns: pd.DataFrame,
    predictions: pd.DataFrame,
    cap: float,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    if not 0.0 <= cap <= 1.0:
        raise ValueError("Probe cap must be between zero and one")
    index = base_weights.index.intersection(base_daily.index)
    index = index.intersection(asset_returns.index)
    probability_frame = predictions.reindex(index)
    previous_addition = 0.0
    rows: list[dict[str, object]] = []

    for date in index:
        weights = base_weights.loc[date]
        growth_weight = float(weights[["QQQ", "SEMIS"]].sum())
        cash_weight = float(weights["CASH"])
        probability = float(probability_frame.loc[date, "probability"])
        base_rate = float(probability_frame.loc[date, "base_rate"])
        desired_probe = 0.0
        if (
            growth_weight < PRIMARY_CAP
            and np.isfinite(probability)
            and np.isfinite(base_rate)
            and probability > base_rate
            and base_rate < 1.0
        ):
            desired_probe = cap * (
                (probability - base_rate) / (1.0 - base_rate)
            )
        addition = min(
            max(desired_probe - growth_weight, 0.0),
            max(cash_weight, 0.0),
        )
        incremental_turnover = abs(addition - previous_addition)
        incremental_cost = (
            2.0 * incremental_turnover * cost_bps / 10_000.0
        )
        excess_return = float(
            asset_returns.loc[date, "QQQ"]
            - asset_returns.loc[date, "CASH"]
        )
        incremental_gross_return = addition * excess_return
        net_return = float(
            base_daily.loc[date, "net_return"]
            + incremental_gross_return
            - incremental_cost
        )
        rows.append(
            {
                "date": date,
                "net_return": net_return,
                "base_net_return": float(
                    base_daily.loc[date, "net_return"]
                ),
                "incremental_gross_return": incremental_gross_return,
                "incremental_cost": incremental_cost,
                "incremental_turnover": incremental_turnover,
                "base_growth_weight": growth_weight,
                "base_cash_weight": cash_weight,
                "probability": probability,
                "base_rate": base_rate,
                "desired_probe": desired_probe,
                "added_qqq_weight": addition,
            }
        )
        previous_addition = addition
    result = pd.DataFrame(rows).set_index("date")
    result["equity"] = (1.0 + result["net_return"]).cumprod()
    result["drawdown"] = (
        result["equity"] / result["equity"].cummax() - 1.0
    )
    return result


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
        np.log1p(aligned["candidate"])
        - np.log1p(aligned["baseline"])
    ).to_numpy(dtype=float)
    count = len(relative)
    block_count = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for sample_index in range(samples):
        starts = rng.integers(0, count, size=block_count)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        estimates[sample_index] = float(
            relative[indices].mean() * 252.0
        )
    return {
        "annualized_relative_log_return": float(
            relative.mean() * 252.0
        ),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "probability_positive": float((estimates > 0.0).mean()),
    }


def load_sample(
    settings: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(
        settings["prices"],
        index_col=0,
        parse_dates=True,
    )
    directory = OUTPUT / str(settings["directory"])
    weights = pd.read_csv(
        directory / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        directory / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    return prices, weights, daily


def calibration_rows(
    sample: str,
    period: str,
    bounds: tuple[str, str | None],
    predictions: pd.DataFrame,
    label: pd.Series,
    weights: pd.DataFrame,
) -> list[dict[str, object]]:
    start, end = bounds
    prediction_frame = predictions.copy()
    if label.name not in prediction_frame:
        prediction_frame = prediction_frame.join(label)
    frame = prediction_frame.join(
        weights[["QQQ", "SEMIS"]].sum(axis=1).rename("growth_weight")
    ).loc[start:end].dropna()
    rows: list[dict[str, object]] = []
    for subset, selected in {
        "all_predictions": frame,
        "growth_below_20pct": frame.loc[frame["growth_weight"] < PRIMARY_CAP],
    }.items():
        if selected.empty:
            continue
        actual = selected[label.name].astype(int)
        probability = selected["probability"]
        base_rate = selected["base_rate"]
        rows.append(
            {
                "sample": sample,
                "period": period,
                "subset": subset,
                "observations": len(selected),
                "positive_rate": float(actual.mean()),
                "mean_probability": float(probability.mean()),
                "mean_training_base_rate": float(base_rate.mean()),
                "model_brier": float(
                    brier_score_loss(actual, probability)
                ),
                "naive_brier": float(
                    np.square(base_rate - actual).mean()
                ),
                "brier_improvement": float(
                    np.square(base_rate - actual).mean()
                    - brier_score_loss(actual, probability)
                ),
                "roc_auc": (
                    float(roc_auc_score(actual, probability))
                    if actual.nunique() == 2
                    else float("nan")
                ),
            }
        )
    return rows


def strategy_rows(
    sample: str,
    period: str,
    bounds: tuple[str, str | None],
    daily: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    growth_return: pd.Series,
) -> list[dict[str, object]]:
    start, end = bounds
    rows: list[dict[str, object]] = []
    simulation_index = next(iter(simulations.values())).index
    strategies = {
        "baseline": daily["net_return"].reindex(simulation_index),
        **{
            name: frame["net_return"]
            for name, frame in simulations.items()
        },
    }
    for strategy, returns in strategies.items():
        selected = returns.loc[start:end]
        rows.append(
            {
                "sample": sample,
                "period": period,
                "strategy": strategy,
                **performance_metrics(selected),
                "up_capture": capture_ratio(
                    selected,
                    growth_return,
                    True,
                ),
                "down_capture": capture_ratio(
                    selected,
                    growth_return,
                    False,
                ),
                "annualized_incremental_cost": (
                    float(
                        simulations[strategy]
                        .loc[start:end, "incremental_cost"]
                        .mean()
                        * 252.0
                    )
                    if strategy in simulations
                    else 0.0
                ),
                "average_added_qqq_weight": (
                    float(
                        simulations[strategy]
                        .loc[start:end, "added_qqq_weight"]
                        .mean()
                    )
                    if strategy in simulations
                    else 0.0
                ),
                "active_probe_fraction": (
                    float(
                        simulations[strategy]
                        .loc[start:end, "added_qqq_weight"]
                        .gt(0.0)
                        .mean()
                    )
                    if strategy in simulations
                    else 0.0
                ),
            }
        )
    return rows


def acceptance_summary(
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    indexed = metrics.set_index(["sample", "period", "strategy"])

    def metric_delta(
        sample: str,
        period: str,
        column: str,
    ) -> float:
        return float(
            indexed.loc[(sample, period, "probability_cap20"), column]
            - indexed.loc[(sample, period, "baseline"), column]
        )

    complete_cagr = metric_delta(
        "extended", "complete_2012_present", "cagr"
    )
    development_cagr = metric_delta(
        "extended", "development_2012_2021", "cagr"
    )
    holdout_cagr = metric_delta(
        "extended", "holdout_2022_present", "cagr"
    )
    complete_up_capture = metric_delta(
        "extended", "complete_2012_present", "up_capture"
    )
    complete_drawdown = metric_delta(
        "extended", "complete_2012_present", "max_drawdown"
    )
    proxy_cagr = metric_delta(
        "proxy", "complete_2006_present", "cagr"
    )
    calibration_indexed = calibration.set_index(
        ["sample", "period", "subset"]
    )
    brier_improvement = float(
        calibration_indexed.loc[
            (
                "extended",
                "complete_2012_present",
                "growth_below_20pct",
            ),
            "brier_improvement",
        ]
    )
    checks = {
        "complete_cagr_nonnegative": complete_cagr >= 0.0,
        "development_cagr_nonnegative": development_cagr >= 0.0,
        "holdout_cagr_nonnegative": holdout_cagr >= 0.0,
        "up_capture_positive": complete_up_capture > 0.0,
        "drawdown_not_worse_50bp": complete_drawdown >= -0.005,
        "proxy_cagr_nonnegative": proxy_cagr >= 0.0,
        "brier_better_than_naive": brier_improvement > 0.0,
    }
    row: dict[str, object] = {
        "complete_cagr_delta": complete_cagr,
        "development_cagr_delta": development_cagr,
        "holdout_cagr_delta": holdout_cagr,
        "complete_up_capture_delta": complete_up_capture,
        "complete_max_drawdown_delta": complete_drawdown,
        "proxy_cagr_delta": proxy_cagr,
        "low_exposure_brier_improvement": brier_improvement,
        **{name: int(value) for name, value in checks.items()},
        "all_gates_pass": int(all(checks.values())),
    }
    return pd.DataFrame([row], index=["probability_cap20"])


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    calibration_result_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for sample, settings in SAMPLES.items():
        prices, weights, daily = load_sample(settings)
        features, label, future_return = build_features_and_labels(prices)
        predictions = walkforward_probabilities(
            features,
            label,
            weights.index,
        )
        predictions = predictions.join(label).join(future_return)
        predictions.to_csv(DESTINATION / f"predictions_{sample}.csv")
        asset_returns = prices.pct_change(fill_method=None)
        simulations = {
            "probability_cap10": simulate_probe(
                weights,
                daily,
                asset_returns,
                predictions,
                SENSITIVITY_CAP,
            ),
            "probability_cap20": simulate_probe(
                weights,
                daily,
                asset_returns,
                predictions,
                PRIMARY_CAP,
            ),
        }
        for strategy, frame in simulations.items():
            frame.to_csv(
                DESTINATION / f"daily_{sample}_{strategy}.csv"
            )

        growth_return = asset_returns[["QQQ", "SEMIS"]].mean(axis=1)
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for period, bounds in periods.items():
            metric_rows.extend(
                strategy_rows(
                    sample,
                    period,
                    bounds,
                    daily,
                    simulations,
                    growth_return,
                )
            )
            calibration_result_rows.extend(
                calibration_rows(
                    sample,
                    period,
                    bounds,
                    predictions,
                    label,
                    weights,
                )
            )

        complete_period = next(
            name for name in periods if name.startswith("complete_")
        )
        start, end = periods[complete_period]
        baseline = daily.loc[start:end, "net_return"]
        for strategy, frame in simulations.items():
            bootstrap_rows.append(
                {
                    "sample": sample,
                    "strategy": strategy,
                    **block_bootstrap(
                        frame.loc[start:end, "net_return"],
                        baseline,
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_result_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    acceptance = acceptance_summary(metrics, calibration)
    metrics.to_csv(DESTINATION / "metrics.csv", index=False)
    calibration.to_csv(DESTINATION / "calibration.csv", index=False)
    bootstrap.to_csv(DESTINATION / "bootstrap.csv", index=False)
    acceptance.to_csv(DESTINATION / "acceptance.csv")

    print("Metrics:")
    print(
        metrics[
            [
                "sample",
                "period",
                "strategy",
                "cagr",
                "max_drawdown",
                "up_capture",
                "down_capture",
                "annualized_incremental_cost",
                "average_added_qqq_weight",
                "active_probe_fraction",
            ]
        ].round(6).to_string(index=False)
    )
    print("\nCalibration:")
    print(calibration.round(6).to_string(index=False))
    print("\nBootstrap:")
    print(bootstrap.round(6).to_string(index=False))
    print("\nAcceptance:")
    print(acceptance.round(6).to_string())


if __name__ == "__main__":
    main()
