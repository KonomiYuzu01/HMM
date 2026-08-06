from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import evaluate_probabilistic_reentry as base


DESTINATION = base.DESTINATION
MINIMUM_CALIBRATION_OBSERVATIONS = 60
CALIBRATION_C = 0.25


def logit(probability: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def state_calibrated_probabilities(
    raw_predictions: pd.DataFrame,
    label: pd.Series,
    weights: pd.DataFrame,
    horizon_days: int = base.HORIZON_DAYS,
    minimum_observations: int = MINIMUM_CALIBRATION_OBSERVATIONS,
    regularization_c: float = CALIBRATION_C,
) -> pd.DataFrame:
    frame = raw_predictions.copy()
    if label.name not in frame:
        frame = frame.join(label)
    frame = frame.join(
        weights[["QQQ", "SEMIS"]].sum(axis=1).rename("growth_weight")
    )
    rows: list[dict[str, object]] = []
    current_month: tuple[int, int] | None = None
    model: Pipeline | None = None
    conditional_base_rate = float("nan")
    training_count = 0
    training_end_date = pd.NaT
    calibration_slope = float("nan")

    for position, (date, row) in enumerate(frame.iterrows()):
        timestamp = pd.Timestamp(date)
        month = (timestamp.year, timestamp.month)
        if month != current_month:
            current_month = month
            cutoff_position = position - horizon_days
            available = (
                frame.iloc[: cutoff_position + 1]
                if cutoff_position >= 0
                else frame.iloc[:0]
            )
            training = available.loc[
                available["growth_weight"] < base.PRIMARY_CAP
            ].dropna(
                subset=[
                    "probability",
                    label.name,
                    "growth_weight",
                ]
            )
            training_count = len(training)
            training_end_date = (
                training.index[-1] if training_count else pd.NaT
            )
            model = None
            conditional_base_rate = float("nan")
            calibration_slope = float("nan")
            if training_count >= minimum_observations:
                training_label = training[label.name].astype(int)
                if training_label.nunique() == 2:
                    conditional_base_rate = float(training_label.mean())
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
                        logit(training["probability"]).reshape(-1, 1),
                        training_label,
                    )
                    calibration_slope = float(
                        model.named_steps["model"].coef_[0, 0]
                    )

        calibrated_probability = float("nan")
        raw_probability = float(row["probability"])
        if model is not None and np.isfinite(raw_probability):
            calibrated_probability = float(
                model.predict_proba(
                    logit(np.array([raw_probability])).reshape(-1, 1)
                )[0, 1]
            )
        rows.append(
            {
                "date": timestamp,
                "probability": calibrated_probability,
                "base_rate": conditional_base_rate,
                "raw_probability": raw_probability,
                "training_count": training_count,
                "training_end_date": training_end_date,
                "calibration_slope": calibration_slope,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def calibration_summary(
    sample: str,
    period: str,
    bounds: tuple[str, str | None],
    calibrated: pd.DataFrame,
    raw: pd.DataFrame,
    label: pd.Series,
    weights: pd.DataFrame,
) -> dict[str, object]:
    start, end = bounds
    frame = calibrated.join(
        raw["probability"].rename("raw_model_probability")
    ).join(label).join(
        weights[["QQQ", "SEMIS"]].sum(axis=1).rename("growth_weight")
    )
    frame = frame.loc[
        (frame["growth_weight"] < base.PRIMARY_CAP)
    ].loc[start:end].dropna()
    actual = frame[label.name].astype(int)
    return {
        "sample": sample,
        "period": period,
        "observations": len(frame),
        "positive_rate": float(actual.mean()),
        "mean_calibrated_probability": float(
            frame["probability"].mean()
        ),
        "mean_conditional_base_rate": float(frame["base_rate"].mean()),
        "mean_calibration_slope": float(
            frame["calibration_slope"].mean()
        ),
        "negative_slope_fraction": float(
            frame["calibration_slope"].lt(0.0).mean()
        ),
        "calibrated_brier": float(
            brier_score_loss(actual, frame["probability"])
        ),
        "raw_brier": float(
            brier_score_loss(actual, frame["raw_model_probability"])
        ),
        "naive_brier": float(
            np.square(frame["base_rate"] - actual).mean()
        ),
        "calibrated_roc_auc": (
            float(roc_auc_score(actual, frame["probability"]))
            if actual.nunique() == 2
            else float("nan")
        ),
    }


def main() -> None:
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for sample, settings in base.SAMPLES.items():
        prices, weights, daily = base.load_sample(settings)
        features, label, future_return = base.build_features_and_labels(
            prices
        )
        raw = base.walkforward_probabilities(
            features,
            label,
            weights.index,
        )
        calibrated = state_calibrated_probabilities(
            raw,
            label,
            weights,
        )
        calibrated.join(label).join(future_return).to_csv(
            DESTINATION / f"state_calibrated_predictions_{sample}.csv"
        )
        asset_returns = prices.pct_change(fill_method=None)
        simulations = {
            "state_calibrated_cap10": base.simulate_probe(
                weights,
                daily,
                asset_returns,
                calibrated,
                base.SENSITIVITY_CAP,
            ),
            "state_calibrated_cap20": base.simulate_probe(
                weights,
                daily,
                asset_returns,
                calibrated,
                base.PRIMARY_CAP,
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
                base.strategy_rows(
                    sample,
                    period,
                    bounds,
                    daily,
                    simulations,
                    growth_return,
                )
            )
            calibration_rows.append(
                calibration_summary(
                    sample,
                    period,
                    bounds,
                    calibrated,
                    raw,
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
                    **base.block_bootstrap(
                        frame.loc[start:end, "net_return"],
                        baseline,
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    metrics.to_csv(
        DESTINATION / "state_calibrated_metrics.csv",
        index=False,
    )
    calibration.to_csv(
        DESTINATION / "state_calibrated_calibration.csv",
        index=False,
    )
    bootstrap.to_csv(
        DESTINATION / "state_calibrated_bootstrap.csv",
        index=False,
    )
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


if __name__ == "__main__":
    main()
