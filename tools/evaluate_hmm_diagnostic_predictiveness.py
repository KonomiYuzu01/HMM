from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from tools.evaluate_hmm_whitebox_diagnostics import causal_percentile
import tools.evaluate_r21_current_engine_industry_momentum as r21


OUTPUT = Path("output/hmm_diagnostic_predictiveness")
DIRECTORIES = {
    "normal_synthetic": Path("output/research_hmm_gaussian_diagnostics"),
    "proxy_synthetic": Path(
        "output/research_hmm_gaussian_diagnostics_20y_proxy"
    ),
}
HORIZON = 21
SEVERE_DRAWDOWN = -0.05
MEMBER_COLUMNS = [
    "one_step_predictive_log_likelihood",
    "template_expected_nearest_distance",
    "paper_risk_on_candidate",
]
CANDIDATE_SIGNALS = [
    "negative_log_likelihood_percentile",
    "template_distance_percentile",
    "favorable_probability_drop_1",
    "favorable_probability_drop_4",
]
BENCHMARK_SIGNALS = [
    "volatility_acceleration",
    "growth_weakness_21d",
    "prior_session_loss",
]


def causal_template_risk_on_probability(frame: pd.DataFrame) -> pd.Series:
    template_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("template_") and column.endswith("_probability")
    )
    if not template_columns:
        raise ValueError("Regime frame has no template probabilities")
    alpha = np.ones(len(template_columns), dtype=float)
    beta = np.ones(len(template_columns), dtype=float)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        probabilities = row[template_columns].to_numpy(dtype=float)
        probabilities = np.maximum(probabilities, 0.0)
        total = float(probabilities.sum())
        if total <= 0.0:
            continue
        probabilities /= total
        prior_rates = alpha / (alpha + beta)
        result.loc[date] = float(probabilities @ prior_rates)
        observed = float(row["paper_risk_on_candidate"])
        alpha += probabilities * observed
        beta += probabilities * (1.0 - observed)
    return result.rename("causal_template_risk_on_probability")


def load_ensemble_regimes(directory: Path) -> pd.DataFrame:
    members: list[pd.DataFrame] = []
    for path in sorted((directory / "members").glob("seed_*/regimes.csv")):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        template_columns = sorted(
            column
            for column in frame.columns
            if column.startswith("template_")
            and column.endswith("_probability")
        )
        missing = sorted(set(MEMBER_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing diagnostics: {missing}")
        selected = frame[MEMBER_COLUMNS + template_columns].astype(float)
        selected["causal_template_risk_on_probability"] = (
            causal_template_risk_on_probability(selected)
        )
        members.append(selected)
    if not members:
        raise ValueError(f"No member regimes found in {directory}")
    common = members[0].index
    for frame in members[1:]:
        common = common.intersection(frame.index)
    stacked = pd.concat(
        [frame.reindex(common) for frame in members],
        keys=range(len(members)),
        names=["member", "date"],
    )
    ensemble = stacked.groupby(level="date").mean()
    ensemble["negative_log_likelihood_percentile"] = causal_percentile(
        -ensemble["one_step_predictive_log_likelihood"]
    )
    ensemble["template_distance_percentile"] = causal_percentile(
        ensemble["template_expected_nearest_distance"]
    )
    favorable = ensemble["causal_template_risk_on_probability"]
    ensemble["favorable_probability_drop_1"] = -favorable.diff(1)
    ensemble["favorable_probability_drop_4"] = -favorable.diff(4)
    return ensemble


def market_signals(closes: pd.DataFrame) -> pd.DataFrame:
    selected = closes[["QQQ", "SEMIS"]].astype(float)
    log_returns = np.log(selected).diff()
    known = log_returns.shift(1)
    vol21 = known.rolling(21, min_periods=21).std(ddof=1)
    vol63 = known.rolling(63, min_periods=63).std(ddof=1).clip(lower=1e-8)
    growth = known.mean(axis=1)
    return pd.DataFrame(
        {
            "volatility_acceleration": (vol21 / vol63).max(axis=1),
            "growth_weakness_21d": -growth.rolling(
                21, min_periods=21
            ).sum(),
            "prior_session_loss": -growth,
        },
        index=selected.index,
    )


def forward_outcomes(
    closes: pd.DataFrame,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    selected = closes[["QQQ", "SEMIS"]].astype(float)
    growth_log_return = np.log(selected).diff().mean(axis=1)
    values = growth_log_return.to_numpy(dtype=float)
    forward_return = pd.Series(np.nan, index=selected.index, dtype=float)
    forward_drawdown = pd.Series(np.nan, index=selected.index, dtype=float)
    for position in range(1, len(selected) - horizon + 1):
        path = values[position : position + horizon]
        if not bool(np.isfinite(path).all()):
            continue
        cumulative = np.exp(np.cumsum(path))
        forward_return.iloc[position] = float(cumulative[-1] - 1.0)
        forward_drawdown.iloc[position] = float(
            np.minimum.accumulate(cumulative).min() - 1.0
        )
    return pd.DataFrame(
        {
            "forward_growth_return_21d": forward_return,
            "forward_growth_drawdown_21d": forward_drawdown,
            "loss_next_21d": forward_return.lt(0.0),
            "severe_drawdown_next_21d": forward_drawdown.le(SEVERE_DRAWDOWN),
        },
        index=selected.index,
    )


def signal_metrics(
    frame: pd.DataFrame,
    signal: str,
) -> dict[str, float | int | str]:
    selected = frame.dropna(
        subset=[
            signal,
            "forward_growth_return_21d",
            "forward_growth_drawdown_21d",
        ]
    )
    loss = selected["loss_next_21d"].astype(int)
    severe = selected["severe_drawdown_next_21d"].astype(int)

    def auc(label: pd.Series) -> float:
        if len(label) == 0 or label.nunique() < 2:
            return np.nan
        return float(roc_auc_score(label, selected[signal]))

    return {
        "signal": signal,
        "observations": len(selected),
        "loss_auc": auc(loss),
        "severe_drawdown_auc": auc(severe),
        "return_loss_spearman": float(
            spearmanr(
                selected[signal], -selected["forward_growth_return_21d"]
            ).statistic
        ),
        "drawdown_severity_spearman": float(
            spearmanr(
                selected[signal], -selected["forward_growth_drawdown_21d"]
            ).statistic
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    rows: list[dict[str, object]] = []
    for sample, directory in DIRECTORIES.items():
        settings = samples[sample]
        closes = settings["closes"]
        periods = settings["periods"]
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)
        frame = (
            load_ensemble_regimes(directory)
            .join(market_signals(closes), how="left")
            .join(forward_outcomes(closes), how="left")
        )
        frame.to_csv(OUTPUT / f"{sample}_signals_and_outcomes.csv")
        for period, (start, end) in periods.items():
            selected = frame.loc[start:end]
            for signal in CANDIDATE_SIGNALS + BENCHMARK_SIGNALS:
                rows.append(
                    {
                        "sample": sample,
                        "period": period,
                        **signal_metrics(selected, signal),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)

    gate_periods = {
        "normal_development": (
            "normal_synthetic",
            "development_2015_2021",
        ),
        "normal_holdout": ("normal_synthetic", "holdout_2022_2025"),
        "proxy_early": ("proxy_synthetic", "early_2006_2014"),
        "proxy_late": ("proxy_synthetic", "late_2015_2026"),
    }

    def beats_benchmarks(signal: str, sample: str, period: str) -> bool:
        selected = metrics.loc[
            metrics["sample"].eq(sample) & metrics["period"].eq(period)
        ]
        candidate = float(
            selected.loc[
                selected["signal"].eq(signal), "severe_drawdown_auc"
            ].iloc[0]
        )
        benchmark = float(
            selected.loc[
                selected["signal"].isin(BENCHMARK_SIGNALS),
                "severe_drawdown_auc",
            ].max()
        )
        return bool(np.isfinite(candidate) and candidate >= benchmark)

    direction_gates: dict[str, dict[str, bool]] = {}
    for direction, signals in {
        "ood": [
            "negative_log_likelihood_percentile",
            "template_distance_percentile",
        ],
        "posterior_velocity": [
            "favorable_probability_drop_1",
            "favorable_probability_drop_4",
        ],
    }.items():
        gates: dict[str, bool] = {}
        for label, (sample, period) in gate_periods.items():
            gates[label] = any(
                beats_benchmarks(signal, sample, period) for signal in signals
            )
        direction_gates[direction] = gates
    passes = {
        direction: bool(all(gates.values()))
        for direction, gates in direction_gates.items()
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_predictive_screen",
        "production_changed": False,
        "orders_generated": False,
        "direction_gates": direction_gates,
        "research_pass": passes,
        "decision": {
            direction: (
                "eligible_for_economic_shadow"
                if passed
                else "stop_before_portfolio_mapping"
            )
            for direction, passed in passes.items()
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
