from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from tools.evaluate_r11_diversified_capital_grid import (
    load_strategy_inputs,
    metric_delta,
)
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r39_relative_damage_concentration_veto as r39


OUTPUT = Path("output/semiconductor_residual_hmm")
NORMAL_DIRECTORY = Path("output/research_hmm_gaussian_diagnostics")
PROXY_DIRECTORY = Path("output/research_hmm_gaussian_diagnostics_20y_proxy")
SEEDS = (7, 42, 123)
TRAINING_DAYS = 1008
REFIT_DAYS = 21
FORWARD_DAYS = 21
WEAK_PROBABILITY_THRESHOLD = 0.50
MAXIMUM_SEMIS_GROWTH_SHARE = 0.50
TRADING_EPSILON = 1e-14


def build_causal_features(closes: pd.DataFrame) -> pd.DataFrame:
    missing = sorted({"QQQ", "SEMIS"}.difference(closes.columns))
    if missing:
        raise ValueError(f"Missing semiconductor residual assets: {missing}")
    selected = closes[["QQQ", "SEMIS"]].astype(float)
    log_returns = np.log(selected.div(selected.shift(1)))
    relative = log_returns["SEMIS"] - log_returns["QQQ"]
    features = pd.DataFrame(index=selected.index)
    for days in (5, 21, 63):
        features[f"relative_log_return_{days}d"] = (
            relative.rolling(days, min_periods=days).sum().shift(1)
        )
    features["relative_volatility_21d"] = (
        relative.rolling(21, min_periods=21).std(ddof=1).shift(1)
    )
    features["semis_qqq_volatility_ratio_21d"] = (
        log_returns["SEMIS"]
        .rolling(21, min_periods=21)
        .std(ddof=1)
        .div(
            log_returns["QQQ"]
            .rolling(21, min_periods=21)
            .std(ddof=1)
            .clip(lower=1e-8)
        )
        .shift(1)
    )
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def forward_relative_log_return(
    closes: pd.DataFrame,
    horizon: int = FORWARD_DAYS,
) -> pd.Series:
    selected = closes[["QQQ", "SEMIS"]].astype(float)
    result = (
        np.log(selected["SEMIS"].shift(-horizon) / selected["SEMIS"])
        - np.log(selected["QQQ"].shift(-horizon) / selected["QQQ"])
    )
    result.name = "forward_relative_log_return_21d"
    return result


def benchmark_scores(closes: pd.DataFrame) -> pd.DataFrame:
    selected = closes[["QQQ", "SEMIS"]].astype(float)
    relative = np.log(selected["SEMIS"].div(selected["SEMIS"].shift(1))) - np.log(
        selected["QQQ"].div(selected["QQQ"].shift(1))
    )
    return pd.DataFrame(
        {
            "r39_relative_damage_score": -relative.rolling(
                21, min_periods=21
            ).sum().shift(1),
            "r38_relative_momentum_weakness_score": -relative.rolling(
                126, min_periods=126
            ).sum().shift(1),
        },
        index=selected.index,
    )


def walk_forward_weak_probability(
    closes: pd.DataFrame,
    *,
    training_days: int = TRAINING_DAYS,
    refit_days: int = REFIT_DAYS,
    horizon: int = FORWARD_DAYS,
    seeds: tuple[int, ...] = SEEDS,
    iterations: int = 150,
) -> pd.DataFrame:
    if training_days <= horizon + 20:
        raise ValueError("training_days is too short for mature outcomes")
    if refit_days < 1 or horizon < 1:
        raise ValueError("refit_days and horizon must be positive")
    features = build_causal_features(closes)
    outcomes = forward_relative_log_return(closes, horizon)
    scores = benchmark_scores(closes)
    rows: list[dict[str, object]] = []
    for position in range(training_days - 1, len(features), refit_days):
        signal_date = features.index[position]
        history = features.iloc[position - training_days + 1 : position + 1]
        scaler = StandardScaler().fit(history)
        observations = scaler.transform(history)
        matured = outcomes.reindex(history.index).copy()
        # At the target for signal_date, only the previous close is known.
        # The final horizon+1 labels therefore have endpoints not yet known.
        matured.iloc[-(horizon + 1) :] = np.nan
        seed_probabilities: list[float] = []
        seed_state_gaps: list[float] = []
        for seed in seeds:
            model = GaussianHMM(
                n_components=2,
                covariance_type="diag",
                n_iter=iterations,
                min_covar=1e-5,
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(observations)
            posterior = model.predict_proba(observations)
            known = matured.notna().to_numpy()
            state_means: list[float] = []
            for state in range(2):
                weights = posterior[known, state]
                values = matured.to_numpy(dtype=float)[known]
                state_means.append(
                    float(np.average(values, weights=weights))
                    if float(weights.sum()) > 1e-8
                    else np.nan
                )
            if not bool(np.isfinite(state_means).all()):
                continue
            weak_state = int(np.argmin(state_means))
            seed_probabilities.append(float(posterior[-1, weak_state]))
            seed_state_gaps.append(float(abs(state_means[1] - state_means[0])))
        if not seed_probabilities:
            continue
        current_scores = scores.reindex([signal_date]).iloc[0]
        rows.append(
            {
                "date": signal_date,
                "weak_probability": float(np.mean(seed_probabilities)),
                "seed_probability_std": float(np.std(seed_probabilities)),
                "state_forward_return_gap": float(np.mean(seed_state_gaps)),
                "seed_count": len(seed_probabilities),
                "forward_relative_log_return_21d": outcomes.loc[signal_date],
                "underperformed_next_21d": bool(outcomes.loc[signal_date] < 0.0)
                if pd.notna(outcomes.loc[signal_date])
                else np.nan,
                **current_scores.to_dict(),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def predictive_metrics(
    predictions: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, float | int]:
    frame = predictions.loc[start:end].dropna(
        subset=["forward_relative_log_return_21d"]
    )
    labels = frame["underperformed_next_21d"].astype(int)
    result: dict[str, float | int] = {"observations": len(frame)}
    if len(frame) == 0 or labels.nunique() < 2:
        return {
            **result,
            "hmm_auc": np.nan,
            "hmm_brier": np.nan,
            "hmm_spearman": np.nan,
            "r39_auc": np.nan,
            "r39_spearman": np.nan,
            "r38_auc": np.nan,
            "r38_spearman": np.nan,
        }
    target = -frame["forward_relative_log_return_21d"].astype(float)
    result.update(
        {
            "hmm_auc": float(roc_auc_score(labels, frame["weak_probability"])),
            "hmm_brier": float(
                brier_score_loss(labels, frame["weak_probability"])
            ),
            "hmm_spearman": float(
                spearmanr(frame["weak_probability"], target).statistic
            ),
            "r39_auc": float(
                roc_auc_score(labels, frame["r39_relative_damage_score"])
            ),
            "r39_spearman": float(
                spearmanr(frame["r39_relative_damage_score"], target).statistic
            ),
            "r38_auc": float(
                roc_auc_score(
                    labels,
                    frame["r38_relative_momentum_weakness_score"],
                )
            ),
            "r38_spearman": float(
                spearmanr(
                    frame["r38_relative_momentum_weakness_score"], target
                ).statistic
            ),
        }
    )
    return result


def apply_weak_state_cap(
    weights: pd.DataFrame,
    execution_daily: pd.DataFrame,
    weak_probability: pd.Series,
    *,
    threshold: float = WEAK_PROBABILITY_THRESHOLD,
    maximum_semis_growth_share: float = MAXIMUM_SEMIS_GROWTH_SHARE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = weights.index.intersection(execution_daily.index)
    probability = weak_probability.reindex(index).ffill()
    adjusted = weights.loc[index].copy()
    growth = adjusted["QQQ"] + adjusted["SEMIS"]
    maximum_semis = maximum_semis_growth_share * growth
    active = probability.ge(threshold).fillna(False)
    removed = (adjusted["SEMIS"] - maximum_semis).clip(lower=0.0).where(
        active, 0.0
    )
    adjusted["SEMIS"] -= removed
    adjusted["QQQ"] += removed
    changed = removed.gt(TRADING_EPSILON)
    daily = execution_daily.loc[index].copy()
    daily.loc[changed, "turnover"] = np.maximum(
        daily.loc[changed, "turnover"].to_numpy(dtype=float), 1e-12
    )
    diagnostics = pd.DataFrame(
        {
            "weak_probability": probability,
            "weak_state_cap_active": active,
            "semis_weight_removed": removed,
            "growth_budget_error": (
                adjusted["QQQ"] + adjusted["SEMIS"] - growth
            ).abs(),
        },
        index=index,
    )
    return adjusted, daily, diagnostics


def with_research_strategy_inputs(
    settings: dict[str, object], directory: Path
) -> dict[str, object]:
    source = str(directory.relative_to("output"))
    weights, daily = load_strategy_inputs(source)
    updated = deepcopy(settings)
    updated["weights"] = weights
    updated["daily"] = daily
    updated["directory"] = directory.name
    return updated


def r39_targets(
    settings: dict[str, object], scenario: object
) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = settings["closes"]
    assert isinstance(closes, pd.DataFrame)
    weights, daily, r38_diagnostics = r39._staged_inputs(settings, scenario)
    adjusted, adjusted_daily, _ = r39.apply_relative_damage_concentration_veto(
        weights,
        daily,
        closes,
        maximum_share_permission=(
            ~r38_diagnostics["volatility_acceleration_block"]
            .fillna(False)
            .astype(bool)
        ),
    )
    return adjusted, adjusted_daily


def bootstrap_positive_probability(
    relative_log_returns: pd.Series,
    *,
    block_days: int = 21,
    replications: int = 5000,
    seed: int = 20260731,
) -> float:
    values = relative_log_returns.dropna().to_numpy(dtype=float)
    if len(values) < block_days:
        return np.nan
    extended = np.concatenate([values, values[: block_days - 1]])
    block_sums = np.convolve(
        extended, np.ones(block_days, dtype=float), mode="valid"
    )[: len(values)]
    blocks_per_replication = max(1, len(values) // block_days)
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0,
        len(block_sums),
        size=(replications, blocks_per_replication),
    )
    return float((block_sums[starts].sum(axis=1) > 0.0).mean())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_samples = r21._build_samples()
    definitions = {
        "normal_synthetic": with_research_strategy_inputs(
            base_samples["normal_synthetic"], NORMAL_DIRECTORY
        ),
        "proxy_synthetic": with_research_strategy_inputs(
            base_samples["proxy_synthetic"], PROXY_DIRECTORY
        ),
    }
    predictions_by_sample: dict[str, pd.DataFrame] = {}
    predictive_rows: list[dict[str, object]] = []
    economic_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for sample, settings in definitions.items():
        closes = settings["closes"]
        periods = settings["periods"]
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)
        predictions = walk_forward_weak_probability(closes)
        predictions_by_sample[sample] = predictions
        predictions.to_csv(OUTPUT / f"{sample}_predictions.csv")
        for period, (start, end) in periods.items():
            predictive_rows.append(
                {
                    "sample": sample,
                    "period": period,
                    **predictive_metrics(predictions, start, end),
                }
            )
        for scenario in COST_SCENARIOS:
            baseline_weights, baseline_daily = r39_targets(settings, scenario)
            baseline = r39._simulate_account(
                settings, scenario, baseline_weights, baseline_daily
            )
            candidate_weights, candidate_daily, diagnostics = apply_weak_state_cap(
                baseline_weights,
                baseline_daily,
                predictions["weak_probability"],
            )
            candidate = r39._simulate_account(
                settings, scenario, candidate_weights, candidate_daily
            )
            common = baseline.index.intersection(candidate.index)
            for period, (start, end) in periods.items():
                selected = common[(common >= start) & (common <= end)]
                economic_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "period": period,
                        **metric_delta(
                            baseline.loc[selected, "net_return"],
                            candidate.loc[selected, "net_return"],
                        ),
                    }
                )
                if scenario.name == "current_liquidity" and len(selected):
                    relative = np.log1p(
                        candidate.loc[selected, "net_return"]
                    ) - np.log1p(baseline.loc[selected, "net_return"])
                    bootstrap_rows.append(
                        {
                            "sample": sample,
                            "period": period,
                            "positive_relative_return_probability": (
                                bootstrap_positive_probability(relative)
                            ),
                        }
                    )
            if scenario.name == "current_liquidity":
                baseline.to_csv(OUTPUT / f"{sample}_r39_baseline_daily.csv")
                candidate.to_csv(OUTPUT / f"{sample}_candidate_daily.csv")
                diagnostics.to_csv(OUTPUT / f"{sample}_cap_diagnostics.csv")

    predictive = pd.DataFrame(predictive_rows)
    economics = pd.DataFrame(economic_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    predictive.to_csv(OUTPUT / "predictive_metrics.csv", index=False)
    economics.to_csv(OUTPUT / "economic_metrics.csv", index=False)
    bootstraps.to_csv(OUTPUT / "bootstrap_metrics.csv", index=False)

    def predictive_gate(sample: str, period: str) -> bool:
        row = predictive.loc[
            predictive["sample"].eq(sample)
            & predictive["period"].eq(period)
        ].iloc[0]
        return bool(
            row["hmm_auc"] >= max(row["r39_auc"], row["r38_auc"])
        )

    def economic_gate(sample: str, period: str) -> bool:
        row = economics.loc[
            economics["sample"].eq(sample)
            & economics["scenario"].eq("current_liquidity")
            & economics["period"].eq(period)
        ].iloc[0]
        return bool(row["cagr_delta"] >= 0.0)

    def bootstrap_gate(sample: str, period: str) -> bool:
        row = bootstraps.loc[
            bootstraps["sample"].eq(sample)
            & bootstraps["period"].eq(period)
        ].iloc[0]
        return bool(row["positive_relative_return_probability"] >= 0.90)

    gates = {
        "development_auc_beats_existing_scores": predictive_gate(
            "normal_synthetic", "development_2015_2021"
        ),
        "holdout_auc_beats_existing_scores": predictive_gate(
            "normal_synthetic", "holdout_2022_2025"
        ),
        "proxy_early_auc_beats_existing_scores": predictive_gate(
            "proxy_synthetic", "early_2006_2014"
        ),
        "proxy_late_auc_beats_existing_scores": predictive_gate(
            "proxy_synthetic", "late_2015_2026"
        ),
        "development_cagr_nonnegative": economic_gate(
            "normal_synthetic", "development_2015_2021"
        ),
        "holdout_cagr_nonnegative": economic_gate(
            "normal_synthetic", "holdout_2022_2025"
        ),
        "proxy_cagr_nonnegative": economic_gate(
            "proxy_synthetic", "complete_2006_2026"
        ),
        "development_bootstrap_probability_at_least_90pct": bootstrap_gate(
            "normal_synthetic", "development_2015_2021"
        ),
        "holdout_bootstrap_probability_at_least_90pct": bootstrap_gate(
            "normal_synthetic", "holdout_2022_2025"
        ),
        "proxy_bootstrap_probability_at_least_90pct": bootstrap_gate(
            "proxy_synthetic", "complete_2006_2026"
        ),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only",
        "production_changed": False,
        "orders_generated": False,
        "definition": {
            "states": 2,
            "seeds": list(SEEDS),
            "training_days": TRAINING_DAYS,
            "refit_days": REFIT_DAYS,
            "forward_days": FORWARD_DAYS,
            "weak_probability_threshold": WEAK_PROBABILITY_THRESHOLD,
            "maximum_semis_growth_share": MAXIMUM_SEMIS_GROWTH_SHARE,
        },
        "gates": gates,
        "research_pass": bool(all(gates.values())),
        "decision": (
            "continue_forward_shadow"
            if all(gates.values())
            else "reject_current_specification_and_retain_r39"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
